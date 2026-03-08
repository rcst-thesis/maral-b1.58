package com.rcst;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import com.rcst.layers.BitDecoder;
import com.rcst.layers.BitEncoder;
import com.rcst.layers.BitLinear;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Stream;

/**
 * Training loop for maral-b1.58.
 *
 * ── Checkpointing ────────────────────────────────────────────────────────────
 * After every N epochs (configured by checkpoint.save_every_n_epochs) the
 * model is flushed to disk at:
 *
 *   checkpoints/epoch-NNN/
 *     src-embed-0000.params
 *     tgt-embed-0000.params
 *     encoder-0000.params
 *     decoder-0000.params
 *     out-proj-0000.params
 *     training-state.txt        ← globalStep + last train/val loss
 *
 * Old checkpoints beyond keep_last_n are deleted automatically.
 *
 * To resume from a checkpoint set resume_from in model-config.yaml:
 *   resume_from: "checkpoints/epoch-005"
 *
 * ── Memory management (DJL issue #2210) ──────────────────────────────────────
 * - BitLinear scopes all weight-derived intermediates in a per-forward wScope.
 * - trainStep uses a stepMgr sub-manager; closing it frees all step tensors.
 * - evaluate runs on GPU (same device as model) in a evalMgr without a
 *   GradientCollector — no device mismatch, no gradient accumulation.
 */
public class Trainer {

    private static final int PAD_ID = 0;
    private static final int BOS_ID = 2;
    private static final float BETA1 = 0.9f;
    private static final float BETA2 = 0.999f;
    private static final float ADAM_EPS = 1e-8f;

    private final ModelConfig cfg;
    private final NDManager manager;
    private final Tokenizer tokenizer;

    private final TokenEmbeddingTable srcEmbed;
    private final TokenEmbeddingTable tgtEmbed;
    private final BitEncoder encoder;
    private final BitDecoder decoder;
    private final BitLinear outProj;

    // Adam state: per-parameter first (m) and second (v) moment vectors
    private final List<Parameter> params = new ArrayList<>();
    private final Map<String, NDArray[]> adamState = new HashMap<>();

    // Training state (persisted in training-state.txt)
    private int globalStep = 0;
    private int startEpoch = 1; // updated when resuming
    private float lastValLoss = Float.MAX_VALUE;

    // Rolling queue of saved checkpoint paths for keep_last_n eviction
    private final Deque<Path> checkpointQueue = new ArrayDeque<>();

    // ── Construction ─────────────────────────────────────────────────────────

    public Trainer() throws Exception {
        this.cfg = ModelConfig.get();
        this.manager = NDManager.newBaseManager();
        System.out.printf("Device: %s%n", manager.getDevice());
        this.tokenizer = new Tokenizer();

        Shape embedInput = new Shape(cfg.batchSize, cfg.maxSeqLen);
        Shape seqShape = new Shape(cfg.batchSize, cfg.maxSeqLen, cfg.dModel);

        srcEmbed = new TokenEmbeddingTable(cfg.vocabSize, cfg.dModel);
        tgtEmbed = new TokenEmbeddingTable(cfg.vocabSize, cfg.dModel);
        encoder = new BitEncoder(
            cfg.nEncoderLayers,
            cfg.dModel,
            cfg.nHeads,
            cfg.dFfn,
            cfg.ropeBase,
            cfg.maxSeqLen,
            cfg.eps,
            cfg.quantEps
        );
        decoder = new BitDecoder(
            cfg.nDecoderLayers,
            cfg.dModel,
            cfg.nHeads,
            cfg.dFfn,
            cfg.ropeBase,
            cfg.maxSeqLen,
            cfg.eps,
            cfg.quantEps
        );
        outProj = new BitLinear(cfg.dModel, cfg.vocabSize, cfg.quantEps);

        srcEmbed.initialize(manager, DataType.FLOAT32, embedInput);
        tgtEmbed.initialize(manager, DataType.FLOAT32, embedInput);
        encoder.initialize(manager, DataType.FLOAT32, seqShape);
        decoder.initialize(manager, DataType.FLOAT32, seqShape, seqShape);
        outProj.initialize(manager, DataType.FLOAT32, seqShape);

        collectParameters(srcEmbed);
        collectParameters(tgtEmbed);
        collectParameters(encoder);
        collectParameters(decoder);
        collectParameters(outProj);

        System.out.printf(
            "Model initialised — %d parameter tensors%n",
            params.size()
        );

        // Resume from checkpoint if configured
        if (!cfg.resumeFrom.isEmpty()) {
            loadCheckpoint(Paths.get(cfg.resumeFrom));
        }
    }

    // ── Public entry point ────────────────────────────────────────────────────

    public void train() throws Exception {
        List<long[][]> all = loadPairs();
        int nTrain = Math.max(
            1,
            Math.min((int) (all.size() * cfg.trainRatio), all.size() - 1)
        );
        List<long[][]> trainSet = new ArrayList<>(all.subList(0, nTrain));
        List<long[][]> valSet = new ArrayList<>(
            all.subList(nTrain, all.size())
        );
        Random rng = new Random(cfg.seed);

        System.out.printf(
            "train=%d  val=%d  epochs=%d  startEpoch=%d%n",
            trainSet.size(),
            valSet.size(),
            cfg.maxEpochs,
            startEpoch
        );

        int accumSteps = 4;

        for (int epoch = startEpoch; epoch <= cfg.maxEpochs; epoch++) {
            Collections.shuffle(trainSet, rng);
            float totalLoss = 0f;
            int nBatches = 0;
            int steps = Math.max(
                1,
                (int) Math.ceil((double) trainSet.size() / cfg.batchSize)
            );

            for (int i = 0; i < steps; i++) {
                totalLoss += trainStep(trainSet, i * cfg.batchSize);
                nBatches++;
                globalStep++;

                if ((i + 1) % accumSteps == 0 || i == steps - 1) {
                    clipGradients();
                    adamUpdate();
                    zeroGradients();
                }
            }

            // Validation: GPU, no gradients, isolated sub-manager
            float valLoss = 0f;
            if (epoch % 5 == 0 || epoch == 1) {
                valLoss = evaluate(valSet);
                lastValLoss = valLoss;
            }

            System.out.printf(
                "epoch %3d  train=%.4f  val=%.4f%n",
                epoch,
                totalLoss / nBatches,
                valLoss
            );

            // ── Checkpoint ───────────────────────────────────────────────────
            if (epoch % cfg.saveEveryNEpochs == 0) {
                saveCheckpoint(epoch, totalLoss / nBatches, valLoss);
            }

            // Nudge Java GC and hint PyTorch to release cached-but-free CUDA blocks.
            // This won't fix leaks, but keeps fragmentation from compounding each epoch.
            System.gc();
        }
    }

    // ── Training step ─────────────────────────────────────────────────────────

    private float trainStep(List<long[][]> data, int offset) {
        long[][][] batch = prepareBatch(data, offset);
        float lossVal;

        try (NDManager stepMgr = manager.newSubManager()) {
            NDArray srcIds = stepMgr.create(batch[0]);
            NDArray tgtIn = stepMgr.create(batch[1]);
            NDArray tgtOut = stepMgr.create(batch[2]);

            ParameterStore ps = new ParameterStore(stepMgr, false);

            try (
                GradientCollector gc =
                    Engine.getInstance().newGradientCollector()
            ) {
                NDArray logits = forward(ps, srcIds, tgtIn, true);
                assertNoNaN("logits", logits);
                NDArray loss = crossEntropyLoss(logits, tgtOut);
                lossVal = loss.getFloat();
                assertNoNaN("loss", loss);
                gc.backward(loss);
            }
        }

        return lossVal;
    }

    // ── Validation ────────────────────────────────────────────────────────────

    private float evaluate(List<long[][]> valSet) {
        long[][][] batch = prepareBatch(valSet, 0);

        try (NDManager evalMgr = manager.newSubManager()) {
            NDArray srcIds = evalMgr.create(batch[0]);
            NDArray tgtIn = evalMgr.create(batch[1]);
            NDArray tgtOut = evalMgr.create(batch[2]);

            ParameterStore ps = new ParameterStore(evalMgr, false);
            NDArray logits = forward(ps, srcIds, tgtIn, false);
            return crossEntropyLoss(logits, tgtOut).getFloat();
        }
    }

    // ── Forward pass ──────────────────────────────────────────────────────────

    private NDArray forward(
        ParameterStore ps,
        NDArray srcIds,
        NDArray tgtIn,
        boolean training
    ) {
        NDArray srcEmb = srcEmbed
            .forward(ps, new NDList(srcIds), training)
            .singletonOrThrow();
        assertNoNaN("srcEmb", srcEmb);
        NDArray memory = encoder
            .forward(ps, new NDList(srcEmb), training)
            .singletonOrThrow();
        assertNoNaN("memory", memory);
        NDArray tgtEmb = tgtEmbed
            .forward(ps, new NDList(tgtIn), training)
            .singletonOrThrow();
        assertNoNaN("tgtEmb", tgtEmb);
        NDArray decoded = decoder
            .forward(ps, new NDList(tgtEmb, memory), training)
            .singletonOrThrow();
        assertNoNaN("decoded", decoded);
        return outProj
            .forward(ps, new NDList(decoded), training)
            .singletonOrThrow();
    }

    // ── Checkpoint save ───────────────────────────────────────────────────────

    /**
     * Saves all five model components as DJL .params files inside a
     * per-epoch sub-directory, then writes a training-state.txt and
     * evicts the oldest checkpoint if the queue exceeds keepLastN.
     *
     * Directory layout:
     *   checkpoints/epoch-005/
     *     src-embed-0000.params
     *     tgt-embed-0000.params
     *     encoder-0000.params
     *     decoder-0000.params
     *     out-proj-0000.params
     *     training-state.txt
     */
    private void saveCheckpoint(int epoch, float trainLoss, float valLoss)
        throws IOException {
        String folderName = String.format("epoch-%03d", epoch);
        Path ckptPath = Paths.get(cfg.checkpointDir, folderName);
        Files.createDirectories(ckptPath);

        saveBlock(srcEmbed, ckptPath, "src-embed");
        saveBlock(tgtEmbed, ckptPath, "tgt-embed");
        saveBlock(encoder, ckptPath, "encoder");
        saveBlock(decoder, ckptPath, "decoder");
        saveBlock(outProj, ckptPath, "out-proj");

        // Training state — plain text, one key=value per line
        Path statePath = ckptPath.resolve("training-state.txt");
        try (BufferedWriter w = Files.newBufferedWriter(statePath)) {
            w.write("globalStep=" + globalStep);
            w.newLine();
            w.write("epoch=" + epoch);
            w.newLine();
            w.write("trainLoss=" + trainLoss);
            w.newLine();
            w.write("valLoss=" + valLoss);
            w.newLine();
        }

        System.out.printf("  ✓ checkpoint saved → %s%n", ckptPath);

        // Rolling eviction: remove oldest when queue exceeds keepLastN
        checkpointQueue.addLast(ckptPath);
        while (checkpointQueue.size() > cfg.keepLastN) {
            Path old = checkpointQueue.removeFirst();
            deleteDirectory(old);
            System.out.printf("  ✗ evicted old checkpoint: %s%n", old);
        }
    }

    /** Use DJL Model.save() to write a single block's .params file. */
    private void saveBlock(AbstractBlock block, Path dir, String name)
        throws IOException {
        try (Model m = Model.newInstance(name)) {
            m.setBlock(block);
            m.save(dir, name);
        }
    }

    // ── Checkpoint load / resume ──────────────────────────────────────────────

    /**
     * Loads all five components from a checkpoint directory and restores
     * globalStep / startEpoch from training-state.txt.
     */
    private void loadCheckpoint(Path ckptPath)
        throws IOException, MalformedModelException {
        System.out.printf("Resuming from checkpoint: %s%n", ckptPath);

        loadBlock(srcEmbed, ckptPath, "src-embed");
        loadBlock(tgtEmbed, ckptPath, "tgt-embed");
        loadBlock(encoder, ckptPath, "encoder");
        loadBlock(decoder, ckptPath, "decoder");
        loadBlock(outProj, ckptPath, "out-proj");

        Path statePath = ckptPath.resolve("training-state.txt");
        if (Files.exists(statePath)) {
            try (BufferedReader r = Files.newBufferedReader(statePath)) {
                String line;
                while ((line = r.readLine()) != null) {
                    String[] parts = line.split("=", 2);
                    if (parts.length < 2) continue;
                    switch (parts[0].trim()) {
                        case "globalStep":
                            globalStep = Integer.parseInt(parts[1].trim());
                        case "epoch":
                            startEpoch = Integer.parseInt(parts[1].trim()) + 1;
                        case "valLoss":
                            lastValLoss = Float.parseFloat(parts[1].trim());
                    }
                }
            }
            System.out.printf(
                "  Resumed: globalStep=%d  nextEpoch=%d  lastValLoss=%.4f%n",
                globalStep,
                startEpoch,
                lastValLoss
            );
        }
    }

    /** Use DJL Model.load() to restore a single block's parameters. */
    private void loadBlock(AbstractBlock block, Path dir, String name)
        throws IOException, MalformedModelException {
        try (Model m = Model.newInstance(name)) {
            m.setBlock(block);
            m.load(dir, name);
        }
    }

    // ── Loss ──────────────────────────────────────────────────────────────────

    private NDArray crossEntropyLoss(NDArray logits, NDArray targets) {
        long B = logits.getShape().get(0);
        long T = logits.getShape().get(1);

        NDArray logProbs = logits.logSoftmax(-1);
        NDArray idx = targets.reshape(B, T, 1).toType(DataType.INT64, false);
        NDArray nll = logProbs.gather(idx, 2).squeeze(2).neg();
        NDArray mask = targets.neq(PAD_ID).toType(DataType.FLOAT32, false);
        NDArray maskSum = mask.sum().maximum(1e-6f);
        return nll.mul(mask).sum().div(maskSum);
    }

    // ── Gradient utilities ────────────────────────────────────────────────────

    private void zeroGradients() {
        for (Parameter p : params) {
            NDArray grad = p.getArray().getGradient();
            if (grad == null) continue;
            grad.subi(grad); // zero in-place
            grad.close(); // release the wrapper — prevents accumulation across steps
        }
    }

    private void clipGradients() {
        float totalSq = 0f;
        for (Parameter p : params) {
            NDArray g = p.getArray().getGradient();
            if (g == null) continue;
            try (NDArray sq = g.pow(2); NDArray s = sq.sum()) {
                totalSq += s.getFloat();
            }
            g.close(); // ← close the norm-pass wrapper
        }
        float scale = cfg.gradClip / ((float) Math.sqrt(totalSq) + 1e-6f);
        if (scale < 1f) {
            for (Parameter p : params) {
                NDArray g = p.getArray().getGradient();
                if (g == null) continue;
                g.muli(scale);
                g.close(); // ← close the scale-pass wrapper
            }
        }
    }

    // ── Adam ──────────────────────────────────────────────────────────────────

    private void adamUpdate() {
        int t = globalStep + 1;
        float warmup = Math.min(1f, (float) t / Math.max(cfg.warmupSteps, 1));
        float biasCorr = (float) (Math.sqrt(1.0 - Math.pow(BETA2, t)) /
            (1.0 - Math.pow(BETA1, t)));
        float lrT = cfg.learningRate * warmup * biasCorr;

        for (Parameter p : params) {
            NDArray weight = p.getArray();
            NDArray grad = weight.getGradient();
            if (grad == null) continue;

            String key = Integer.toString(System.identityHashCode(p));
            NDArray[] mv = adamState.get(key);
            NDArray m = mv[0];
            NDArray v = mv[1];

            try (NDArray gm = grad.mul(1f - BETA1)) {
                m.muli(BETA1).addi(gm);
            }
            try (
                NDArray gsq = grad.square();
                NDArray gv = gsq.mul(1f - BETA2)
            ) {
                v.muli(BETA2).addi(gv);
            }
            try (
                NDArray mLr = m.mul(lrT);
                NDArray vSqrt = v.sqrt();
                NDArray denom = vSqrt.add(ADAM_EPS);
                NDArray step = mLr.div(denom)
            ) {
                weight.subi(step);
            }

            grad.close(); // ← CRITICAL: release gradient wrapper after each param update
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private void collectParameters(AbstractBlock block) {
        block
            .getParameters()
            .values()
            .forEach(p -> {
                NDArray arr = p.getArray();
                arr.setRequiresGradient(true);
                params.add(p);
                String key = Integer.toString(System.identityHashCode(p));
                adamState.put(
                    key,
                    new NDArray[] {
                        manager.zeros(arr.getShape(), DataType.FLOAT32),
                        manager.zeros(arr.getShape(), DataType.FLOAT32),
                    }
                );
            });
    }

    private static void assertNoNaN(String stage, NDArray x) {
        try (NDArray nanMask = x.isNaN(); NDArray any = nanMask.any()) {
            if (!any.getBoolean()) return;
        }
        try (NDArray flat = x.flatten()) {
            float[] sample = flat.toFloatArray();
            int show = Math.min(8, sample.length);
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < show; i++) sb.append(sample[i]).append(' ');
            System.err.printf(
                "NaN at stage=%s  shape=%s  sample=[%s]%n",
                stage,
                x.getShape(),
                sb
            );
        }
    }

    private long[][][] prepareBatch(List<long[][]> data, int offset) {
        int T = cfg.maxSeqLen;
        long[][] src = new long[cfg.batchSize][T];
        long[][] tin = new long[cfg.batchSize][T];
        long[][] tout = new long[cfg.batchSize][T];

        for (int b = 0; b < cfg.batchSize; b++) {
            long[] s = data.get((offset + b) % data.size())[0];
            long[] tgt = data.get((offset + b) % data.size())[1];
            for (int t = 0; t < T; t++) {
                src[b][t] = t < s.length ? s[t] : PAD_ID;
                tin[b][t] =
                    t == 0 ? BOS_ID : t - 1 < tgt.length ? tgt[t - 1] : PAD_ID;
                tout[b][t] = t < tgt.length ? tgt[t] : PAD_ID;
            }
        }
        return new long[][][] { src, tin, tout };
    }

    private List<long[][]> loadPairs() throws IOException {
        List<long[][]> pairs = new ArrayList<>();
        try (
            BufferedReader br = Files.newBufferedReader(
                Paths.get(cfg.parallelPath)
            )
        ) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] cols = line.split("\t", 2);
                if (cols.length < 2) continue;
                List<Integer> src = tokenizer.encode(cols[0].trim());
                List<Integer> tgt = tokenizer.encode(cols[1].trim());
                if (src.isEmpty() || tgt.isEmpty()) continue;
                pairs.add(
                    new long[][] {
                        src.stream().mapToLong(Integer::longValue).toArray(),
                        tgt.stream().mapToLong(Integer::longValue).toArray(),
                    }
                );
            }
        }
        System.out.printf(
            "Loaded %d sentence pairs from %s%n",
            pairs.size(),
            cfg.parallelPath
        );
        return pairs;
    }

    /** Recursively delete a checkpoint directory. */
    private static void deleteDirectory(Path dir) throws IOException {
        if (!Files.exists(dir)) return;
        try (Stream<Path> stream = Files.walk(dir)) {
            stream
                .sorted(java.util.Comparator.reverseOrder())
                .forEach(p -> {
                    try {
                        Files.delete(p);
                    } catch (IOException ignored) {}
                });
        }
    }

    // ── Main ──────────────────────────────────────────────────────────────────

    public static void main(String[] args) throws Exception {
        int totalCores = Runtime.getRuntime().availableProcessors();
        int usableCores = Math.max(1, (int) (totalCores * 0.8));
        System.setProperty(
            "ai.djl.pytorch.num_interop_threads",
            String.valueOf(usableCores)
        );
        System.setProperty(
            "ai.djl.pytorch.num_threads",
            String.valueOf(usableCores)
        );

        // Reduce CUDA allocator fragmentation — avoids OOM from allocator holding
        // freed blocks in rigid-size buckets. Set before PyTorch engine initialises.
        System.setProperty(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True"
        );

        long totalRam = Runtime.getRuntime().maxMemory();
        System.out.printf(
            "Using %d / %d cores  |  JVM heap %.1f GiB%n",
            usableCores,
            totalCores,
            totalRam / 1073741824.0
        );

        new Trainer().train();
    }
}
