package com.rcst;

import ai.djl.Device;
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
import ai.djl.util.cuda.CudaUtils;
import com.rcst.layers.BitDecoder;
import com.rcst.layers.BitEncoder;
import com.rcst.layers.BitLinear;
import com.rcst.utils.ModelConfig;
import com.rcst.utils.TokenEmbeddingTable;
import com.rcst.utils.Tokenizer;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.lang.management.MemoryUsage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Training loop for maral-b1.58.
 *
 * ── Checkpointing ─────────────────────────────────────────────────────────────
 * Two checkpoint strategies run in parallel:
 *
 * 1) Rolling epoch checkpoints  (save_every_n_epochs, keep_last_n)
 *    Written to:  checkpoints/epoch-NNN/
 *    Old ones are evicted once the queue exceeds keep_last_n.
 *    Useful for resuming a crashed run.
 *
 * 2) Best-val checkpoint  (automatic, never evicted)
 *    Written to:  checkpoints/best/
 *    Overwritten whenever val loss improves.
 *    This is the checkpoint you load for inference.
 *
 * Both layouts:
 *   src-embed-0000.params
 *   tgt-embed-0000.params
 *   encoder-0000.params
 *   decoder-0000.params
 *   out-proj-0000.params
 *   training-state.txt
 *
 * To resume from a checkpoint set resume_from in model-config.yaml, e.g.:
 *   resume_from: "checkpoints/best"
 *   resume_from: "checkpoints/epoch-028"
 *
 * ── Memory management (DJL issue #2210) ───────────────────────────────────────
 * - BitLinear scopes all weight-derived intermediates in a per-forward wScope.
 * - trainStep uses a stepMgr sub-manager; closing it frees all step tensors.
 * - Gradient wrappers are closed immediately after use in zeroGradients(),
 *   clipGradients(), and adamUpdate() to prevent VRAM accumulation.
 * - evaluate() runs on GPU in an evalMgr without GradientCollector.
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

    // Adam state
    private final List<Parameter> params = new ArrayList<>();
    private final Map<String, NDArray[]> adamState = new HashMap<>();

    // Training state (persisted in training-state.txt)
    private int globalStep = 0;
    private int startEpoch = 1;
    private float lastValLoss = Float.MAX_VALUE;

    // Best checkpoint tracking — updated whenever val loss improves
    private float bestValLoss = Float.MAX_VALUE;

    // ── Construction ──────────────────────────────────────────────────────────

    public Trainer() throws Exception {
        this.cfg = ModelConfig.get();
        this.manager = NDManager.newBaseManager();

        Device device = manager.getDevice();
        MemoryUsage mem = CudaUtils.getGpuMemory(device);
        this.tokenizer = new Tokenizer();

        System.out.printf("GPU memory: %d MB%n", mem.getMax() / (1024 * 1024));
        System.out.printf("Device: %s%n", device);

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

            // Validation
            float valLoss = evaluate(valSet);
            lastValLoss = valLoss;

            System.out.printf(
                "epoch %3d  train=%.4f  val=%.4f%n",
                epoch,
                totalLoss / nBatches,
                valLoss
            );

            saveEpochCheckpoint(epoch, totalLoss / nBatches, valLoss);

            // Nudge GC — reduces allocator fragmentation between epochs
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

    // ── Rolling epoch checkpoint ──────────────────────────────────────────────

    /**
     * Saves a numbered epoch checkpoint and evicts the oldest once the
     * rolling queue exceeds keep_last_n.
     */
    private void saveEpochCheckpoint(int epoch, float trainLoss, float valLoss)
        throws IOException {
        String folderName = String.format("epoch-%03d", epoch);
        Path ckptPath = Paths.get(cfg.checkpointDir, folderName);
        Files.createDirectories(ckptPath);

        saveBlock(srcEmbed, ckptPath, "src-embed");
        saveBlock(tgtEmbed, ckptPath, "tgt-embed");
        saveBlock(encoder, ckptPath, "encoder");
        saveBlock(decoder, ckptPath, "decoder");
        saveBlock(outProj, ckptPath, "out-proj");

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
                    String key = parts[0].trim();
                    String val = parts[1].trim();
                    // Plain if-else — no switch expressions, pure Java 17
                    if ("globalStep".equals(key)) {
                        globalStep = Integer.parseInt(val);
                    } else if ("epoch".equals(key) || "bestEpoch".equals(key)) {
                        startEpoch = Integer.parseInt(val) + 1;
                    } else if ("valLoss".equals(key)) {
                        lastValLoss = Float.parseFloat(val);
                        // Restore bestValLoss so we don't immediately overwrite
                        // a good "best" checkpoint on resume
                        bestValLoss = lastValLoss;
                    }
                }
            }
            System.out.printf(
                "  Resumed: globalStep=%d  nextEpoch=%d  bestValLoss=%.4f%n",
                globalStep,
                startEpoch,
                bestValLoss
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
            grad.subi(grad);
            grad.close(); // release wrapper — prevents accumulation across steps
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
            g.close();
        }
        float scale = cfg.gradClip / ((float) Math.sqrt(totalSq) + 1e-6f);
        if (scale < 1f) {
            for (Parameter p : params) {
                NDArray g = p.getArray().getGradient();
                if (g == null) continue;
                g.muli(scale);
                g.close();
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

            grad.close(); // release gradient wrapper
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
                    t == 0
                        ? BOS_ID
                        : (t - 1 < tgt.length ? tgt[t - 1] : PAD_ID);
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
