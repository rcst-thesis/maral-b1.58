package com.rcst;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.nn.Parameter;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.util.cuda.CudaUtils;
import com.rcst.utils.ModelConfig;
import com.sentencepiece.Scoring;
import com.sentencepiece.SentencePieceAlgorithm;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.lang.management.MemoryUsage; // ADD THIS IMPORT
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Training loop for maral-b1.58 with comprehensive debugging.
 */
public class Trainer implements AutoCloseable {

    private static final int PAD_ID = 0;
    private static final int BOS_ID = 2;
    private static final float BETA1 = 0.9f;
    private static final float BETA2 = 0.999f;
    private static final float ADAM_EPS = 1e-8f;

    private final ModelConfig cfg;
    private final Model model;
    private final NDManager manager;

    // SPM for data loading
    private final com.sentencepiece.Model spm;
    private final SentencePieceAlgorithm spmAlgo;

    private final List<long[][]> trainSet = new ArrayList<>();
    private final List<long[][]> valSet = new ArrayList<>();
    private final Map<String, NDArray[]> adamState = new HashMap<>();

    private int globalStep = 0;
    private int startEpoch = 1;
    private float bestValLoss = Float.MAX_VALUE;

    public Trainer() throws Exception {
        this.cfg = ModelConfig.get();
        this.model = new Model();
        this.manager = model.getManager();

        Device device = manager.getDevice();
        long mem = CudaUtils.getGpuMemory(device).getMax();
        System.out.printf(
            "Device: %s  GPU memory: %d MB%n",
            device,
            mem / (1024 * 1024)
        );

        // Load SPM for data loading
        this.spm = com.sentencepiece.Model.parseFrom(
            Paths.get(cfg.tokenizerModelPath)
        );
        this.spmAlgo = new SentencePieceAlgorithm(true, Scoring.HIGHEST_SCORE);

        // Load data
        loadData();

        // Initialize optimizer state
        System.out.println("[INIT] Initializing Adam state tensors...");
        int paramCount = 0;
        long totalParams = 0;
        for (Parameter p : model.getParameterList()) {
            NDArray arr = p.getArray();
            arr.setRequiresGradient(true);
            String key = Integer.toString(System.identityHashCode(p));
            adamState.put(
                key,
                new NDArray[] {
                    manager.zeros(arr.getShape(), DataType.FLOAT32),
                    manager.zeros(arr.getShape(), DataType.FLOAT32),
                }
            );
            long size = arr.getShape().size();
            totalParams += size;
            paramCount++;
            System.out.printf(
                "[INIT] Param %d: shape=%s, size=%d%n",
                paramCount,
                arr.getShape(),
                size
            );
        }

        System.out.printf(
            "[INIT] Total parameters: %d tensors, %,d elements%n",
            paramCount,
            totalParams
        );
        printMemoryStats("POST_INIT");

        // Resume if specified
        if (!cfg.resumeFrom.isEmpty()) {
            loadCheckpoint(Paths.get(cfg.resumeFrom));
        }
    }

    private void loadData() throws IOException {
        System.out.println("[DATA] Loading parallel corpus...");
        List<long[][]> all = new ArrayList<>();
        try (
            BufferedReader br = Files.newBufferedReader(
                Paths.get(cfg.parallelPath)
            )
        ) {
            String line;
            int lines = 0;
            while ((line = br.readLine()) != null) {
                String[] cols = line.split("\t", 2);
                if (cols.length < 2) continue;
                List<Integer> src = spm.encodeNormalized(
                    cols[0].trim(),
                    spmAlgo
                );
                List<Integer> tgt = spm.encodeNormalized(
                    cols[1].trim(),
                    spmAlgo
                );
                if (src.isEmpty() || tgt.isEmpty()) continue;
                all.add(
                    new long[][] {
                        src.stream().mapToLong(Integer::longValue).toArray(),
                        tgt.stream().mapToLong(Integer::longValue).toArray(),
                    }
                );
                lines++;
                if (lines % 1000 == 0) {
                    System.out.printf("[DATA] Loaded %d lines...%n", lines);
                }
            }
        }

        int nTrain = Math.max(
            1,
            Math.min((int) (all.size() * cfg.trainRatio), all.size() - 1)
        );
        Collections.shuffle(all, new Random(cfg.seed));
        trainSet.addAll(all.subList(0, nTrain));
        valSet.addAll(all.subList(nTrain, all.size()));

        System.out.printf(
            "[DATA] Dataset: total=%d, train=%d, val=%d%n",
            all.size(),
            trainSet.size(),
            valSet.size()
        );
    }

    public void train() throws Exception {
        System.out.printf(
            "[TRAIN] Starting training: epochs=%d, startEpoch=%d, batchSize=%d, gradAccum=%d%n",
            cfg.maxEpochs,
            startEpoch,
            cfg.batchSize,
            cfg.gradAccumSteps
        );
        Random rng = new Random(cfg.seed);

        for (int epoch = startEpoch; epoch <= cfg.maxEpochs; epoch++) {
            System.out.printf("%n[EPOCH %d] Starting epoch...%n", epoch);
            printMemoryStats("EPOCH_START");

            Collections.shuffle(trainSet, rng);

            float totalLoss = 0f;
            int nBatches = 0;
            int steps = Math.max(
                1,
                (int) Math.ceil((double) trainSet.size() / cfg.batchSize)
            );
            System.out.printf("[EPOCH %d] Steps per epoch: %d%n", epoch, steps);

            zeroGradients();

            long epochStartTime = System.currentTimeMillis();

            for (int i = 0; i < steps; i++) {
                long stepStartTime = System.currentTimeMillis();

                if (i % 10 == 0) {
                    System.out.printf(
                        "[EPOCH %d] Step %d/%d (%.1f%%)...%n",
                        epoch,
                        i,
                        steps,
                        (100.0 * i) / steps
                    );
                    printMemoryStats("STEP_" + i);
                }

                float loss = trainStep(i * cfg.batchSize);
                totalLoss += loss;
                nBatches++;
                globalStep++;

                long stepTime = System.currentTimeMillis() - stepStartTime;

                if (i % 10 == 0) {
                    System.out.printf(
                        "[EPOCH %d] Step %d loss=%.4f, time=%dms%n",
                        epoch,
                        i,
                        loss,
                        stepTime
                    );
                }

                if ((i + 1) % cfg.gradAccumSteps == 0 || i == steps - 1) {
                    System.out.printf(
                        "[EPOCH %d] Updating weights (step %d)...%n",
                        epoch,
                        i
                    );
                    long updateStart = System.currentTimeMillis();
                    clipAndUpdate();
                    long updateTime = System.currentTimeMillis() - updateStart;
                    System.out.printf(
                        "[EPOCH %d] Weight update took %dms%n",
                        epoch,
                        updateTime
                    );
                    zeroGradients();
                }
            }

            long epochTime = System.currentTimeMillis() - epochStartTime;
            float trainLoss = totalLoss / nBatches;

            System.out.printf(
                "[EPOCH %d] Computing validation loss...%n",
                epoch
            );
            float valLoss = evaluate();

            System.out.printf(
                "[EPOCH %d] Results: train=%.4f, val=%.4f, time=%.1fs%n",
                epoch,
                trainLoss,
                valLoss,
                epochTime / 1000.0
            );
            printMemoryStats("EPOCH_END");

            if (epoch % cfg.saveEveryNEpochs == 0) {
                System.out.printf("[EPOCH %d] Saving checkpoint...%n", epoch);
                saveRollingCheckpoint(epoch, trainLoss, valLoss);
                evictOldCheckpoints();
            }

            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                System.out.printf(
                    "[EPOCH %d] New best validation loss! Saving to best/%n",
                    epoch
                );
                saveCheckpoint(
                    Paths.get(cfg.checkpointDir, "best"),
                    epoch,
                    trainLoss,
                    valLoss
                );
            }

            if (
                cfg.sampleEveryNEpochs > 0 &&
                epoch % cfg.sampleEveryNEpochs == 0
            ) {
                System.out.printf(
                    "[EPOCH %d] Generating sample translation...%n",
                    epoch
                );
                String sample = model.greedyTranslate(
                    "Good morning, how are you?",
                    cfg.maxSeqLen
                );
                System.out.printf("[EPOCH %d] Sample: %s%n", epoch, sample);
            }

            // Force cleanup
            System.out.printf(
                "[EPOCH %d] Forcing garbage collection...%n",
                epoch
            );
            System.gc();
            clearGpuCache();
            printMemoryStats("POST_GC");
        }
    }

    private float trainStep(int offset) {
        long[][][] batch = prepareBatch(trainSet, offset);

        try (NDManager stepMgr = manager.newSubManager()) {
            NDArray srcIds = stepMgr.create(batch[0]);
            NDArray tgtIn = stepMgr.create(batch[1]);
            NDArray tgtOut = stepMgr.create(batch[2]);
            ParameterStore ps = new ParameterStore(stepMgr, false);

            try (
                GradientCollector gc =
                    Engine.getInstance().newGradientCollector()
            ) {
                NDArray logits = model.forward(ps, srcIds, tgtIn, true);
                NDArray loss = crossEntropyLoss(logits, tgtOut);
                float lossVal = loss.getFloat();

                loss.setRequiresGradient(true);
                gc.backward(loss);

                return lossVal;
            }
        }
    }

    private float evaluate() {
        long[][][] batch = prepareBatch(valSet, 0);
        try (NDManager evalMgr = manager.newSubManager()) {
            NDArray srcIds = evalMgr.create(batch[0]);
            NDArray tgtIn = evalMgr.create(batch[1]);
            NDArray tgtOut = evalMgr.create(batch[2]);
            ParameterStore ps = new ParameterStore(evalMgr, false);
            NDArray logits = model.forward(ps, srcIds, tgtIn, false);
            return crossEntropyLoss(logits, tgtOut).getFloat();
        }
    }

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

    private void clipAndUpdate() {
        System.out.println("  [CLIP] Computing gradient norm...");
        printMemoryStats("CLIP_START");

        // Compute gradient norm
        float totalSq = 0f;
        int gradCount = 0;
        for (Parameter p : model.getParameterList()) {
            NDArray g = p.getArray().getGradient();
            if (g == null) continue;

            gradCount++;
            float normSq;
            try (NDArray sq = g.pow(2)) {
                try (NDArray sum = sq.sum()) {
                    normSq = sum.getFloat();
                }
            }
            totalSq += normSq;
        }
        System.out.printf("  [CLIP] Processed %d gradients%n", gradCount);

        float norm = (float) Math.sqrt(totalSq) + 1e-6f;
        float scale = (norm > cfg.gradClip) ? cfg.gradClip / norm : 1f;

        int t = globalStep + 1;
        float warmup = Math.min(1f, (float) t / Math.max(cfg.warmupSteps, 1));
        float biasCorr = (float) (Math.sqrt(1.0 - Math.pow(BETA2, t)) /
            (1.0 - Math.pow(BETA1, t)));
        float lrT = cfg.learningRate * warmup * biasCorr;

        System.out.printf(
            "  [CLIP] gradNorm=%.4f, scale=%.4f, lr=%.6f, warmup=%.4f%n",
            norm,
            scale,
            lrT,
            warmup
        );

        // Adam update
        int updateCount = 0;
        for (Parameter p : model.getParameterList()) {
            NDArray weight = p.getArray();
            NDArray g = weight.getGradient();
            if (g == null) continue;

            if (scale < 1f) {
                g.muli(scale);
            }

            String key = Integer.toString(System.identityHashCode(p));
            NDArray[] mv = adamState.get(key);
            NDArray m = mv[0];
            NDArray v = mv[1];

            // m = beta1 * m + (1 - beta1) * g
            m.muli(BETA1);
            try (NDArray gScaled = g.mul(1f - BETA1)) {
                m.addi(gScaled);
            }

            // v = beta2 * v + (1 - beta2) * g^2
            v.muli(BETA2);
            try (
                NDArray gSq = g.square();
                NDArray gSqScaled = gSq.mul(1f - BETA2)
            ) {
                v.addi(gSqScaled);
            }

            // Update weights
            try (NDArray vSqrt = v.sqrt()) {
                vSqrt.addi(ADAM_EPS);

                try (NDArray step = m.duplicate()) {
                    step.muli(lrT);
                    step.divi(vSqrt);
                    weight.subi(step);
                }
            }

            // Weight decay
            float decayCoef = cfg.weightDecay * cfg.learningRate * warmup;
            weight.muli(1f - decayCoef);

            updateCount++;
        }

        System.out.printf("  [CLIP] Updated %d parameters%n", updateCount);
        printMemoryStats("CLIP_END");
    }

    private void zeroGradients() {
        int count = 0;
        for (Parameter p : model.getParameterList()) {
            NDArray g = p.getArray().getGradient();
            if (g == null) continue;
            g.subi(g);
            count++;
        }
        System.out.printf("  [ZERO] Zeroed %d gradients%n", count);
    }

    private void printMemoryStats(String label) {
        try {
            Device device = manager.getDevice();
            if (device.isGpu()) {
                MemoryUsage mem = CudaUtils.getGpuMemory(device); // Uses java.lang.management.MemoryUsage
                long total = mem.getMax();
                long used = mem.getCommitted();
                long free = total - used;
                float pct = (100f * used) / total;
                System.out.printf(
                    "[MEM %s] GPU: used=%.2f MiB / %.2f MiB (%.1f%%), free=%.2f MiB%n",
                    label,
                    used / 1048576.0,
                    total / 1048576.0,
                    pct,
                    free / 1048576.0
                );
            } else {
                Runtime rt = Runtime.getRuntime();
                long total = rt.totalMemory();
                long free = rt.freeMemory();
                long used = total - free;
                System.out.printf(
                    "[MEM %s] JVM: used=%.2f MiB / %.2f MiB%n",
                    label,
                    used / 1048576.0,
                    total / 1048576.0
                );
            }
        } catch (Exception e) {
            System.out.printf(
                "[MEM %s] Error reading memory: %s%n",
                label,
                e.getMessage()
            );
        }
    }

    private void saveRollingCheckpoint(
        int epoch,
        float trainLoss,
        float valLoss
    ) throws IOException {
        saveCheckpoint(
            Paths.get(cfg.checkpointDir, String.format("epoch-%03d", epoch)),
            epoch,
            trainLoss,
            valLoss
        );
    }

    private void saveCheckpoint(
        Path dir,
        int epoch,
        float trainLoss,
        float valLoss
    ) throws IOException {
        System.out.printf("[SAVE] Saving checkpoint to %s%n", dir);
        model.save(dir);
        try (
            BufferedWriter w = Files.newBufferedWriter(
                dir.resolve("training-state.txt")
            )
        ) {
            w.write("globalStep=" + globalStep);
            w.newLine();
            w.write("epoch=" + epoch);
            w.newLine();
            w.write("trainLoss=" + trainLoss);
            w.newLine();
            w.write("valLoss=" + valLoss);
            w.newLine();
        }
        System.out.println("[SAVE] Checkpoint saved");
    }

    private void evictOldCheckpoints() throws IOException {
        Path ckptDir = Paths.get(cfg.checkpointDir);
        if (!Files.isDirectory(ckptDir)) return;

        List<Path> rolling = Files.list(ckptDir)
            .filter(p -> p.getFileName().toString().matches("epoch-\\d{3}"))
            .sorted(Comparator.comparing(p -> p.getFileName().toString()))
            .collect(Collectors.toList());

        while (rolling.size() > cfg.keepLastN) {
            Path oldest = rolling.remove(0);
            System.out.printf("[CLEANUP] Evicting %s%n", oldest.getFileName());
            try (Stream<Path> stream = Files.walk(oldest)) {
                stream
                    .sorted(Comparator.reverseOrder())
                    .forEach(f -> {
                        try {
                            Files.delete(f);
                        } catch (IOException ignored) {}
                    });
            }
        }
    }

    private void loadCheckpoint(Path ckptPath)
        throws IOException, MalformedModelException {
        System.out.printf("[RESUME] Loading checkpoint from %s%n", ckptPath);
        model.load(ckptPath);

        Path statePath = ckptPath.resolve("training-state.txt");
        if (Files.exists(statePath)) {
            try (BufferedReader r = Files.newBufferedReader(statePath)) {
                String line;
                while ((line = r.readLine()) != null) {
                    String[] parts = line.split("=", 2);
                    if (parts.length < 2) continue;
                    String key = parts[0].trim();
                    String val = parts[1].trim();
                    if ("globalStep".equals(key)) {
                        globalStep = Integer.parseInt(val);
                    } else if ("epoch".equals(key)) {
                        startEpoch = Integer.parseInt(val) + 1;
                    } else if ("valLoss".equals(key)) {
                        bestValLoss = Float.parseFloat(val);
                    }
                }
            }
            System.out.printf(
                "[RESUME] globalStep=%d, nextEpoch=%d, bestValLoss=%.4f%n",
                globalStep,
                startEpoch,
                bestValLoss
            );
        }
    }

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
            "[MAIN] Cores: %d / %d, JVM heap: %.1f GiB%n",
            usableCores,
            totalCores,
            totalRam / 1073741824.0
        );

        try (Trainer trainer = new Trainer()) {
            trainer.train();
        }
        System.out.println("[MAIN] Training complete");
    }

    private void clearGpuCache() {
        try {
            // Try to access PyTorch engine's emptyCache method via reflection
            Class<?> ptEngineClass = Class.forName(
                "ai.djl.pytorch.engine.PtEngine"
            );
            Object engine = ptEngineClass.getMethod("getInstance").invoke(null);
            ptEngineClass.getMethod("emptyCache").invoke(engine);
            System.out.println("  [CACHE] PyTorch cache cleared");
        } catch (Exception e) {
            // Fallback: force System.gc and hope for the best
            System.gc();
        }
    }

    @Override
    public void close() throws Exception {
        System.out.println("[SHUTDOWN] Closing model...");
        model.close();
    }
}
