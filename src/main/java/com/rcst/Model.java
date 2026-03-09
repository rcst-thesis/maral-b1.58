package com.rcst;

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.rcst.common.RMSNorm;
import com.rcst.layers.BitLinear;
import com.rcst.layers.DecoderBlock;
import com.rcst.layers.Embedder;
import com.rcst.layers.EncoderBlock;
import com.rcst.utils.ModelConfig;
import com.sentencepiece.Scoring;
import com.sentencepiece.SentencePieceAlgorithm;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * BitNet b1.58 Transformer Model for Machine Translation.
 */
public class Model extends AbstractBlock implements AutoCloseable {

    private static final int BOS_ID = 2;
    private static final int EOS_ID = 3;

    private final ModelConfig cfg;
    private final NDManager manager;

    private final Embedder srcEmbedder;
    private final Embedder tgtEmbedder;
    private final List<EncoderBlock> encoderBlocks;
    private final RMSNorm encoderNorm;
    private final List<DecoderBlock> decoderBlocks;
    private final RMSNorm decoderNorm;
    private final BitLinear outProj;

    // SPM for greedy translation
    private final com.sentencepiece.Model spm;
    private final SentencePieceAlgorithm spmAlgo;

    public Model() throws IOException {
        this(NDManager.newBaseManager());
    }

    public Model(NDManager manager) throws IOException {
        this.cfg = ModelConfig.get();
        this.manager = manager;

        // Load SPM for inference
        this.spm = com.sentencepiece.Model.parseFrom(
            Paths.get(cfg.tokenizerModelPath)
        );
        this.spmAlgo = new SentencePieceAlgorithm(true, Scoring.HIGHEST_SCORE);

        // Initialize shapes
        Shape embedInput = new Shape(cfg.batchSize, cfg.maxSeqLen);
        Shape seqShape = new Shape(cfg.batchSize, cfg.maxSeqLen, cfg.dModel);

        // Embedders
        this.srcEmbedder = new Embedder(cfg.vocabSize, cfg.dModel);
        this.tgtEmbedder = new Embedder(cfg.vocabSize, cfg.dModel);
        this.srcEmbedder.initialize(manager, DataType.FLOAT32, embedInput);
        this.tgtEmbedder.initialize(manager, DataType.FLOAT32, embedInput);

        // Encoder stack
        this.encoderBlocks = new ArrayList<>();
        for (int i = 0; i < cfg.nEncoderLayers; i++) {
            EncoderBlock b = new EncoderBlock(
                cfg.dModel,
                cfg.nHeads,
                cfg.dFfn,
                cfg.ropeBase,
                cfg.maxSeqLen,
                cfg.quantEps
            );
            b.initialize(manager, DataType.FLOAT32, seqShape);
            encoderBlocks.add(b);
        }
        this.encoderNorm = new RMSNorm(cfg.dModel, cfg.eps);
        this.encoderNorm.initialize(manager, DataType.FLOAT32, seqShape);

        // Decoder stack
        this.decoderBlocks = new ArrayList<>();
        for (int i = 0; i < cfg.nDecoderLayers; i++) {
            DecoderBlock b = new DecoderBlock(
                cfg.dModel,
                cfg.nHeads,
                cfg.dFfn,
                cfg.ropeBase,
                cfg.maxSeqLen,
                cfg.quantEps
            );
            b.initialize(manager, DataType.FLOAT32, seqShape, seqShape);
            decoderBlocks.add(b);
        }
        this.decoderNorm = new RMSNorm(cfg.dModel, cfg.eps);
        this.decoderNorm.initialize(manager, DataType.FLOAT32, seqShape);

        // Output projection
        this.outProj = new BitLinear(cfg.dModel, cfg.vocabSize, cfg.quantEps);
        this.outProj.initialize(manager, DataType.FLOAT32, seqShape);
    }

    @Override
    protected NDList forwardInternal(
        ParameterStore ps,
        NDList inputs,
        boolean training,
        PairList<String, Object> params
    ) {
        NDArray srcIds = inputs.get(0);
        NDArray tgtIn = inputs.get(1);

        NDArray logits = forward(ps, srcIds, tgtIn, training);
        return new NDList(logits);
    }

    public NDArray forward(
        ParameterStore ps,
        NDArray srcIds,
        NDArray tgtIn,
        boolean training
    ) {
        NDArray h = srcEmbedder
            .forward(ps, new NDList(srcIds), training)
            .singletonOrThrow();

        for (EncoderBlock b : encoderBlocks) {
            h = b.forward(ps, new NDList(h), training).singletonOrThrow();
        }
        NDArray memory = encoderNorm
            .forward(ps, new NDList(h), training)
            .singletonOrThrow();

        h = tgtEmbedder
            .forward(ps, new NDList(tgtIn), training)
            .singletonOrThrow();

        for (DecoderBlock b : decoderBlocks) {
            h = b
                .forward(ps, new NDList(h, memory), training)
                .singletonOrThrow();
        }
        NDArray decoded = decoderNorm
            .forward(ps, new NDList(h), training)
            .singletonOrThrow();

        return outProj
            .forward(ps, new NDList(decoded), training)
            .singletonOrThrow();
    }

    public String greedyTranslate(String sourceText, int maxLen) {
        try (NDManager stepMgr = manager.newSubManager()) {
            ParameterStore ps = new ParameterStore(stepMgr, false);

            List<Integer> srcTokens = spm.encodeNormalized(
                sourceText.trim(),
                spmAlgo
            );
            long[] srcArr = new long[cfg.maxSeqLen];
            for (
                int i = 0;
                i < Math.min(srcTokens.size(), cfg.maxSeqLen);
                i++
            ) {
                srcArr[i] = srcTokens.get(i);
            }

            NDArray srcIds = stepMgr.create(srcArr).reshape(1, cfg.maxSeqLen);
            NDArray h = srcEmbedder
                .forward(ps, new NDList(srcIds), false)
                .singletonOrThrow();

            for (EncoderBlock b : encoderBlocks) {
                h = b.forward(ps, new NDList(h), false).singletonOrThrow();
            }
            NDArray memory = encoderNorm
                .forward(ps, new NDList(h), false)
                .singletonOrThrow();

            List<Integer> generated = new ArrayList<>();
            generated.add(BOS_ID);

            for (int step = 0; step < maxLen; step++) {
                long[] tgtArr = new long[cfg.maxSeqLen];
                for (
                    int i = 0;
                    i < Math.min(generated.size(), cfg.maxSeqLen);
                    i++
                ) {
                    tgtArr[i] = generated.get(i);
                }

                NDArray tgtIds = stepMgr
                    .create(tgtArr)
                    .reshape(1, cfg.maxSeqLen);
                NDArray dh = tgtEmbedder
                    .forward(ps, new NDList(tgtIds), false)
                    .singletonOrThrow();

                for (DecoderBlock b : decoderBlocks) {
                    dh = b
                        .forward(ps, new NDList(dh, memory), false)
                        .singletonOrThrow();
                }
                NDArray decoded = decoderNorm
                    .forward(ps, new NDList(dh), false)
                    .singletonOrThrow();
                NDArray logits = outProj
                    .forward(ps, new NDList(decoded), false)
                    .singletonOrThrow();

                int pos = Math.min(step, cfg.maxSeqLen - 1);
                long next = logits.get("0, " + pos + ", :").argMax().getLong();

                if (next == EOS_ID) break;
                generated.add((int) next);
            }

            return spm.decodeSmart(generated.subList(1, generated.size()));
        }
    }

    /**
     * Returns flat list of parameters for optimizer use.
     */
    public List<Parameter> getParameterList() {
        List<Parameter> params = new ArrayList<>();
        collectParameters(srcEmbedder, params);
        collectParameters(tgtEmbedder, params);
        encoderBlocks.forEach(b -> collectParameters(b, params));
        collectParameters(encoderNorm, params);
        decoderBlocks.forEach(b -> collectParameters(b, params));
        collectParameters(decoderNorm, params);
        collectParameters(outProj, params);
        return params;
    }

    private void collectParameters(
        ai.djl.nn.AbstractBlock block,
        List<Parameter> params
    ) {
        params.addAll(block.getParameters().values());
    }

    public void save(Path dir) throws IOException {
        java.nio.file.Files.createDirectories(dir);
        saveBlock(srcEmbedder, dir, "src-embed");
        saveBlock(tgtEmbedder, dir, "tgt-embed");
        for (int i = 0; i < encoderBlocks.size(); i++) {
            saveBlock(encoderBlocks.get(i), dir, "encoder-" + i);
        }
        saveBlock(encoderNorm, dir, "encoder-norm");
        for (int i = 0; i < decoderBlocks.size(); i++) {
            saveBlock(decoderBlocks.get(i), dir, "decoder-" + i);
        }
        saveBlock(decoderNorm, dir, "decoder-norm");
        saveBlock(outProj, dir, "out-proj");
    }

    public void load(Path dir) throws IOException, MalformedModelException {
        loadBlock(srcEmbedder, dir, "src-embed");
        loadBlock(tgtEmbedder, dir, "tgt-embed");
        for (int i = 0; i < encoderBlocks.size(); i++) {
            loadBlock(encoderBlocks.get(i), dir, "encoder-" + i);
        }
        loadBlock(encoderNorm, dir, "encoder-norm");
        for (int i = 0; i < decoderBlocks.size(); i++) {
            loadBlock(decoderBlocks.get(i), dir, "decoder-" + i);
        }
        loadBlock(decoderNorm, dir, "decoder-norm");
        loadBlock(outProj, dir, "out-proj");
    }

    private void saveBlock(ai.djl.nn.AbstractBlock block, Path dir, String name)
        throws IOException {
        try (ai.djl.Model m = ai.djl.Model.newInstance(name)) {
            m.setBlock(block);
            m.save(dir, name);
        }
    }

    private void loadBlock(ai.djl.nn.AbstractBlock block, Path dir, String name)
        throws IOException, MalformedModelException {
        try (ai.djl.Model m = ai.djl.Model.newInstance(name)) {
            m.setBlock(block);
            m.load(dir, name);
        }
    }

    public NDManager getManager() {
        return manager;
    }

    public ModelConfig getConfig() {
        return cfg;
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        Shape in = inputShapes[0];
        return new Shape[] { new Shape(in.get(0), in.get(1), cfg.vocabSize) };
    }

    // AutoCloseable implementation - no @Override needed
    public void close() {
        manager.close();
    }
}
