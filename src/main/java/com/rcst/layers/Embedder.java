package com.rcst.layers;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.training.dataset.Batch;
import ai.djl.util.PairList;
import com.sentencepiece.Model;
import com.sentencepiece.Scoring;
import com.sentencepiece.SentencePieceAlgorithm;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Reads corpus from disk, tokenizes, flattens into a token stream,
 * splits into train/val, and performs embedding lookup.
 *
 * forward() output shape: (B, T, n_embed) which go directly into the EncoderBlock.
 * RoPE handles positional encoding inside each attention head.
 */
public class Embedder extends AbstractBlock {

    private static final int BOS_ID = 2;
    private static final int EOS_ID = 3;
    private static final String MODEL_PATH =
        "src/main/resources/models/maral.bpe.model";
    private static final String CORPUS_PATH =
        "src/main/resources/data/corpus.txt";

    private final Model spmModel;
    private final SentencePieceAlgorithm algorithm;
    private final int nEmbed;
    private final Parameter weight;

    // Set by initLoader()
    private NDArray train;
    private NDArray val;
    private int blockSize;
    private int batchSize;
    private NDManager loaderManager;
    private Random rng;

    public Embedder(int vocabSize, int nEmbed) throws IOException {
        this(vocabSize, nEmbed, Paths.get(MODEL_PATH));
    }

    public Embedder(int vocabSize, int nEmbed, Path modelPath)
        throws IOException {
        this.nEmbed = nEmbed;
        this.spmModel = Model.parseFrom(modelPath);
        this.algorithm = new SentencePieceAlgorithm(
            true,
            Scoring.HIGHEST_SCORE
        );

        this.weight = addParameter(
            Parameter.builder()
                .setName("weight")
                .setType(Parameter.Type.WEIGHT)
                .optShape(new Shape(vocabSize, nEmbed))
                .build()
        );
    }

    /**
     * Reads corpus from disk, tokenizes, and splits into train/val.
     * Call once before sampleTrain() / sampleValidation().
     *
     * @param manager    NDManager that owns the token stream
     * @param blockSize  context window length
     * @param batchSize  parallel chunks per batch
     * @param trainRatio fraction of tokens for training (e.g. 0.9)
     * @param seed       RNG seed for reproducibility
     */
    public void initLoader(
        NDManager manager,
        int blockSize,
        int batchSize,
        double trainRatio,
        long seed
    ) throws IOException {
        this.blockSize = blockSize;
        this.batchSize = batchSize;
        this.loaderManager = manager;
        this.rng = new Random(seed);

        List<NDArray> shuffled = Files.readAllLines(Paths.get(CORPUS_PATH))
            .stream()
            .map(s -> toTensor(s, manager))
            .collect(Collectors.toList());
        Collections.shuffle(shuffled, new Random(seed));

        // Flatten into one long token stream
        long total = shuffled
            .stream()
            .mapToLong(t -> t.getShape().get(0))
            .sum();

        long[] flat = new long[(int) total];

        int cursor = 0;
        for (NDArray t : shuffled) {
            long[] ids = t.toLongArray();

            System.arraycopy(ids, 0, flat, cursor, ids.length);

            cursor += ids.length;
        }

        int trainLen = (int) Math.round(flat.length * trainRatio);
        this.train = manager.create(Arrays.copyOfRange(flat, 0, trainLen));
        this.val = manager.create(
            Arrays.copyOfRange(flat, trainLen, flat.length)
        );
    }

    public void initLoader(
        NDManager manager,
        int blockSize,
        int batchSize,
        double trainRatio
    ) throws IOException {
        initLoader(manager, blockSize, batchSize, trainRatio, 42L);
    }

    public Batch getSampleFromTrainingSplit() {
        return sample(train);
    }

    public Batch getSampleFromValidationSplit() {
        return sample(val);
    }

    public long getTrainSize() {
        return train.getShape().get(0);
    }

    public long getValSize() {
        return val.getShape().get(0);
    }

    private Batch sample(NDArray data) {
        long streamLen = data.getShape().get(0);
        long[][] xRows = new long[batchSize][blockSize];
        long[][] yRows = new long[batchSize][blockSize];

        for (int b = 0; b < batchSize; b++) {
            long start = (long) (rng.nextDouble() *
                (streamLen - blockSize - 1));
            long[] chunk = data
                .get(start + ":" + (start + blockSize + 1))
                .toLongArray();

            // x: [0, blockSize)
            System.arraycopy(chunk, 0, xRows[b], 0, blockSize);
            // y: [1, blockSize]
            System.arraycopy(chunk, 1, yRows[b], 0, blockSize);
        }

        NDManager batchManager = loaderManager.newSubManager();
        NDArray x = batchManager.create(xRows); // (B, T)
        NDArray y = batchManager.create(yRows); // (B, T)

        return new Batch(
            batchManager,
            new NDList(x),
            new NDList(y),
            batchSize,
            null,
            null,
            0,
            0
        );
    }

    /**
     * input  shape: (B, T)          — int64 token ids
     * output shape: (B, T, n_embed) — full-precision token embeddings
     */
    @Override
    protected NDList forwardInternal(
        ParameterStore parameterStore,
        NDList inputs,
        boolean training,
        PairList<String, Object> params
    ) {
        NDArray x = inputs.singletonOrThrow();
        NDArray w = parameterStore.getValue(
            weight,
            x.getManager().getDevice(),
            training
        );

        return new NDList(w.get(x.toType(DataType.INT64, false)));
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        Shape in = inputShapes[0]; // (B, T)

        return new Shape[] { new Shape(in.get(0), in.get(1), nEmbed) };
    }

    /** Tokenize text → 1-D tensor of ids with BOS/EOS, shape: (seq_len,) */
    private NDArray toTensor(String text, NDManager manager) {
        List<Integer> ids = new ArrayList<>();
        ids.add(BOS_ID);
        ids.addAll(spmModel.encodeNormalized(text, algorithm));
        ids.add(EOS_ID);

        long[] arr = ids.stream().mapToLong(Integer::longValue).toArray();
        return manager.create(arr);
    }
}
