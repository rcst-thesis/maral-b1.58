package com.rcst.utils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.Batch;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class TrainingDataLoader {

    private final NDArray train;
    private final NDArray val;
    private final int blockSize;
    private final int batchSize;
    private final NDManager manager;
    private final Random rng;

    /**
     * @param tensors    pre-encoded 1-D NDArrays from TensorEncoder.encodeBatch()
     * @param manager    NDManager that owns the tensors
     * @param blockSize  context length — maximum context length for predicitons
     * @param batchSize  how many independent chunks will be processed in parallel
     * @param trainRatio fraction of tokens used for training (e.g. 0.8)
     * @param seed       random seed for reproducibility
     */
    public TrainingDataLoader(
        NDArray[] tensors,
        NDManager manager,
        int blockSize,
        int batchSize,
        double trainRatio,
        long seed
    ) {
        List<NDArray> shuffled = new ArrayList<>(Arrays.asList(tensors));
        Collections.shuffle(shuffled, new Random(seed));

        // flatten all 1-D tensors into one long token stream
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
        long[] trainFlat = new long[trainLen];
        long[] valFlat = new long[flat.length - trainLen];
        System.arraycopy(flat, 0, trainFlat, 0, trainLen);
        System.arraycopy(flat, trainLen, valFlat, 0, valFlat.length);

        this.train = manager.create(trainFlat);
        this.val = manager.create(valFlat);
        this.blockSize = blockSize;
        this.batchSize = batchSize;
        this.manager = manager;
        this.rng = new Random(seed);
    }

    public TrainingDataLoader(
        NDArray[] tensors,
        NDManager manager,
        int blockSize,
        int batchSize,
        double trainRatio
    ) {
        this(tensors, manager, blockSize, batchSize, trainRatio, 42L);
    }

    /** Sample a batch from the training split. */
    public Batch sampleTrain() {
        return sample(train);
    }

    /** Sample a batch from the validation split. */
    public Batch sampleValidation() {
        return sample(val);
    }

    /**
     * For each item in the batch:
     *   1. Pick a random start in the flat token stream.
     *   2. Slice blockSize + 1 tokens.
     *   3. x = chunk[0..blockSize-1], y = chunk[1..blockSize]  (shifted by 1)
     * Returns a Batch with:
     *   data   → NDList containing x  shape: (batch_size, block_size)
     *   labels → NDList containing y  shape: (batch_size, block_size)
     */
    private Batch sample(NDArray data) {
        long streamLen = data.getShape().get(0);
        long[][] xRows = new long[batchSize][blockSize];
        long[][] yRows = new long[batchSize][blockSize];

        for (int b = 0; b < batchSize; b++) {
            long start = (long) (rng.nextDouble() *
                (streamLen - blockSize - 1));
            long end = start + blockSize + 1;
            long[] chunk = data.get(start + ":" + end).toLongArray();
            System.arraycopy(chunk, 0, xRows[b], 0, blockSize); // x: [0, blockSize)
            System.arraycopy(chunk, 1, yRows[b], 0, blockSize); // y: [1, blockSize]
        }

        // Use a child manager so Batch.close() only releases the batch arrays,
        // not the parent manager that owns train/val.
        NDManager batchManager = manager.newSubManager();
        NDArray x = batchManager.create(xRows); // shape: (batch_size, block_size)
        NDArray y = batchManager.create(yRows); // shape: (batch_size, block_size)
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

    public long trainSize() {
        return train.getShape().get(0);
    }

    public long valSize() {
        return val.getShape().get(0);
    }
}
