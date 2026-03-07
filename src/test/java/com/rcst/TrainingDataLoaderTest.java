package com.rcst;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.Batch;
import java.util.Arrays;
import java.util.List;
import junit.framework.TestCase;

public class TrainingDataLoaderTest extends TestCase {

    private static final int BLOCK_SIZE = 8;
    private static final int BATCH_SIZE = 4;
    private static final double TRAIN_RATIO = 0.8;

    private static final List<String> SENTENCES = List.of(
        "Good morning, how are you?",
        "Magandang umaga, kumusta ka?",
        "I am going to the market.",
        "Pupunta ako sa palengke.",
        "The weather is nice today.",
        "Maganda ang panahon ngayon.",
        "Can you help me please?",
        "Maaari mo ba akong tulungan?",
        "I love learning new languages.",
        "Mahilig akong matuto ng bagong wika."
    );

    private NDManager manager;
    private Tokenizer tokenizer;
    private TensorEncoder encoder;
    private TrainingDataLoader loader;

    @Override
    protected void setUp() throws Exception {
        manager = NDManager.newBaseManager();
        tokenizer = new Tokenizer();
        encoder = new TensorEncoder(tokenizer, manager);
        NDArray[] tensors = encoder.encodeBatch(SENTENCES);

        loader = new TrainingDataLoader(
            tensors,
            manager,
            BLOCK_SIZE,
            BATCH_SIZE,
            TRAIN_RATIO,
            42L
        );
    }

    @Override
    protected void tearDown() throws Exception {
        tokenizer.close();
        manager.close();
    }

    public void testSplitSizesAreNonZero() {
        assertTrue(loader.trainSize() > 0);
        assertTrue(loader.valSize() > 0);

        System.out.printf(
            "tokens: train=%,d  val=%,d%n",
            loader.trainSize(),
            loader.valSize()
        );
    }

    public void testTrainSplitIsLargerThanVal() {
        assertTrue(loader.trainSize() > loader.valSize());

        System.out.printf(
            "tokens: train=%,d > val=%,d%n",
            loader.trainSize(),
            loader.valSize()
        );
    }

    public void testTrainBatchShape() {
        try (Batch batch = loader.sampleTrain()) {
            NDArray x = batch.getData().head();
            NDArray y = batch.getLabels().head();

            assertEquals(BATCH_SIZE, (int) x.getShape().get(0));
            assertEquals(BLOCK_SIZE, (int) x.getShape().get(1));
            assertEquals(BATCH_SIZE, (int) y.getShape().get(0));
            assertEquals(BLOCK_SIZE, (int) y.getShape().get(1));

            System.out.println("x " + x.toDebugString(true));
            System.out.println("y " + y.toDebugString(true));
            System.out.printf(
                "train batch  x=%s  y=%s%n",
                x.getShape(),
                y.getShape()
            );
        }
    }

    public void testValBatchShape() {
        try (Batch batch = loader.sampleValidation()) {
            NDArray x = batch.getData().head();
            assertEquals(BATCH_SIZE, (int) x.getShape().get(0));
            assertEquals(BLOCK_SIZE, (int) x.getShape().get(1));

            System.out.println("x " + x.toDebugString(true));
            System.out.printf("val batch    x=%s%n", x.getShape());
        }
    }

    public void testXAndYAreShiftedByOne() {
        try (Batch batch = loader.sampleTrain()) {
            long[] x = batch.getData().head().toLongArray();
            long[] y = batch.getLabels().head().toLongArray();

            for (int col = 0; col < BLOCK_SIZE - 1; col++) {
                assertEquals(x[col + 1], y[col]);
            }

            System.out.printf(
                "x=%s%ny=%s%n",
                Arrays.toString(Arrays.copyOf(x, BLOCK_SIZE)),
                Arrays.toString(Arrays.copyOf(y, BLOCK_SIZE))
            );
        }
    }

    public void testSamplesAreDifferentAcrossCalls() {
        try (Batch a = loader.sampleTrain(); Batch b = loader.sampleTrain()) {
            long[] aIds = a.getData().head().toLongArray();
            long[] bIds = b.getData().head().toLongArray();
            boolean differs = false;

            for (int i = 0; i < aIds.length; i++) {
                if (aIds[i] != bIds[i]) {
                    differs = true;
                    break;
                }
            }

            assertTrue(differs);

            System.out.printf(
                "sample1=%s%nsample2=%s%n",
                Arrays.toString(Arrays.copyOf(aIds, BLOCK_SIZE)),
                Arrays.toString(Arrays.copyOf(bIds, BLOCK_SIZE))
            );
        }
    }
}
