package com.rcst;

import ai.djl.ndarray.NDArray;
import ai.djl.training.dataset.Batch;
import java.util.Arrays;
import junit.extensions.TestSetup;
import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

public class TrainingDataLoaderTest extends TestCase {

    public static Test suite() {
        return new TestSetup(new TestSuite(TrainingDataLoaderTest.class)) {
            @Override
            protected void setUp() throws Exception {
                TestFixture.init();
            }

            @Override
            protected void tearDown() throws Exception {
                TestFixture.destroy();
            }
        };
    }

    public void testSplitSizesAreNonZero() {
        assertTrue(TestFixture.loader.trainSize() > 0);
        assertTrue(TestFixture.loader.valSize() > 0);

        System.out.printf(
            "train=%,d  val=%,d%n",
            TestFixture.loader.trainSize(),
            TestFixture.loader.valSize()
        );
    }

    public void testTrainSplitIsLargerThanVal() {
        assertTrue(
            TestFixture.loader.trainSize() > TestFixture.loader.valSize()
        );
    }

    public void testTrainBatchShape() {
        try (Batch batch = TestFixture.loader.sampleTrain()) {
            NDArray x = batch.getData().head();
            NDArray y = batch.getLabels().head();

            assertEquals(TestFixture.BATCH_SIZE, (int) x.getShape().get(0));
            assertEquals(TestFixture.BLOCK_SIZE, (int) x.getShape().get(1));
            assertEquals(TestFixture.BATCH_SIZE, (int) y.getShape().get(0));
            assertEquals(TestFixture.BLOCK_SIZE, (int) y.getShape().get(1));

            System.out.printf(
                "train batch x=%s  y=%s%n",
                x.getShape(),
                y.getShape()
            );
        }
    }

    public void testValBatchShape() {
        try (Batch batch = TestFixture.loader.sampleValidation()) {
            NDArray x = batch.getData().head();
            assertEquals(TestFixture.BATCH_SIZE, (int) x.getShape().get(0));
            assertEquals(TestFixture.BLOCK_SIZE, (int) x.getShape().get(1));
            System.out.printf("val batch x=%s%n", x.getShape());
        }
    }

    public void testXAndYAreShiftedByOne() {
        try (Batch batch = TestFixture.loader.sampleTrain()) {
            long[] x = batch.getData().head().toLongArray();
            long[] y = batch.getLabels().head().toLongArray();

            // x[col+1] == y[col] for every column in the first row
            for (int col = 0; col < TestFixture.BLOCK_SIZE - 1; col++) {
                assertEquals(
                    "shift mismatch at col " + col,
                    x[col + 1],
                    y[col]
                );
            }

            System.out.printf(
                "x=%s%ny=%s%n",
                Arrays.toString(Arrays.copyOf(x, TestFixture.BLOCK_SIZE)),
                Arrays.toString(Arrays.copyOf(y, TestFixture.BLOCK_SIZE))
            );
        }
    }

    public void testContextTargetPairs() {
        try (Batch batch = TestFixture.loader.sampleTrain()) {
            NDArray xTensor = batch.getData().head();
            NDArray yTensor = batch.getLabels().head();

            for (int b = 0; b < TestFixture.BATCH_SIZE; b++) {
                long[] x = xTensor.get(b + ":").toLongArray();
                long[] y = yTensor.get(b + ":").toLongArray();

                System.out.printf("batch %d%n", b);

                for (int t = 0; t < TestFixture.BLOCK_SIZE; t++) {
                    System.out.printf(
                        "  context=%-40s → target=%d%n",
                        Arrays.toString(Arrays.copyOfRange(x, 0, t + 1)),
                        y[t]
                    );
                }
            }
        }
    }

    public void testSamplesAreDifferentAcrossCalls() {
        try (
            Batch a = TestFixture.loader.sampleTrain();
            Batch b = TestFixture.loader.sampleTrain()
        ) {
            long[] aIds = a.getData().head().toLongArray();
            long[] bIds = b.getData().head().toLongArray();
            boolean differs = false;

            for (int i = 0; i < aIds.length; i++) {
                if (aIds[i] != bIds[i]) {
                    differs = true;
                    break;
                }
            }

            assertTrue("consecutive samples must differ", differs);
        }
    }
}
