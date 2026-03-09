package com.rcst.common;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.training.dataset.Batch;
import com.rcst.TestFixture;
import junit.extensions.TestSetup;
import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

public class RMSNormTest extends TestCase {

    private static RMSNorm rmsNorm;

    public static Test suite() {
        return new TestSetup(new TestSuite(RMSNormTest.class)) {
            @Override
            protected void setUp() throws Exception {
                TestFixture.init();
                rmsNorm = TestFixture.buildRMSNorm();
            }

            @Override
            protected void tearDown() throws Exception {
                TestFixture.destroy();
            }
        };
    }

    private NDArray pipeline(Batch batch, ParameterStore ps) {
        return rmsNorm
            .forward(ps, new NDList(TestFixture.embed(batch, ps)), false)
            .singletonOrThrow();
    }

    public void testOutputShapeMatchesInput() {
        ParameterStore ps = TestFixture.freshPs();
        try (Batch batch = TestFixture.embedder.getSampleFromTrainingSplit()) {
            NDArray out = pipeline(batch, ps);
            assertEquals(TestFixture.BATCH_SIZE, (int) out.getShape().get(0));
            assertEquals(TestFixture.BLOCK_SIZE, (int) out.getShape().get(1));
            assertEquals(TestFixture.D_MODEL, (int) out.getShape().get(2));
            System.out.printf("RMSNorm output: %s%n", out.getShape());
        }
    }

    public void testGetOutputShapes() {
        Shape input = new Shape(
            TestFixture.BATCH_SIZE,
            TestFixture.BLOCK_SIZE,
            TestFixture.D_MODEL
        );
        Shape[] out = rmsNorm.getOutputShapes(new Shape[] { input });
        assertEquals(1, out.length);
        assertEquals(input, out[0]);
    }

    public void testOutputIsFloat32() {
        ParameterStore ps = TestFixture.freshPs();
        try (Batch batch = TestFixture.embedder.getSampleFromTrainingSplit()) {
            assertEquals(DataType.FLOAT32, pipeline(batch, ps).getDataType());
        }
    }

    public void testRmsOfEachTokenIsApproxOne() {
        ParameterStore ps = TestFixture.freshPs();
        try (Batch batch = TestFixture.embedder.getSampleFromTrainingSplit()) {
            float[] vals = pipeline(batch, ps).toFloatArray();
            int total = TestFixture.BATCH_SIZE * TestFixture.BLOCK_SIZE;
            for (int bt = 0; bt < total; bt++) {
                float sumSq = 0f;
                for (int d = 0; d < TestFixture.D_MODEL; d++) {
                    float v = vals[bt * TestFixture.D_MODEL + d];
                    sumSq += v * v;
                }
                float rms = (float) Math.sqrt(sumSq / TestFixture.D_MODEL);
                assertEquals(
                    "RMS of token " +
                        bt +
                        " should be approx 1.0 but was " +
                        rms,
                    1.0f,
                    rms,
                    0.05f
                );
            }
            System.out.println("All token RMS values approx 1.0");
        }
    }

    public void testZeroInputGivesZeroOutput() {
        ParameterStore ps = TestFixture.freshPs();
        NDArray zeros = TestFixture.manager.zeros(
            new Shape(
                TestFixture.BATCH_SIZE,
                TestFixture.BLOCK_SIZE,
                TestFixture.D_MODEL
            ),
            DataType.FLOAT32
        );
        for (float v : rmsNorm
            .forward(ps, new NDList(zeros), false)
            .singletonOrThrow()
            .toFloatArray()) {
            assertEquals("RMSNorm(0) must be 0", 0f, v, 1e-6f);
        }
    }

    public void testGammaInitialisedToOnes() {
        NDArray g = rmsNorm.getParameters().get("gamma").getArray();
        for (float v : g.toFloatArray()) {
            assertEquals("gamma must be initialised to 1.0", 1.0f, v, 1e-6f);
        }
        System.out.printf("gamma %s all ones%n", g.getShape());
    }

    public void testShapeIsConsistentAcrossBatches() {
        for (int i = 0; i < 3; i++) {
            ParameterStore ps = TestFixture.freshPs();
            try (
                Batch batch = TestFixture.embedder.getSampleFromTrainingSplit()
            ) {
                NDArray out = pipeline(batch, ps);
                assertEquals(
                    TestFixture.BATCH_SIZE,
                    (int) out.getShape().get(0)
                );
                assertEquals(
                    TestFixture.BLOCK_SIZE,
                    (int) out.getShape().get(1)
                );
                assertEquals(TestFixture.D_MODEL, (int) out.getShape().get(2));
            }
        }
    }

    public void testValBatchShape() {
        ParameterStore ps = TestFixture.freshPs();
        try (
            Batch batch = TestFixture.embedder.getSampleFromValidationSplit()
        ) {
            NDArray out = pipeline(batch, ps);
            assertEquals(TestFixture.BATCH_SIZE, (int) out.getShape().get(0));
            assertEquals(TestFixture.BLOCK_SIZE, (int) out.getShape().get(1));
            assertEquals(TestFixture.D_MODEL, (int) out.getShape().get(2));
        }
    }
}
