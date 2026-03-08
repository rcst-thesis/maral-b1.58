package com.rcst;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import com.rcst.utils.TensorEncoder;
import com.rcst.utils.Tokenizer;
import java.nio.file.Paths;
import java.util.List;
import junit.framework.TestCase;

public class TensorEncoderTest extends TestCase {

    private static final String MODEL_PATH =
        "src/main/resources/models/maral.bpe.model";

    private NDManager manager;
    private TensorEncoder encoder;

    @Override
    protected void setUp() throws Exception {
        manager = NDManager.newBaseManager();
        encoder = new TensorEncoder(
            new Tokenizer(Paths.get(MODEL_PATH)),
            manager
        );
    }

    @Override
    protected void tearDown() {
        manager.close();
    }

    public void testEncodeShape() {
        NDArray tensor = encoder.encode("Good morning");
        assertEquals(1, tensor.getShape().dimension()); // 1-D
    }

    public void testEncodeBatchedShape() {
        NDArray tensor = encoder.encodeBatched("Good morning");
        assertEquals(2, tensor.getShape().dimension()); // 2-D
        assertEquals(1, tensor.getShape().get(0)); // batch size = 1
    }

    public void testEncodeBatchCount() {
        List<String> sentences = List.of("Hello", "Kumusta", "Magandang gabi");
        NDArray[] tensors = encoder.encodeBatch(sentences);
        assertEquals(3, tensors.length);
    }

    public void testEncodeStartsWithBos() {
        NDArray tensor = encoder.encode("Hello");
        assertEquals(2L, tensor.getLong(0)); // BOS_ID
    }

    public void testEncodeEndsWithEos() {
        NDArray tensor = encoder.encode("Hello");
        long lastIdx = tensor.getShape().get(0) - 1;
        assertEquals(3L, tensor.getLong(lastIdx)); // EOS_ID
    }
}
