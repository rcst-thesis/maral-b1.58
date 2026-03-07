import os
import re

import sentencepiece as spm  # pyright: ignore[reportMissingImports]

# config
INPUT_TSV = "src/main/resources/data/en-tl.tsv"
CORPUS_TXT = "src/main/resources/data/corpus.txt"
MODEL_PREFIX = "src/main/resources/models/en_tl"
MODEL_TYPE = "bpe"
TARGET_VOCAB = 8000
SMOKE_SAMPLES = [
    "Good morning, how are you?",
    "Magandang umaga, kumusta ka?",
]


# Intentionally cause an error to automatically get the max vocab size.
def probe_max_vocab(corpus_path: str) -> int:
    try:
        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix="/tmp/_probe",
            vocab_size=99999,
            model_type="bpe",
            character_coverage=0.9995,
        )
        return 99999
    except RuntimeError as e:
        match = re.search(r"<=\s*(\d+)", str(e))
        if match:
            return int(match.group(1))
        raise


# ensure output dirs exist
for path in (CORPUS_TXT, MODEL_PREFIX):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# flatten TSV by streamimg line-by-line
# we dont load it to memory
print("Building corpus ...")
with (
    open(INPUT_TSV, encoding="utf-8") as tsv,
    open(CORPUS_TXT, "w", encoding="utf-8") as out,
):
    for line in tsv:
        if not line.strip():
            continue
        cols = line.rstrip("\n").split("\t")
        if len(cols) < 2:
            continue
        for col in cols:
            text = " ".join(col.split())
            if text:
                out.write(text + "\n")
print("  done.")

# get max vocab
print("Probing vocab ...")
max_vocab = probe_max_vocab(CORPUS_TXT)
VOCAB_SIZE = min(TARGET_VOCAB, max_vocab)
print(f"  max={max_vocab}  using={VOCAB_SIZE}")

# train
TRAIN_ARGS = dict(
    input=CORPUS_TXT,
    model_prefix=MODEL_PREFIX,
    vocab_size=VOCAB_SIZE,
    model_type=MODEL_TYPE,
    character_coverage=0.9995,
    # special token ids
    pad_id=0,
    pad_piece="[PAD]",
    unk_id=1,
    unk_piece="[UNK]",
    bos_id=2,
    bos_piece="[BOS]",
    eos_id=3,
    eos_piece="[EOS]",
    # reserved symbols for BERT-style fine-tuning
    user_defined_symbols="<sep>,<cls>",
)
print("Training model ...")
spm.SentencePieceTrainer.train(**TRAIN_ARGS)
print(f"  saved: {MODEL_PREFIX}.model")

# test
sp = spm.SentencePieceProcessor()
sp.load(f"{MODEL_PREFIX}.model")

print("Smoke test ...")

for text in SMOKE_SAMPLES:
    ids = sp.encode_as_ids(text)
    print(f"  input  : {text}")
    print(f"  pieces : {sp.encode_as_pieces(text)}")
    print(f"  ids    : {ids}")
    print(f"  decoded: {sp.decode_ids(ids)}")

print(
    f"\nvocab={sp.get_piece_size()}"
    f"  [PAD]={sp.pad_id()}  [UNK]={sp.unk_id()}"
    f"  [BOS]={sp.bos_id()}  [EOS]={sp.eos_id()}"
)
