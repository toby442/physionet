from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("./bluebert_local")

# 拿一条真实的redacted记录做测试
test_sentences = [
    "(LMC) transfer for renal transplant. pt SOB x 1 day. possibly rejecting kidney. kidney tx in 2008.",
    "(<<HOSPITAL>>) transfer for renal transplant. pt SOB x 1 day. possibly rejecting kidney. kidney tx in <<DATE>>.",
]

labels = ["Synthetic版本", "Redacted版本"]

for label, sentence in zip(labels, test_sentences):
    tokens = tokenizer.tokenize(sentence)
    print(f"\n===== {label} =====")
    print(f"原文: {sentence}")
    print(f"分词结果 ({len(tokens)}个token): {tokens}")

    # 单独看一下占位符/缩写部分具体被拆成了什么
    print(f"\n开头部分单独分词:")
    if "HOSPITAL" in sentence:
        print(f"  '(<<HOSPITAL>>)' -> {tokenizer.tokenize('(<<HOSPITAL>>)')}")
        print(f"  '<<DATE>>' -> {tokenizer.tokenize('<<DATE>>')}")
    else:
        print(f"  '(LMC)' -> {tokenizer.tokenize('(LMC)')}")
        print(f"  '2008' -> {tokenizer.tokenize('2008')}")

    # 检查有没有变成[UNK]未知token
    unk_count = tokens.count("[UNK]")
    print(f"\n[UNK]未知token数量: {unk_count}")