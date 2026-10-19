import numpy as np
from transformers import AutoTokenizer

from tatoeba import load_pairs

# Two subword tokenizers each trained once on about a hundred languages.
TOKENIZERS = ["bert-base-multilingual-cased", "xlm-roberta-base"]

pairs = load_pairs()
example = ("Nous nous sommes dépêchés pour attraper le bus.", "私たちはバスに乗るために急いだ。")

for name in TOKENIZERS:
    tokenizer = AutoTokenizer.from_pretrained(name)

    def count(text):
        return len(tokenizer.tokenize(text))

    fr = np.array([count(fr) for fr, _ in pairs])
    ja = np.array([count(ja) for _, ja in pairs])

    print(f"--- {name} ---")
    print(f"French   {fr.mean():5.1f} tokens per sentence")
    print(f"Japanese {ja.mean():5.1f} tokens per sentence")
    print(f"Japanese uses more tokens than French in {np.mean(ja > fr):.0%} of the pairs")
    for text in example:
        print(f"{count(text):3d}  {tokenizer.tokenize(text)}")
    print()
