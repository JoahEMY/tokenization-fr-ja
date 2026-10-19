import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import keras

corpus = [
    "私はテキストです",
    "私は別のテキストです",
    "私は猫ではありません",
    "我思う、ゆえに我あり",
    "作家は書くことができる",
]
sentence = "私は書くことができる猫です"


def demo(title, **kwargs):
    vectorizer = keras.layers.TextVectorization(output_mode="int", **kwargs)
    vectorizer.adapt(corpus)
    vocab = vectorizer.get_vocabulary()

    print(f"--- {title} ---")
    print("Vocabulary :", [str(t) for t in vocab])
    ids = vectorizer(sentence).numpy()
    print("Sentence   :", sentence)
    print("Encoded    :", ids.tolist())
    print("Decoded    :", [str(vocab[i]) for i in ids])
    print()


# Japanese does not put spaces between words,
# so each sentence becomes a single token.
demo("Whitespace split (default)")

# Character-level split.
demo("Character split", split="character")
