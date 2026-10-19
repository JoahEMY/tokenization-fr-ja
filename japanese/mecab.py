import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import keras
import MeCab

corpus = [
    "私はテキストです",
    "私は別のテキストです",
    "私は猫ではありません",
    "我思う、ゆえに我あり",
    "作家は書くことができる",
]
sentence = "私は書くことができる猫です"

# MeCab cuts the sentence into words and tags each one.
tagger = MeCab.Tagger()
print("--- MeCab analysis ---")
node = tagger.parseToNode(sentence)
while node:
    if node.surface:
        print(f"{node.surface}\t{node.feature.split(',')[0]}")
    node = node.next
print()

# -Owakati only outputs the words separated by spaces,
# so TextVectorization can split them as in French.
wakati = MeCab.Tagger("-Owakati")


def segment(text):
    return wakati.parse(text).strip()


vectorizer = keras.layers.TextVectorization(output_mode="int")
vectorizer.adapt([segment(t) for t in corpus])
vocab = vectorizer.get_vocabulary()

print("--- MeCab + TextVectorization ---")
print("Vocabulary :", [str(t) for t in vocab])
ids = vectorizer(segment(sentence)).numpy()
print("Sentence   :", segment(sentence))
print("Encoded    :", ids.tolist())
print("Decoded    :", [str(vocab[i]) for i in ids])
