import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import keras

corpus = [
    "Je suis un texte",
    "Je suis un autre texte",
    "Je ne suis pas un chat",
    "Si je pense, je suis",
    "Un écrivain sait écrire",
]
sentence = "Je suis un chat qui sait écrire"

# Default: lowercase, strip punctuation, split on whitespace.
vectorizer = keras.layers.TextVectorization(output_mode="int")
vectorizer.adapt(corpus)
vocab = vectorizer.get_vocabulary()

print("Vocabulary :", [str(t) for t in vocab])
ids = vectorizer(sentence).numpy()
print("Sentence   :", sentence)
print("Encoded    :", ids.tolist())
print("Decoded    :", [str(vocab[i]) for i in ids])
