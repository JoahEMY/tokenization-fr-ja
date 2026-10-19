import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from collections import Counter
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import MeCab
import numpy as np
import tensorflow as tf

from tatoeba import load_pairs

FIGURES = Path(__file__).resolve().parent.parent / "report" / "figures"
COLORS = {"French": "#2a78d6", "Japanese": "#eb6834"}
VOCAB_SIZES = [500, 1_000, 2_000, 5_000, 10_000, 20_000]

plt.rcParams["font.family"] = ["IBM Plex Sans", "DejaVu Sans"]


def standardize(text):
    # Lowercase and strip every punctuation mark and symbol, French and
    # Japanese alike, so that 。 and 、 are not counted as Japanese words.
    return tf.strings.regex_replace(tf.strings.lower(text), r"[\p{P}\p{S}]", "")


def tokenize(texts):
    return [
        [t.decode() for t in sentence]
        for sentence in tf.strings.split(standardize(texts)).to_list()
    ]


def unk_rate(train, test, max_tokens):
    vectorizer = keras.layers.TextVectorization(
        max_tokens=max_tokens, standardize=standardize, output_mode="int"
    )
    vectorizer.adapt(train)
    ids = vectorizer(test).numpy()
    return (ids == 1).sum() / (ids > 0).sum()


def vocab_growth(sentences, step=500):
    seen, sizes = set(), []
    for i, tokens in enumerate(sentences, 1):
        seen.update(tokens)
        if i % step == 0:
            sizes.append((i, len(seen)))
    return sizes


def plot(curves, xlabel, ylabel, path, xlog=False):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for name, (x, y) in curves.items():
        ax.plot(x, y, color=COLORS[name], linewidth=2, marker="o" if xlog else None, markersize=4)
        ax.annotate(name, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color="#0b0b0b")
    if xlog:
        ax.set_xscale("log")
        ax.set_xticks(x, [f"{v:,}" for v in x])
        ax.minorticks_off()
    else:
        ax.set_xlim(left=0)
        ax.xaxis.set_major_formatter("{x:,.0f}")
    ax.yaxis.set_major_formatter("{x:,.0f}")
    ax.set_ylim(bottom=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color="#e4e3df", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.margins(x=0.15)
    fig.tight_layout()
    fig.savefig(path)
    print(f"Saved {path}")


pairs = load_pairs()
split = int(len(pairs) * 0.8)
wakati = MeCab.Tagger("-Owakati")

text = {
    "French": [fr for fr, _ in pairs],
    "Japanese": [wakati.parse(ja).strip() for _, ja in pairs],
}
tokens = {name: tokenize(sentences) for name, sentences in text.items()}

print(f"{len(pairs)} pairs, {split} for train, {len(pairs) - split} for test\n")

print("--- Length per sentence (mean) ---")
for name in text:
    chars = np.mean([sum(len(t) for t in s) for s in tokens[name]])
    words = np.mean([len(s) for s in tokens[name]])
    print(f"{name:9s} {chars:5.1f} characters  {words:5.1f} tokens  {chars / words:4.2f} characters per token")
print()

print("--- Vocabulary (train) ---")
for name in text:
    counts = Counter(t for s in tokens[name][:split] for t in s)
    once = sum(1 for c in counts.values() if c == 1)
    print(f"{name:9s} {len(counts):6d} distinct tokens, {once / len(counts):.0%} seen only once")
print()

print("--- [UNK] rate on test ---")
rates = {name: [unk_rate(text[name][:split], text[name][split:], k) for k in VOCAB_SIZES] for name in text}
print("vocab size  " + "  ".join(f"{k:>7d}" for k in VOCAB_SIZES))
for name in text:
    print(f"{name:9s}   " + "  ".join(f"{r:7.1%}" for r in rates[name]))
print()

print("--- What spaces miss in French ---")
merged = Counter(
    standardize(w).numpy().decode()
    for s in text["French"]
    for w in s.lower().split()
    if any(c in w for c in "'’-")
)
share = sum(merged.values()) / sum(len(s) for s in tokens["French"])
print(f"{len(merged)} distinct tokens glued by an apostrophe or a hyphen, {share:.0%} of all French tokens")
print("most common:", ", ".join(w for w, _ in merged.most_common(10)))
print()

print("--- What MeCab outputs in Japanese ---")
tagger = MeCab.Tagger()
pos = Counter()
for _, ja in pairs:
    node = tagger.parseToNode(ja)
    while node:
        if node.surface and node.feature.split(",")[0] != "補助記号":
            pos[node.feature.split(",")[0]] += 1
        node = node.next
total = sum(pos.values())
for tag, label in [("助詞", "particles"), ("助動詞", "auxiliary verbs")]:
    print(f"{label:16s} {pos[tag] / total:.0%} of Japanese tokens")
print()

FIGURES.mkdir(parents=True, exist_ok=True)
growth = {name: tuple(zip(*vocab_growth(tokens[name][:split]))) for name in text}
plot(growth, "Sentences", "Distinct tokens", FIGURES / "vocabulary_growth.pdf")
plot({name: (VOCAB_SIZES, [100 * r for r in rates[name]]) for name in text},
     "Vocabulary size (max_tokens)", "[UNK] tokens in test (%)", FIGURES / "unk_rate.pdf", xlog=True)
