# Tokenization: French vs Japanese

Before a model can read text, the text has to be cut into tokens. In French,
cutting on spaces mostly works. Japanese does not put spaces between words, so
it needs another approach.

This repository compares the two languages:

1. **A toy example**: five sentences in each language passed through
   `keras.layers.TextVectorization`, with the default split, a character split,
   and MeCab (a Japanese morphological analyzer).
2. **A real corpus**: about 38,000 French–Japanese sentence pairs from
   [Tatoeba](https://tatoeba.org), to compare sentence length, vocabulary size
   and unknown tokens.
3. **Multilingual tokenizers**: the same pairs passed through the mBERT and
   XLM-R tokenizers.

The results and explanations are in [`report/report.pdf`](report/report.pdf).

## Files

```
french/textvectorization.py     toy example, French
japanese/textvectorization.py   toy example, Japanese (default and character split)
japanese/mecab.py               toy example, Japanese with MeCab
corpus/tatoeba.py               downloads and loads the Tatoeba pairs
corpus/compare.py               French vs Japanese on the corpus, makes the figures
corpus/multilingual.py          mBERT and XLM-R on the corpus
report/report.tex               source of the report (xelatex report.tex)
```

## Running

```sh
pip install -r requirements.txt
python french/textvectorization.py
python japanese/textvectorization.py
python japanese/mecab.py
python corpus/compare.py
python corpus/multilingual.py
```

The Tatoeba files (about 13 MB) are downloaded to `data/` on the first run.
