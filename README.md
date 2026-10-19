# Tokenization: French vs Japanese

Two small scripts that run a corpus through `keras.layers.TextVectorization`,
one in French, one in Japanese, to see what splitting into words gives.

```sh
pip install -r requirements.txt
python french.py
python japanese.py
```

## French

By default, `TextVectorization` lowercases, strips punctuation and splits on
whitespace. In French, that directly gives words:

```
Vocabulary : ['', '[UNK]', 'je', 'un', 'suis', 'texte', 'écrivain', 'écrire', 'si', 'sait', 'pense', 'pas', 'ne', 'chat', 'autre']
Sentence   : Je suis un chat qui sait écrire
Encoded    : [2, 4, 3, 13, 1, 9, 7]
Decoded    : ['je', 'suis', 'un', 'chat', '[UNK]', 'sait', 'écrire']
```

Only `qui`, which is not in the corpus, becomes `[UNK]`.

## Japanese

Japanese does not use spaces between words. With the default settings, each
whole sentence becomes a single token:

```
Vocabulary : ['', '[UNK]', '私は猫ではありません', '私は別のテキストです', '私はテキストです', '我思う、ゆえに我あり', '作家は書くことができる']
Sentence   : 私は書くことができる猫です
Encoded    : [1]
Decoded    : ['[UNK]']
```

The test sentence does not appear as-is in the corpus, so the whole thing is
`[UNK]`, even though it reuses known pieces (`私は`, `書くことができる`, `猫`).

Splitting by character (`split="character"`) gives something usable:

```
Vocabulary : ['', '[UNK]', 'は', 'で', '私', '我', 'ト', 'テ', 'ス', 'キ', 'り', 'す', 'あ', '猫', '書', '思', '家', '別', '作', 'ん', 'る', 'ゆ', 'ま', 'の', 'に', 'と', 'せ', 'こ', 'く', 'き', 'が', 'え', 'う', '、']
Sentence   : 私は書くことができる猫です
Encoded    : [4, 2, 14, 28, 27, 25, 30, 3, 29, 20, 13, 3, 11]
Decoded    : ['私', 'は', '書', 'く', 'こ', 'と', 'が', 'で', 'き', 'る', '猫', 'で', 'す']
```

No more `[UNK]`, but the notion of a word is gone: `テキスト` is split into 4
characters. Note also that `、` stays in the vocabulary: the default
punctuation stripping only covers ASCII punctuation.

## Summary

|                      | French          | Japanese              |
|----------------------|-----------------|-----------------------|
| Word separator       | space           | none                  |
| Default split gives  | words           | whole sentences       |
| Simple workaround    | —               | split by character    |

To get actual words in Japanese, you need a morphological analyzer (MeCab,
Sudachi, Janome…) or a subword tokenizer (SentencePiece).
