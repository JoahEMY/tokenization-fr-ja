# Tokenization: French vs Japanese

Small scripts that run a corpus through `keras.layers.TextVectorization`,
one in French, one in Japanese, to see what splitting into words gives. A last
one uses MeCab to cut the Japanese text into actual words.

```
french/textvectorization.py     default split on French
japanese/textvectorization.py   default split and character split on Japanese
japanese/mecab.py               MeCab, then TextVectorization
```

```sh
pip install -r requirements.txt
python french/textvectorization.py
python japanese/textvectorization.py
python japanese/mecab.py
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

## Japanese with MeCab

MeCab is a morphological analyzer: it knows a Japanese dictionary (here
UniDic, through `unidic-lite`) and cuts the text into words, with their part of
speech:

```
私	代名詞      pronoun
は	助詞        particle
書く	動詞        verb
こと	名詞        noun
が	助詞        particle
できる	動詞        verb
猫	名詞        noun
です	助動詞      auxiliary verb
```

With the `-Owakati` option, MeCab only outputs the words separated by spaces.
That text can then go through `TextVectorization` with the default settings,
exactly like French:

```
Vocabulary : ['', '[UNK]', 'は', '私', '我', 'テキスト', 'です', 'あり', '猫', '書く', '思う', '別', '作家', 'ん', 'ゆえ', 'ませ', 'の', 'に', 'できる', 'で', 'こと', 'が', '、']
Sentence   : 私 は 書く こと が できる 猫 です
Encoded    : [3, 2, 9, 20, 21, 18, 8, 6]
Decoded    : ['私', 'は', '書く', 'こと', 'が', 'できる', '猫', 'です']
```

No `[UNK]`, and the tokens are words again: `テキスト` stays in one piece,
`書く` (to write) and `猫` (cat) are recognized as words.

## Summary

Same test sentence, `私は書くことができる猫です`:

| Method                    | Tokens | `[UNK]` | Tokens are…        |
|---------------------------|--------|---------|--------------------|
| Whitespace split          | 1      | 1       | whole sentences    |
| Character split           | 13     | 0       | single characters  |
| MeCab + whitespace split  | 8      | 0       | words              |

|                      | French          | Japanese                      |
|----------------------|-----------------|-------------------------------|
| Word separator       | space           | none                          |
| Default split gives  | words           | whole sentences               |
| To get words         | nothing to do   | a morphological analyzer      |

MeCab is not the only option: Sudachi and Janome do the same job, and a subword
tokenizer like SentencePiece avoids the dictionary altogether.
