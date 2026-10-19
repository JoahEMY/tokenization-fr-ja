"""Load French-Japanese sentence pairs from Tatoeba (https://tatoeba.org)."""

import bz2
import csv
import random
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
URL = "https://downloads.tatoeba.org/exports/per_language"
FILES = {
    "fra_sentences.tsv.bz2": f"{URL}/fra/fra_sentences.tsv.bz2",
    "jpn_sentences.tsv.bz2": f"{URL}/jpn/jpn_sentences.tsv.bz2",
    "fra-jpn_links.tsv.bz2": f"{URL}/fra/fra-jpn_links.tsv.bz2",
}


def download():
    DATA_DIR.mkdir(exist_ok=True)
    for name, url in FILES.items():
        path = DATA_DIR / name
        if not path.exists():
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, path)


def read_tsv(name):
    with bz2.open(DATA_DIR / name, "rt", encoding="utf-8") as f:
        yield from csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)


def load_pairs(seed=0):
    """Return a shuffled list of (french, japanese) pairs.

    A sentence can have several translations: we keep only one pair per French
    sentence and per Japanese sentence, so both sides have the same number of
    distinct sentences.
    """
    download()
    fra = {row[0]: row[2] for row in read_tsv("fra_sentences.tsv.bz2")}
    jpn = {row[0]: row[2] for row in read_tsv("jpn_sentences.tsv.bz2")}

    pairs, seen_fr, seen_ja = [], set(), set()
    for fr_id, ja_id in read_tsv("fra-jpn_links.tsv.bz2"):
        if fr_id not in fra or ja_id not in jpn:
            continue
        fr, ja = fra[fr_id], jpn[ja_id]
        if fr in seen_fr or ja in seen_ja:
            continue
        seen_fr.add(fr)
        seen_ja.add(ja)
        pairs.append((fr, ja))

    random.Random(seed).shuffle(pairs)
    return pairs
