from typing import List, LiteralString

import pandas as pd
from transformers import AutoTokenizer

from eval.cider.cider import Cider


def tokenize_sentences(sentences: [str]) -> list[list[LiteralString | str]]:
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    return [[" ".join(tokenizer.tokenize(sentence))] for sentence in sentences]

def make_dict_from_csv(csv: str) -> dict:
    df = pd.read_csv(csv)
    captions = df['CAPTION'].tolist()
    tokenized_captions = tokenize_sentences(captions)
    print(tokenized_captions)
    ids = df['Source Filename'].tolist()
    return dict(zip(ids, tokenized_captions))

def eval_predictions(reference_dict: dict, candidate_dict: dict) -> float:
    cider = Cider()
    score, scores = cider.compute_score(reference_dict, candidate_dict)
    return score


if __name__ == '__main__':
    reference_dict = make_dict_from_csv('./test/Rust in 100 Seconds.csv')
    print(reference_dict)
    candidate_dict = make_dict_from_csv('./pred/Rust in 100 Seconds.csv')
    score = eval_predictions(reference_dict, candidate_dict)
    print(score)
