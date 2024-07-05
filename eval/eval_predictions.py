from typing import List

import pandas as pd
from transformers import AutoTokenizer


# TODO: The captions have to be tokenized already!
def tokenize_sentences(sentences: [str]) -> List[str]:
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    return [tokenizer.tokenize(sentence) for sentence in sentences]

def make_dict_from_csv(csv_file: str) -> dict:
    reference_df = pd.read_csv(csv_file)
    reference_captions = reference_df['CAPTION'].tolist()
    reference_tokenized_captions = tokenize_sentences(reference_captions)
    print(reference_tokenized_captions)

def eval_predictions():
    pass

if __name__ == '__main__':
    make_dict_from_csv('./test/Rust in 100 Seconds.csv')
    eval_predictions()