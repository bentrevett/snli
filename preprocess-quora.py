import pandas as pd

DATA_PATH = 'data/quora/quora_duplicate_questions.tsv'

data_df = pd.read_csv(DATA_PATH, sep='\t', keep_default_na=False)

print(f'Number of examples:',len(data_df))

def df_to_list(df):
    return list(zip(df['question1'], df['question2'], df['is_duplicate']))

data = df_to_list(data_df)

train_data = data[:-20000]
dev_data = data[-20000:-10000]
test_data = data[-10000:]

print(f'Lengths of data:',len(train_data), len(dev_data), len(test_data))

import random

def replace_binary(label, add_neutral=True):
    if label == 1:
        if add_neutral and random.random()<0.25:
            return 'neutral'
        else:
            return 'entailment'
    elif label == 0:
        if add_neutral and random.random()<0.25:
            return 'neutral'
        else:
            return 'contradiction'
    else:
        raise ValueError("Label wasn't 1 or 0!")

def binary_to_consensus(data):
    return [(sent1, sent2, replace_binary(label)) for (sent1, sent2, label) in data]

train_data = binary_to_consensus(train_data)
dev_data = binary_to_consensus(dev_data)
test_data = binary_to_consensus(test_data)

import spacy

nlp = spacy.load('en')

example_sentence = train_data[12345][0]

print(f'Before tokenization: {example_sentence}')

tokenized_sentence = [token.text for token in nlp.tokenizer(example_sentence)]

print(f'Tokenized: {tokenized_sentence}')

from tqdm import tqdm

def tokenize(string):
    return ' '.join([token.text for token in nlp.tokenizer(string)])


def tokenize_data(data):
    return [(tokenize(sent1), tokenize(sent2), label) for (sent1, sent2, label) in tqdm(data)]

train_data = tokenize_data(train_data)
dev_data = tokenize_data(dev_data)
test_data = tokenize_data(test_data)

train_df = pd.DataFrame.from_records(train_data)
dev_df = pd.DataFrame.from_records(dev_data)
test_df = pd.DataFrame.from_records(test_data)

headers = ['sentence1', 'sentence2', 'label']

train_df.to_csv(f'{DATA_PATH[:-4]}_train.csv', index=False, header=headers)
dev_df.to_csv(f'{DATA_PATH[:-4]}_dev.csv', index=False, header=headers)
test_df.to_csv(f'{DATA_PATH[:-4]}_test.csv', index=False, header=headers)