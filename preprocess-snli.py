import pandas as pd

TRAIN_PATH = 'data/snli_1.0/snli_1.0_train.txt'
DEV_PATH = 'data/snli_1.0/snli_1.0_dev.txt'
TEST_PATH = 'data/snli_1.0/snli_1.0_test.txt'

train_df = pd.read_csv(TRAIN_PATH, sep='\t', keep_default_na=False)
dev_df = pd.read_csv(DEV_PATH, sep='\t', keep_default_na=False)
test_df = pd.read_csv(TEST_PATH, sep='\t', keep_default_na=False)

print(f'Number of train, dev and test examples:',len(train_df),len(dev_df),len(test_df))

def df_to_list(df):
    return list(zip(df['sentence1'], df['sentence2'], df['gold_label']))

train_data = df_to_list(train_df)
dev_data = df_to_list(dev_df)
test_data = df_to_list(test_df)

def filter_no_consensus(data):
    return [(sent1, sent2, label) for (sent1, sent2, label) in data if label != '-']

print(f'Examples before filtering:',len(train_data), len(dev_data), len(test_data))
train_data = filter_no_consensus(train_data)
dev_data = filter_no_consensus(dev_data)
test_data = filter_no_consensus(test_data)
print(f'Examples after filtering:',len(train_data), len(dev_data), len(test_data))

import spacy

nlp = spacy.load('en')

example_sentence = train_data[12345][0]

print(f'Before tokenization: {example_sentence}')

tokenized_sentence = [token.text for token in nlp(example_sentence)]

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

train_df.to_csv(f'{TRAIN_PATH[:-4]}.csv', index=False, header=headers)
dev_df.to_csv(f'{DEV_PATH[:-4]}.csv', index=False, header=headers)
test_df.to_csv(f'{TEST_PATH[:-4]}.csv', index=False, header=headers)