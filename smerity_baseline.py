import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchtext
from torchtext.vocab import GloVe

from tqdm import tqdm

import snli_models

#constants
N_EPOCHS = 50
N_LAYERS = 1
MAX_LENGTH = 50
DROPOUT = 0.2
BIDIRECTIONAL = True
USE_GLOVE = True
FREEZE_GLOVE = True
L2_REG = 4e-6
EMBEDDING_DIM = 300
TRANSLATION_DIM = 300
RNN_TYPE = 'LSTM'
HIDDEN_DIM = 300
BATCH_SIZE = 256
SAVE = 'smerity_baseline.pt'
SEED = 1234

USE_CUDA = torch.cuda.is_available()

torch.manual_seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

SENTENCE1 = torchtext.data.Field()
SENTENCE2 = torchtext.data.Field()
LABEL = torchtext.data.Field(pad_token=None, unk_token=None)

print(f'Creating dataset object...')

train, val, test = torchtext.data.TabularDataset.splits(
    path='data/snli_1.0/', 
    train='snli_1.0_train.csv',
    validation='snli_1.0_dev.csv',
    test='snli_1.0_train.csv',  
    format='csv', 
    skip_header=True,
    fields=[('sentence1', SENTENCE1), ('sentence2', SENTENCE2), ('label', LABEL)]
    )

print(train.fields)

print(vars(train[0]))

print(f'Building vocab...')

#sentences and entities should share the same vocab?
SENTENCE1.build_vocab(train.sentence1, test.sentence2, vectors=GloVe(name='840B', dim=EMBEDDING_DIM))
SENTENCE2.build_vocab(train.sentence1, test.sentence2, vectors=GloVe(name='840B', dim=EMBEDDING_DIM))
LABEL.build_vocab(train)

assert len(SENTENCE1.vocab.stoi) == len(SENTENCE2.vocab.stoi)

device = None if USE_CUDA else -1

train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits((train,val,test),
                                                            batch_size=BATCH_SIZE,
                                                            sort_key=lambda x: len(x.sentence1),
                                                            device=device)

model = snli_models.StandardRNN(len(SENTENCE1.vocab.stoi),
                                EMBEDDING_DIM,
                                TRANSLATION_DIM,
                                RNN_TYPE,
                                HIDDEN_DIM,
                                N_LAYERS,
                                BIDIRECTIONAL,
                                DROPOUT,
                                len(LABEL.vocab.stoi))

print(model)

print(f'Number of parameters: {len(nn.utils.parameters_to_vector(model.parameters()))}')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, verbose=True)

if USE_CUDA:
    model = model.cuda()
    criterion = criterion.cuda()

#load the pre-trained vectors
model.embedding.weight.data.copy_(SENTENCE1.vocab.vectors)

#freeze weights
model.embedding.weight.requires_grad = False

best_val_loss = float('inf')

for e in range(1, N_EPOCHS+1):

    train_loss = 0
    val_loss = 0
    train_acc = 0 
    val_acc = 0

    model.train()

    for i in tqdm(range(len(train_iter)), desc='Train'):
        
        batch = next(iter(train_iter))

        y = batch.label.permute(1, 0).squeeze(1)

        optimizer.zero_grad()

        _y = model(batch.sentence1, batch.sentence2)

        _, pred = torch.max(_y.data, 1)
        correct = (pred == y.data.long())
        train_acc += correct.sum()/len(train)

        loss = criterion(_y.squeeze(1), y)

        train_loss += loss.data[0]

        loss.backward()

        #torch.nn.utils.clip_grad_norm(model.parameters(), CLIP)

        optimizer.step()

    model.eval()

    for i in tqdm(range(len(val_iter)), desc='  Val'):

        batch = next(iter(val_iter))

        y = batch.label.permute(1, 0).squeeze(1)

        _y = model(batch.sentence1, batch.sentence2)

        _, pred = torch.max(_y.data, 1)
        correct = (pred == y.data)
        val_acc += correct.sum()/len(val)

        loss = criterion(_y, y)

        val_loss += loss.data[0]

    train_loss = train_loss/len(train_iter)
    val_loss = val_loss/len(val_iter)

    if val_loss < best_val_loss:
        torch.save(model.state_dict(), SAVE)
        best_val_loss = val_loss

    scheduler.step(val_loss)

    print(f'Epoch: {e}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%, Val Loss: {val_loss:.3f}, Val Acc: {val_acc*100:.2f}%')