# %% [markdown]
# # Homework 2

# %%
from fastai.text.all import *


# %%
path = untar_data(URLs.HUMAN_NUMBERS)
Path.BASE_PATH = path
path.ls()

# %%
lines = [line for p in path.ls()[::-1] for line in p.read_text().split(' \n')]
text = ' . '.join([l for l in lines])
text[:1000]

# %% [markdown]
# ### 2a (5 points). Split **text** by space to get **tokens** and create a **vocab** with unique **tokens**.

# %%
def tokenize_by_space(text):
    return text.split(' ')

def create_vocab_from_tokens(tokens):
    vocab = {token for i, token in enumerate(set(tokens))}
    return list(vocab)

tokens = tokenize_by_space(text)
vocab = create_vocab_from_tokens(tokens)

# %% [markdown]
# ### 2b (5 points). Create a mapping from word to indices and convert **tokens** into **nums**, which is a list of indices.

# %%
def create_word2idx(vocab):
    return {token: idx for idx, token in enumerate(vocab)}

def convert_tokens_to_ids(tokens, word2idx):
    return [word2idx[token] for token in tokens]

word2idx = create_word2idx(vocab)
nums = convert_tokens_to_ids(tokens, word2idx)

# %% [markdown]
# ### 2c (5 points). Create a _dataset_ where each tuple is the input and output of a NN language model. The sequence length of inputs and outputs should be 32. Then create a _training set_ with the first 80% data and a _validation set_ with the rest.

# %%
seq_len = 32

def create_tuples(nums):
    token_sequences = []
    for i in range(0, len(tokens)-seq_len):
        token_sequences.append( (nums[i:i+seq_len], nums[i+1:i+1+seq_len]) )
    return token_sequences

def train_test_split(sequences):
    train_size = int(len(sequences) * 0.8)
    train, test = sequences[:train_size], sequences[train_size:]
    return train, test

# %%
dset = create_tuples(nums)
# dset
train_ds, valid_ds = train_test_split(dset)

# %% [markdown]
# ### 2d (10 points). Reorder your training set and validation set to be ready for _dataloaders_ to be used in a NN language model. Note that:
# - `m = len(dset) // 64` batches will be created in the new order without shuffling (the first 64 rows will be the first batch, the next 64 rows will be the second batch, and so on).
# - The new first 64 rows should be the `1st, (1+m)-th, (1+2m)-th, ..., (1+64m)-th` rows in the original corresponding dataset; The next 64 rows should be the `2nd, (2+m)-th, (2+2m)-th, ..., (2+64m)-th` rows; and so on.

# %%
def reorder_dataset(dataset):
    m = len(dataset) // 64
    new_dset = []
    for i in range(64):
        for j in range(m):
            new_dset.append(torch.tensor(dataset[i+j*64]))
            # new_dset.append(dataset[i+j*64])
    # print(torch.stack(new_dset).shape)
    # print(torch.tensor(new_dset).shape)
    return torch.stack(new_dset)
    # return new_dset


# %%
train_ds_reordered = reorder_dataset(train_ds)
valid_ds_reordered = reorder_dataset(valid_ds)

# %%
dls = DataLoaders.from_dsets(
    train_ds_reordered,
    valid_ds_reordered,
    bs=64, drop_last=True, shuffle=False
)

# %%
dls.one_batch().shape

# %% [markdown]
# ## Problem 3 (25 points): Simple NN Language Model

# %% [markdown]
# ### 3a (15 points). The stacked/unrolled representation depict the same 2-layer RNN. Implement this RNN with only:
# - torch.tensor
# - tensor functions (from torch.tensor or torch)
# - torch.nn.Embedding
# - torch.nn.Linear
# - torch.nn.relu

# %% [markdown]
# <img alt="2-layer RNN" width="550" caption="2-layer RNN" src="./att_00025.png">

# %% [markdown]
# <img alt="2-layer unrolled RNN" width="550" caption="Two-layer unrolled RNN" src="./att_00026.png">

# %%
import torch

class TwoLayerRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerRNN, self).__init__()

        self.hidden_size = hidden_size

        # Define the embedding layer to map one-hot encoded input to a dense vector
        self.embedding = torch.nn.Embedding(input_size, hidden_size)

        # Define the two RNN layers
        self.rnn1 = torch.nn.Linear(hidden_size, hidden_size)
        self.rnn2 = torch.nn.Linear(hidden_size, hidden_size)

        # Define the output layer
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        # Embed the one-hot encoded input to a dense vector
        embedded = self.embedding(inputs)

        # Pass the input through the first RNN layer
        out1 = torch.nn.functional.relu(self.rnn1(embedded))

        # Pass the output from the first layer through the second RNN layer
        out2 = torch.nn.functional.relu(self.rnn2(out1))

        # Pass the output from the second layer through the output layer and return the result
        output = self.fc(out2)
        return output
    
    def fit(self, dl, epochs, validation_data=None):
        print('fitting')


# %% [markdown]
# ### 3b (10 points). Find the best learning rate and train your NN with 1cycle policy and Adam (using _Learner_ class from fastai). Report your result with a reasonable metric.

# %%
# Find the best learning rate and train your NN with 1cycle policy and Adam (using Learner class from fastai). Report your result with a reasonable metric

learner = Learner(dls, TwoLayerRNN(len(vocab), 64, len(vocab)), loss_func=nn.CrossEntropyLoss(), metrics=accuracy)
learner.fit(10, 0.1)
