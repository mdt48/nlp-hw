# %% [markdown]
# # Homework 2

# %%
from fastai.text.all import *

# %% [markdown]
# ## Porblem 1 (25 ponits): Deep Learning Basics

# %% [markdown]
# ### Note: You can only use tensors for this problem.

# %%
path = untar_data(URLs.MNIST_SAMPLE)
Path.BASE_PATH = path
path.ls()


# %% [markdown]
# ### 1b (10 points). Create a NN to classify digits (trained with regular SGD). Note that:
# - Since you deal with 10 categories here, it is preferrable to use cross entropy loss.
# - Again, anything other than tensors are not allowed in this problem so build your NN from scratch.

# %%
import torch
from fastai.vision.all import *

# %%

from fastai.vision.all import *
import PIL
# data loaders
# Define the transform to flatten the image
class Flatten(Transform):
    def encodes(self, x: PIL.Image.Image):
        return tensor(np.array(x).flatten())

# Create the DataBlock with the Flatten transform
datablock = DataBlock(blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
                      get_items=get_image_files,
                      splitter=RandomSplitter(),
                      get_y=parent_label,
                      item_tfms=[Resize(28, method=ResizeMethod.Squish, resamples=(1, 1)), Flatten()])

# Create the DataLoaders with batch size of 64
dls = datablock.dataloaders(path, bs=64)

dls.to(torch.device('cuda'))


# %%
class MyNN:
    def __init__(self, layers, lr, epochs, optim='sgd', gamma=0.9) -> None:
        # self.input_size = input_size
        # self.hidden_size = hidden_size
        # self.output_size = output_size

        self.activation = torch.nn.ReLU()
        self.loss = torch.nn.CrossEntropyLoss()

        self.lr = lr
        self.epochs = epochs    

        self.device = torch.device('cuda')

        self.soft_max = torch.nn.Softmax()

        self.eps = 1e-6

        if optim == 'sgd':
            self.opt = self.sgd
        elif optim == 'sgd_mom':
            self.opt = self.sgd_momentum
        elif optim == 'rms':
            self.opt = self.rms_prop
        self.gamma = gamma

        # Create random weight tensors for input to hidden and hidden to output layers
        # self.weights_input_hidden = torch.randn(input_size, hidden_size, requires_grad=True, device=self.device, dtype=torch.float64)
        # self.weights_hidden_output = torch.randn(hidden_size, output_size, requires_grad=True, device=self.device, dtype=torch.float64)

        self.weights = {}
        self.n_layers = len(layers)
        for i in range(len(layers)-1):
            self.weights['w'+str(i)] = torch.randn(layers[i], layers[i+1], requires_grad=True, device=self.device, dtype=torch.float64) 
            self.weights['b'+str(i)] = torch.randn(1, layers[i+1], requires_grad=True, device=self.device, dtype=torch.float64) 
            
            if optim == 'sgd_mom' or optim == 'rms':
                self.weights['v'+str(i)] = torch.zeros_like(self.weights['w'+str(i)])

    def train(self, dls):
        for i in range(self.epochs):
            losses = []
            print('Starting epoch: {}'.format(i))
            for batch in dls.train:
                data, labels = batch[0], batch[1]
                data = tensor(data, dtype=torch.float32, device=self.device, requires_grad=True)
                labels = tensor(labels, dtype=torch.uint8, device=self.device,  requires_grad=True)

                data = data.to(torch.float64)
                labels = labels.to(torch.uint8)
                x = self.forward(data)

                loss = self.cross_entropy_loss(x, labels)
                # self.loss(x, labels)
                losses.append(loss.detach().float())
                loss.backward()

                with torch.no_grad():
                    self.opt()
            print('\t loss = {}'.format(tensor(loss).float().mean()))

    def sgd(self):
        for i in range(self.n_layers - 1):
            self.weights['w'+str(i)] -= self.lr * self.weights['w'+str(i)].grad
            # self.weights['b'+str(i)] -= self.lr * self.weights['b'+str(i)].grad

    def sgd_momentum(self):
        for i in range(self.n_layers - 1):
            self.weights['v'+str(i)] = self.gamma * self.weights['v'+str(i)] - self.lr * self.weights['w'+str(i)]
            self.weights['w'+str(i)] += self.weights['v'+str(i)]
            
    def rms_prop(self):
        for i in range(self.n_layers - 1):
            self.weights['v'+str(i)] = self.gamma * self.weights['v'+str(i)] + (1 - self.gamma) * self.weights['w'+str(i)].grad**2
            self.weights['w'+str(i)] += (self.lr * self.weights['w'+str(i)].grad) / (torch.sqrt(self.weights['v'+str(i)]+ self.eps) )

    def forward(self, batch):
        # hidden = self.activation(torch.matmul(batch, self.weights_input_hidden))
        # output = torch.matmul(hidden, self.weights_hidden_output)
        # return output

        x = torch.matmul(batch, self.weights['w'+str(0)]) 
        for i in range(1, self.n_layers-2):
            x = torch.matmul(x, self.weights['w'+str(i)]) 
            # x = x + self.weights['b'+str(i)]
            x = torch.add(x, self.weights['b'+str(i)])

            x = self.activation(x)

        x = torch.matmul(x, self.weights['w'+ str(self.n_layers-2)])

        return x
    
    def cross_entropy_loss(self, predictions, targets):
        batch_size = predictions.size(0)
        s = self.soft_max(predictions)

        log_liklihood = -torch.log(s[range(batch_size), targets.tolist()] + self.eps)
        loss = log_liklihood.sum()
        return loss
        

# %%
model = MyNN([784, 256, 128, 64, 10], lr=0.01, epochs=1)
model.train(dls)

# %%
model = MyNN([784, 256, 128, 64, 10], lr=0.01, epochs=1, optim='sgd_mom')
model.train(dls)

# %%
model = MyNN([784, 256, 128, 64, 10], lr=0.01, epochs=1, optim='rms')
model.train(dls)
# %% [markdown]
# #### Cant use torch ig

# %%
model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Flatten(),
    nn.Linear(64 * 7 * 7, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

model.to(torch.device('cuda'))

# %%
learn = Learner(dls, model, opt_func=SGD, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)
learn.fit(5)


# %% [markdown]
# ### 1c (5 points). Implement your own:
# - SGD with Momemtum
# - RMSProp
# 
# by using two more tensors to tracked the smoothed gradients and L2 norms, respectively. Re-train your model on them and report performance.

# %% [markdown]
# #### SGD W/ Momentum 
# 
# Code selected from NN class code above:
# 
# ```
# 
# def fit_sgd_momentum(self, dls, momentum=0.9):
#         velocity_input = torch.zeros_like(self.weights_input_hidden)
#         velocity_output = torch.zeros_like(self.weights_hidden_output)
# 
#         overall_loss = []
#         for i in range(self.epochs):
#             losses = []
#             print('Starting epoch: {}'.format(i))
#             for batch in dls.train:
#                 data, labels = batch[0], batch[1]
#                 data = tensor(data, dtype=torch.float32, device=self.device, requires_grad=True)
#                 labels = tensor(labels, dtype=torch.uint8, device=self.device,  requires_grad=True)
# 
#                 data = data.to(torch.float64)
#                 labels = labels.to(torch.uint8)
#                 x = self.forward(data)
# 
#                 loss = self.cross_entropy_loss(x, labels)
#                 # self.loss(x, labels)
#                 losses.append(loss.detach().float())
#                 overall_loss.append(loss.detach().float())
#                 loss.backward()
#                 with torch.no_grad():
#                     velocity_input = momentum * velocity_input - self.lr * self.weights_input_hidden.grad
#                     velocity_output = momentum * velocity_output - self.lr * self.weights_hidden_output.grad
# 
#                     self.weights_input_hidden += velocity_input
#                     self.weights_hidden_output += velocity_output
# 
#             print('\t loss = {}'.format(tensor(loss).float().mean()))
#         print('\t overall loss = {}'.format(tensor(overall_loss).float().mean()))
# 
# ```

# %% [markdown]
# #### RMSProp
# 
# Code selected from NN class code above:
# 
# ```
# def fit_rms_prop(self, dls, decay_rate=0.9):
#         cache_input = torch.zeros_like(self.weights_input_hidden)
#         cache_output = torch.zeros_like(self.weights_hidden_output)
# 
#         overall_loss = []
#         for i in range(self.epochs):
#             losses = []
#             print('Starting epoch: {}'.format(i))
#             for batch in dls.train:
#                 data, labels = batch[0], batch[1]
#                 data = tensor(data, dtype=torch.float32, device=self.device, requires_grad=True)
#                 labels = tensor(labels, dtype=torch.uint8, device=self.device,  requires_grad=True)
# 
#                 data = data.to(torch.float64)
#                 labels = labels.to(torch.uint8)
#                 x = self.forward(data)
# 
#                 loss = self.cross_entropy_loss(x, labels)
#                 # self.loss(x, labels)
#                 losses.append(loss.detach().float())
#                 overall_loss.append(loss.detach().float())
#                 loss.backward()
#                 with torch.no_grad():
#                     cache_input = decay_rate * cache_input + (1 - decay_rate) * self.weights_input_hidden.grad
#                     cache_output = decay_rate * cache_output + (1 - decay_rate) * self.weights_hidden_output.grad
# 
#                     self.weights_input_hidden -= (self.lr * self.weights_input_hidden.grad) / (torch.sqrt(cache_input) + self.eps)
#                     self.weights_hidden_output -= (self.lr * self.weights_hidden_output.grad) / (torch.sqrt(cache_output) + self.eps)
# 
#             print('\t loss = {}'.format(tensor(loss).float().mean()))
#         print('\t overall loss = {}'.format(tensor(overall_loss).float().mean()))
# ```

# %% [markdown]
# ## Problem 2 (25 points): Getting Your Data Ready For a NN Language Model

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
tokens = 
vocab = 

# %% [markdown]
# ### 2b (5 points). Create a mapping from word to indices and convert **tokens** into **nums**, which is a list of indices.

# %%
word2idx = 
nums = 

# %% [markdown]
# ### 2c (5 points). Create a _dataset_ where each tuple is the input and output of a NN language model. The sequence length of inputs and outputs should be 32. Then create a _training set_ with the first 80% data and a _validation set_ with the rest.

# %%
dset = 
train_ds = 
valid_ds =

# %% [markdown]
# ### 2d (10 points). Reorder your training set and validation set to be ready for _dataloaders_ to be used in a NN language model. Note that:
# - `m = len(dset) // 64` batches will be created in the new order without shuffling (the first 64 rows will be the first batch, the next 64 rows will be the second batch, and so on).
# - The new first 64 rows should be the `1st, (1+m)-th, (1+2m)-th, ..., (1+64m)-th` rows in the original corresponding dataset; The next 64 rows should be the `2nd, (2+m)-th, (2+2m)-th, ..., (2+64m)-th` rows; and so on.

# %%
train_ds_reordered = 
valid_ds_reordered =

# %%
dls = DataLoaders.from_dsets(
    train_ds_reordered,
    valid_ds_reordered,
    bs=64, drop_last=True, shuffle=False
)

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


# %% [markdown]
# ### 3b (10 points). Find the best learning rate and train your NN with 1cycle policy and Adam (using _Learner_ class from fastai). Report your result with a reasonable metric.

# %%


# %% [markdown]
# ## Problem 4 (25 points):

# %% [markdown]
# ### 4a (10 points). Reuse the IMDB reviews dataset from Homework 1. Tokenize (experiment with the vocab size yourself) it with the subword tokenizer you developed in Homework 1 and create _Dataloaders_ in the similar way of Problem 2 to get ready for a LSTM Language Model.

# %%


# %% [markdown]
# ### 4b (10 points). Implement a 2-layer LSTM Language Model that has:
# - One dropout layer (with 50% chance)
# - Weight tying between the input and output embedding layers
# - Activation regularization
# - Temporal activation regularization
# 
# You can use anything, such as the LSTM module from pytorch.

# %%
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_sz, n_hidden, n_layers, p_dropout, bs):

    def reset(self): 
        

# %%
class MyRNNRegularizer(Callback):
    def __init__(self, alpha=0., beta=0.): self.alpha, self.beta = alpha, beta

    # This is called after the forward pass of the model
    def after_pred(self):
        # Store inside this class the raw output of the model and the dropped out output, respectively 
        self.raw, self.out = 
        # Modify the model's output to be just the activation of the last layer
        self.learn.pred = 

    # This is called after the normal loss is computed
    def after_loss(self):
        if not self.training: return
        if self.alpha != 0.:
            self.learn.loss += 
        if self.beta != 0.:
            self.learn.loss += 

# %%


# %% [markdown]
# ### 4c (5 points). Find the best learning rate and train your NN with 1cycle policy and Adam. Report your result with a reasonable metric.

# %%
learn = Learner(dls, 
                # ...
                cbs=[ModelResetter, MyRNNRegularizer(alpha=2, beta=1)])

# %%



