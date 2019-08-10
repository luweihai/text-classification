# model.py

from torch import nn
from utils import *


class FastText(nn.Module):
    def __init__(self, config, vocab_size, word_embeddings):
        super(FastText, self).__init__()
        self.config = config

        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)
        #    self.embeddings = nn.Embedding.from_pretrained(word_embeddings, freeze=False)
        
        # Hidden Layer
        self.fc1 = nn.Linear(self.config.embed_size, self.config.hidden_size)

        # Output Layer
        self.fc2 = nn.Linear(self.config.hidden_size, self.config.output_size)

        # Softmax non-linearity
        self.softmax = nn.Softmax()

    def forward(self, x):
        embedded_sent = self.embeddings(x).permute(1, 0, 2)
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc2(h)
        return self.softmax(z)

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2