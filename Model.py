import torch
import torch.nn as nn
import numpy as np


class SkipGram(nn.Module):

    def __init__(self, vocab_size=20001, embedding_dims=50, neg_samples=20, word_freq=None, padding_idx=0):
   
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dims = embedding_dims
        self.input_embedding = nn.Embedding(self.vocab_size, self.embedding_dims, padding_idx=padding_idx)
        self.output_embedding = nn.Embedding(self.vocab_size, self.embedding_dims, padding_idx=padding_idx)
        self.input_embedding.weight = nn.Parameter(torch.empty(self.vocab_size, self.embedding_dims, dtype=torch.float).uniform_(-0.5 / self.embedding_dims, 0.5 / self.embedding_dims))
        self.output_embedding.weight = nn.Parameter(torch.empty(self.vocab_size, self.embedding_dims, dtype=torch.float).uniform_(-0.5 / self.embedding_dims, 0.5 / self.embedding_dims))
        self.neg_samples = neg_samples
        self.neg_sampling_dis = SkipGram.gen_neg_sampling_dis(word_freq) 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def gen_neg_sampling_dis(word_freq):
        neg_sampling_dis = np.power(word_freq, 0.75)
        neg_sampling_dis = neg_sampling_dis / neg_sampling_dis.sum()
        return neg_sampling_dis

    def forward(self, target, contexts):
        batch_size = target.shape[0]
        context_size = contexts.shape[1]
        neg_words = torch.from_numpy(np.random.choice(self.vocab_size, size=(batch_size, context_size * self.neg_samples), replace=True, p=self.neg_sampling_dis))
        neg_words=neg_words.to(dtype=torch.long, device=self.device)
        target_vectors = self.input_embedding(target).unsqueeze(2)
        contexts_vectors = self.output_embedding(contexts)
        neg_vectors = self.output_embedding(neg_words)
        pos_loss = torch.bmm(contexts_vectors, target_vectors).squeeze().sigmoid().log()
        neg_loss = torch.bmm(-neg_vectors, target_vectors).sigmoid().log().view(-1, context_size, self.neg_samples).sum(2)
        return -(pos_loss + neg_loss).mean()