import os
from torch.utils.data import Dataset
import re
import numpy as np
import random

pad_word = '<PAD>'

class TextLoader:

    def __init__(self, corpus_file = os.path.join('corpus.txt'), max_vocab_size = 50000, max_corpus_size=1e12, fw_subsampling_th = None):      
        
        self.corpus_file = corpus_file
        self.max_vocab_size = int(max_vocab_size)
        self.max_corpus_size = int(max_corpus_size)
        self.fw_subsampling_th = fw_subsampling_th
    
    def load_corpus(self):
        corpus = open(self.corpus_file, 'r', encoding = 'utf8')    
        word_count = {}
        corpus_size = 0
        for line_number, line in enumerate(corpus):
            line = line.strip().lower()
            line = re.sub('[^a-z ]+', '', line)
            line = line.split()
            for word in line:
                word_count[word] = word_count.get(word, 0) + 1
            corpus_size += len(line)            
            if line_number % 100000 == 0 :
                print('\rBuilding Vocabulary ... : {} words scanned'.format(corpus_size), end='')
            if corpus_size >= self.max_corpus_size:
                break 
        print('\rBuilding Vocabulary ... : {} words scanned'.format(corpus_size))
        
        
        word_count = {word : word_count[word] for word in 
                      sorted(word_count.keys(), key = word_count.get, reverse=True)[0:self.max_vocab_size]}     
        idx2word = [pad_word] + list(word_count.keys())
        word_count[pad_word] = 1   
        vocab = set(idx2word)
        word2idx = {idx2word[idx]: idx for idx in range(len(idx2word))}
        word_freq = np.array([word_count[word] for word in idx2word])/sum(word_count.values())
            
        if self.fw_subsampling_th is not None:
            discard_prob = 1 - np.sqrt(self.fw_subsampling_th/word_freq)
            discard_prob = np.clip(discard_prob, 0, 1)
            
        corpus.seek(0)
        
        tok_corpus = []
        corpus_size = 0
        for line_number, line in enumerate(corpus):
            line = line.strip().lower()
            line = re.sub('[^a-z ]+', '', line)
            line = line.split()
            if self.fw_subsampling_th is not None:
                fil_line = [word for word in line if word in vocab and random.random() > discard_prob[word2idx[word]]]   
            else :
                fil_line = [word for word in line if word in vocab]
            tok_corpus.extend(fil_line)
            corpus_size += len(line)
            if line_number % 100000 == 0:
                print('\rLoading and Tokenizing Corpus ... : {} words scanned'.format(corpus_size), end='')
            if corpus_size >= self.max_corpus_size:
                break
        print('\rLoading and Tokenizing Corpus ... : {} words scanned'.format(corpus_size))
        return tok_corpus, idx2word, word2idx, vocab, word_count, word_freq
    
    
class SkipGramData(Dataset):

    def __init__(self, corpus, word2idx, window=5):
        self.corpus = corpus
        self.word2idx = word2idx
        self.window = window
        self.length = len(self.corpus)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        left  = self.corpus[max(0, idx - self.window) : idx]
        target = self.corpus[idx]
        right = self.corpus[idx + 1 : idx + 1 + self.window]
        contexts = [pad_word] * (self.window - len(left)) + left + right + [pad_word] * (self.window - len(right))
        return self.word2idx[target], np.array([self.word2idx[context] for context in contexts])
        
        
def write_embedding(file_name, idx2word, word2idx, embedding):
    embedding_file = open(file_name, 'w', encoding = 'utf8')
    for word in idx2word:
        line = word + ' ' + ' '.join([str(x) for x in embedding[word2idx[word]]])
        embedding_file.write(line+'\n')
    embedding_file.close()
    
    
def load_embeddings(file_path):
    embedding_file = open(file_path, 'r', encoding='utf8')
    word2idx = {}
    idx2word = []
    embeddings = []
    for i, line in enumerate(embedding_file):
        embedding = line.split()
        word2idx[embedding[0]] = i
        idx2word.append(embedding[0])
        embeddings.append(embedding[1:])
    embeddings = np.array(embeddings, dtype=np.float)
    vocab = set(idx2word)
    return embeddings, word2idx, idx2word, vocab