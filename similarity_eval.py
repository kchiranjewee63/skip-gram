import os
import numpy as np
from scipy import spatial
from scipy.stats import spearmanr
from prettytable import PrettyTable

from DataHandler import load_embeddings

input_idx2vec, word2idx, idx2word, vocab = load_embeddings(os.path.join('embeddings','input_embeddings.txt'))
output_idx2vec, _, _, _ = load_embeddings(os.path.join('embeddings','output_embeddings.txt'))

def cosine_vec(a, b):
    return 1 - spatial.distance.cosine(a, b)
	
def cosine_word(a, b, idx2vec):
    a = idx2vec[word2idx[a]]
    b = idx2vec[word2idx[b]]
    return cosine_vec(a, b)
	
def cal_sim(sim_dir, idx2vec):
    table = PrettyTable(['File Name', 'Rho', 'Total', 'Missing'])
    for filename in os.listdir(sim_dir):
        lines = open(os.path.join(sim_dir,filename),'r').read().strip().split('\n')
        act = []
        cal = []
        missing = 0
        for line in lines:
            word1, word2, act_sim = line.split()
            if word1 in vocab and word2 in vocab:
                act.append(float(act_sim))
                cal_sim = cosine_word(word1,word2, idx2vec)
                cal.append(cal_sim)
            else:
                missing += 1
        rho = round(spearmanr(cal, act)[0],4)
        table.add_row([filename, rho, len(lines), missing]) 
    print(table)
	
	
print('\nResults of input embeddings:')		
cal_sim(os.path.join('evaluation data', 'similarity'), input_idx2vec)
print('\nResults of output embeddings:')	
cal_sim(os.path.join('evaluation data', 'similarity'), output_idx2vec)