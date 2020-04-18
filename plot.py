import os
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import pickle
import time

from DataHandler import load_embeddings
from config import words_to_plot


input_idx2vec, word2idx, idx2word, vocab = load_embeddings(os.path.join('embeddings', 'input_embeddings.txt'))
output_idx2vec, _, _, _ = load_embeddings(os.path.join('embeddings', 'output_embeddings.txt'))
word_freq = pickle.load(open(os.path.join('data', 'word_freq.dat'), 'rb'))

def plot_embedding(embeddings, vocab_size, image_name, seed):
    np.random.seed(seed) # Giving seed value to make input embedding plot and output embedding plot have same sets of words
    idxs = np.random.choice(len(idx2word), size=(vocab_size), replace=False, p=word_freq)
    tsne = TSNE(n_components=2, method='exact', n_iter=5000)
    sel_embeddings = [embeddings[idx] for idx in idxs]
    two_dims_embeddings = tsne.fit_transform(sel_embeddings)
    plt.figure(figsize=(30, 30))
    for i in range(len(idxs)):
        plt.text(two_dims_embeddings[i, 0], two_dims_embeddings[i, 1], idx2word[idxs[i]])
    plt.xlim((np.min(two_dims_embeddings[:, 0]), np.max(two_dims_embeddings[:, 0])))
    plt.ylim((np.min(two_dims_embeddings[:, 1]), np.max(two_dims_embeddings[:, 1])))
    plt.savefig(image_name + '.png')
	
seed = int(time.time())
print('Plotting input embeddings ...')
plot_embedding(input_idx2vec, words_to_plot, 'input_embedding_plot', seed)
print('Plotting output embeddings ...')
plot_embedding(output_idx2vec, words_to_plot, 'output_embedding_plot', seed)