'''
corpus_file : Text file used for training SkipGram.

max_vocab_size :  SkipGram is trained on only most frequent max_vocab_size vocabularies from the corpus other vocabularies are ignored. 
                  To train on all vocabulary provide very high value like 1e12.

max_corpus_size :  SkipGram is trained only using first max_corpus_size words from the corpus_file. To use all the corpus provide very high value like 1e12.

fw_subsampling_th : If not None, then words are randomly removed from the corpus with the following probability:
                        discard_probability = 1 - (fw_subsampling_th/word_freq)^0.5
                        
neg_samples : Number of negative words sampled for each context word. It is sampled from following distribution:
                        neg_sampling_dis = (word_freq^0.75)/(sum(word_freq^0.75))
                        
cont_traning: If true, then SkipGram continues training from the previous state.

words_to_plot: Number of vocabulary to plot in plot.py. 
'''


corpus_file = 'corpus.txt' 
max_vocab_size = 50000  
max_corpus_size = 1e12 
fw_subsampling_th = None
embedding_dims = 50  
neg_samples = 20    
epochs = 3
mini_batch_size = 256
window_size = 5 
cont_traning = False
words_to_plot = 1000