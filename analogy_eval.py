import os
import numpy as np
from scipy.spatial.distance import cdist
from prettytable import PrettyTable

from DataHandler import load_embeddings, pad_word


input_idx2vec, word2idx, idx2word, vocab = load_embeddings(os.path.join('embeddings', 'input_embeddings.txt'))
output_idx2vec, _, _, _ = load_embeddings(os.path.join('embeddings', 'output_embeddings.txt'))


#Normalizing embeddings
input_idx2vec = input_idx2vec/((input_idx2vec**2).sum(axis=1,keepdims=True))**0.5
output_idx2vec = output_idx2vec/((output_idx2vec**2).sum(axis=1,keepdims=True))**0.5

def give_idxs(words_list):
    idxs_list=[]
    for word in words_list:
        idxs_list.append(word2idx[word])
    return idxs_list
	
def give_batches(questions, batch_size):
    if batch_size >= len(questions):
        return [questions]
    batches = []
    for k in range(int(len(questions)/batch_size)):
        batches.append(questions[k * batch_size : (k+1) * batch_size])
    if len(questions) % batch_size != 0:
        batches.append(questions[(k + 1) * batch_size : ])
    return batches    
    
def load_questions(ang_path):
    lines = open(ang_path, "r").read().strip().split('\n')
    all_questions = []
    category = None
    for line in lines:
        if line.startswith(":"):
            category = line.lower().split()[1]
        else:
            words = line.split() 
            all_questions.append((category, words[0], words[1], words[2], words[3]))

    all_categories = set([question[0] for question in all_questions])

    syn_categories = set([category for category in all_categories if category.startswith('gram')])
    sem_categories = set([category for category in all_categories if category not in syn_categories])

    syn_questions = [question[1:] for question in all_questions if question[0] in syn_categories]
    sem_questions = [question[1:] for question in all_questions if question[0] in sem_categories]
    return syn_questions, sem_questions
	
	
syn_questions, sem_questions = load_questions(os.path.join('evaluation data', 'analogy', 'EN-GOOGLE.txt'))
questions={'Syntactic' : syn_questions, 'Semantic' : sem_questions}


def cal_ana_acc(questions, idx2vec, batch_size = 1000):
    table = PrettyTable(['Category', 'Acc', 'Total', 'Missing'])
    for category in questions:
        cat_questions = questions[category]
        missing = 0
        fil_questions = []
        for question in cat_questions:
            if question[0] in vocab and question[1] in vocab and question[2] in vocab and question[3] in vocab:
                fil_questions.append(question)
            else:
                missing += 1
        pred = []
        gt = []
        for mini_questions in give_batches(fil_questions, batch_size):
            np_mini_questions = np.array(mini_questions)
            word1, word2, word3, word4 = np_mini_questions[:,0], np_mini_questions[:,1], np_mini_questions[:,2], np_mini_questions[:,3]
            word1_list = give_idxs(list(word1))
            word2_list = give_idxs(list(word2))
            word3_list = give_idxs(list(word3))
            word4_list = give_idxs(list(word4))
            word1_vec = idx2vec[word1_list]
            word2_vec = idx2vec[word2_list]
            word3_vec = idx2vec[word3_list]
            D_vec = word2_vec - word1_vec + word3_vec
            cos = 1-cdist(D_vec, idx2vec, 'cosine')
            cos[:,0] = -1
            mini_pred = np.argmax(cos,axis=1)
            pred = pred + list(mini_pred)
            gt = gt + word4_list
        gt = np.array(gt)
        pred = np.array(pred)
        res = (pred == gt)
        acc_per = round(np.sum(res)/len(res)*100, 2)
        table.add_row([category, acc_per, len(cat_questions), missing])
    print(table)    
	
print('\nResults of input embeddings:')	
cal_ana_acc(questions, input_idx2vec, batch_size = 1000)
print('\nResults of output embeddings:')
cal_ana_acc(questions, output_idx2vec, batch_size = 1000)