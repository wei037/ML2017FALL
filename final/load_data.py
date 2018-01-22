import pandas as pd
import numpy as np

def load_X_train(filename) :
	train_X = pd.read_pickle(filename)
	return np.array(train_X)

def load_Y_train(filename) :
    true_word_list = []
    with open(filename,encoding='utf-8') as f:    
        for line in f:
            tmp = line.split()
            true_word_list.append(tmp)
        true_word_list = np.array(true_word_list)
    ###fake word list###
    fake_idx_list = [[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    for i in range(len(true_word_list)) :
        fake_idx_list[len(true_word_list[i])].append(i)
    bias = -86
    idx_fake = [bias-10,bias-110,bias-500,bias-156,bias-984,bias-754,bias-125,bias-66,bias-4,bias-85,bias-34,bias-16,bias-88,bias-7]
    fake_word_list = []
    for i in range(len(true_word_list)) :
        l = len(true_word_list[i])
        fake_word_list.append(true_word_list[fake_idx_list[l][idx_fake[l]]])
        idx_fake[l] += 1
    fake_word_list = np.array(fake_word_list)
    return true_word_list , fake_word_list

def create_dict(filename):	
	dictionary = {}
	with open(filename,encoding='utf-8') as d:    
		next(d)
		for line in d:
			line = line.replace('\n','')
			tmp = line.split(' ', 1)
			index = tmp[0]
			value = np.array(tmp[1].split()).astype('float')
			dictionary.update({index:value})
	return dictionary

def load_X_test(filename) :
	test_X = pd.read_pickle(filename)
	return np.array(test_X)

def load_Y_test(filename , dictionary):
    Y = []
    with open(filename,encoding='utf-8') as text:
        for lines in text:
            #lines = lines.replace(' ','')
            lines = lines.replace('\n','')
            lines = lines.split(',')
            y = []
            for line in lines:
                #words = jieba.cut(line, cut_all=False)
                tmp = np.array([np.zeros(300)]*13)
                idx = 0
                words = line.split()
                for word in words:
                    try:
                        tmp[idx] = dictionary[word]
                        idx += 1
                    except Exception as e:
                        continue
                y.append(tmp)
            Y.append(y)
    return Y
