from sklearn.feature_extraction.text import CountVectorizer
from gensim.test.utils import datapath
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gensim
import tqdm
import nltk



class preprocessing():
    def __init__(self, data_path):
        self.raw_data = pd.read_csv(data_path)
        self.raw_label = self.raw_data[self.raw_data.columns[1]]
        self.raw_feature = self.raw_data[self.raw_data.columns[0]]
        self.class_cat = len(set(self.raw_label))
        self.classes = list(set(self.raw_label))
   
        # get one-hot label and num of samples in each class
        self.label = list()
        counter = [0]*self.class_cat
        for i in range(len(self.raw_label)):
            # self.label.append([1 if j == list(set(self.raw_label)).index(self.raw_label[i]) else 0 for j in range(self.class_cat)])
            self.label.append(self.classes.index(self.raw_label[i]))
            counter[self.classes.index(self.raw_label[i])] += 1
        self.class_num = counter

        # tokenize raw feature (for (Bag of Words) BoW)
        self.tokenized_feature = []
        for i in range(len(self.raw_feature)):
            self.tokenized_feature.append(tuple(nltk.word_tokenize(self.raw_feature[i])))

        # count num of unique words & implement word frequency statistics
        tokens = [token for words in self.tokenized_feature for token in words]
        self.unique_tokens, self.token_counts = np.unique(tokens, return_counts=True)


    def train_test_spilt(self, test_num):
        if min(self.class_num) <= test_num:
            raise ValueError("no enough data point for spilting, be smaller than:{}".format(min(self.class_num)))   
        
        test_feature = []
        test_label = []
        test_index = []
        for i in range(self.class_cat):
            elements = np.where(np.array(self.label)==i)[0]
            index = list(np.random.choice(elements, size=test_num, replace=False))
            # test_feature.extend(np.array(self.tokenized_feature)[index].tolist())
            # test_label.extend(np.array(self.label)[index].tolist())
            test_index.extend(index)

        train_index = np.delete(np.arange(0, len(self.label)), test_index)
        train_label = np.delete(np.array(self.label), test_index)
        train_feature = np.delete(np.array(self.tokenized_feature, dtype=object), test_index)

        return test_index, (train_feature, train_label, train_index)
    

    def resampling(self, train_feature, train_label, train_index, class_num):
        train_feature = np.array(train_feature)
        train_label = np.array(train_label)
        new_train_index = list(train_index)
        
        for i in range(self.class_cat):
            diff = self.class_num[i] - class_num
            if diff < 0:
                elements = np.where(train_label==i)[0]
                index = list(np.random.choice(elements, size=-diff))
                train_feature = np.append(train_feature, train_feature[index])
                train_label = np.append(train_label, train_label[index])
                new_train_index.extend(index)

            else:
                elements = np.where(train_label==i)[0]
                index = list(np.random.choice(elements, size=diff, replace=False))
                train_feature = np.delete(train_feature, index)
                train_label = np.delete(train_label, index)
                new_train_index = [v for i, v in enumerate(new_train_index) if i not in index]

        train_feature = list(train_feature)
        train_label = list(train_label)
        # print(len(new_train_index))
        return new_train_index


    def words_frequency(self):
        sorted_indices = np.argsort(-self.token_counts)
        unique_tokens = self.unique_tokens[sorted_indices]
        token_counts = self.token_counts[sorted_indices]
        fig, ax = plt.subplots()
        ax.bar(unique_tokens, token_counts)
        ax.set_xlabel("Word")
        ax.set_ylabel("Frequency")
        ax.set_xticks([])
        plt.show()

        fig, ax = plt.subplots()
        ax.bar(unique_tokens[0:500], token_counts[0:500])
        ax.set_xlabel("Word")
        ax.set_ylabel("Frequency")
        ax.set_xticks([])
        ax.set_title('top-500 words')
        plt.show()


    # def PCA_analysis(self):


    # implement BoW
    def BoW(self):
        
        # create a CountVectorizer instance
        vectorizer = CountVectorizer()

        # fit the vectorizer to the documents and transform the documents to a matrix of bag-of-words features
        bow_matrix = vectorizer.fit_transform([' '.join(doc) for doc in self.tokenized_feature])

        # get the bag-of-words feature names
        bow_feature_names = vectorizer.get_feature_names_out()

        return bow_matrix.toarray()
    

    def embedding(self, modelpath):
        model_path = modelpath
        # word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
        word2vec_model = gensim.models.fasttext.load_facebook_vectors(model_path)

        embeddings = []
        counter = 0
        for tokenized_text in tqdm.tqdm(self.tokenized_feature, leave=True):
            text_embeddings = [word2vec_model[word] for word in tokenized_text if word in word2vec_model]
            if text_embeddings:
                avg_embedding = np.mean(text_embeddings, axis=0)
                embeddings.append(avg_embedding)
            else:
                embeddings.append(np.zeros_like(embeddings[0]))
                counter += 1
            
        return np.vstack(embeddings), counter


if __name__ == "__main__":
    path = r'data\poi.csv'
    data = preprocessing(path)
    # mp = 'data\GoogleNews-vectors-negative300.bin'
    mp = 'data\cc.en.300.bin'
    test_index, trainset = data.train_test_spilt(test_num=20)
    trainindex = data.resampling(trainset[0], trainset[1], trainset[2], class_num=920)
    # data.words_frequency()
    bow = data.BoW()
    embeddings, num = data.embedding(mp)
    test_index = np.array(test_index)
    trainindex = np.array(trainindex)
    label = np.array(data.label)

    test_f_bow = bow[test_index]
    test_label = label[test_index]
    test_f_emb = embeddings[test_index]

    train_f_bow = bow[trainindex]
    train_label = label[trainindex]
    train_f_emb = embeddings[trainindex]
    
    print(data.classes)
    print(data.class_num)
    np.save(r'data\class_name.npy', data.classes)
    np.save(r'data\class_nums.npy', data.class_num)
    np.save(r'data\Bow-feature_test.npy', test_f_bow)
    np.save(r'data\Bow-feature_train.npy', train_f_bow)
    np.save(r'data\test_label.npy', np.array(test_label))
    np.save(r'data\train_label.npy', np.array(train_label))
    np.save(r'data\Embedding_feature_test.npy', test_f_emb)
    np.save(r'data\Embedding_feature_train.npy', train_f_emb)
    print('number of features can\'t be embedded is: {}'.format(num))

