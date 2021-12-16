from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import scipy.stats as st
from copy import deepcopy

class Histogram_Entropy:
    def __init__(self, size, dataset) -> None:
        self.size = size
        self.dataset = dataset

    def _calc_entropy(self, S):
        corpus = [ self.dataset.metadata[i]['utterance'] for i in S ]
        vectorizer = CountVectorizer()
        pk = np.sum(vectorizer.fit_transform(corpus), axis=0)
        return st.entropy(pk)

    def _calc_entropy_diff(self, ind, S):
        S_ = deepcopy(S)
        S_add = S_.add(ind)
        return self._calc_entropy(S_add) - self._calc_entropy(S)

    def get_subset_ind(self, eps=0.01) -> float:
        index_set = set()
        remain_index_set = list(range(len(self.dataset)))
        I = len(self.dataset)
        for i in range(I):
            diff = self._calc_entropy_diff(i, index_set)
            if diff > eps:
                index_set.add(i)
                remain_index_set.remove(i)
            
            if len(index_set) == self.size:
                break
        
        m = self.size - len(index_set)
        if m > 0:
            index_set = index_set | set(remain_index_set[:m])

        return index_set

class Submodular:
    def __init__(self, size, dataset) -> None:
        self.size = size
        self.dataset = dataset

    def _get_best_ind(self, S, S_bar) -> float:
        vectorizer = TfidfVectorizer()
        vectorizer.fit(S)
        corpus = [ self.dataset.metadata[i]['utterance'] for i in S_bar ]
        score_list = [ (i, np.sqrt(np.sum(vectorizer.transform(corpus[i])))) for i in S_bar ]
        index, _ = max(score_list, key=lambda x: x[1])
        return index

    def get_subset_ind(self):
        index_set = set()
        remain_index_set = list(range(len(self.dataset)))
        for _ in range(self.size):
            index = self._get_best_ind(index_set, remain_index_set)
            index_set.add(index)
            remain_index_set.remove(index)


def select_subset(dataset, size=50, mode='random'):
    dataset.load_utterance()
    assert (mode == 'rand' or mode == 'he' or mode == 'sm')
    if mode == 'rand':
        index = np.random.permutation(np.arrange(len(dataset)))[:size]

    elif mode == 'he':
        alg = Histogram_Entropy(size, dataset)
        index = alg.get_subset_ind()

    elif mode == 'sm':
        alg = Submodular(size, dataset)
        index = alg.get_subset_ind()
    
    dataset.select_subset(index)
    return dataset
