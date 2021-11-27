from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import scipy.stats as st

class Histogram_Entropy:
    def __init__(self, size, dataset, trial=100) -> None:
        self.size = size
        self.trial = trial
        self.dataset = dataset

    def _calc_entropy(self, sub_meta):
        corpus = [ meta['utterance'] for meta in sub_meta ]
        vectorizer = CountVectorizer()
        pk = np.sum(vectorizer.fit_transform(corpus), axis=0)
        return st.entropy(pk)

    def get_subset_ind(self) -> float:
        index_set = list(range(len(self.dataset)))
        best_score = -float('inf')
        best_index_set = None
        for _ in range(self.trial):
            sub_index_set = np.random.sample(index_set, self.size)
            sub_meta = [ self.dataset.metadata[i] for i in sub_index_set ]
            score = self._calc_entropy(sub_meta)
            if score > best_score:
                best_score = score
                best_index_set = sub_index_set

        return best_index_set

class Submodular:
    def __init__(self, size, dataset) -> None:
        self.size = size
        self.dataset = dataset

    def _get_best_ind(self, S, S_bar) -> float:
        vectorizer = TfidfVectorizer()
        vectorizer.fit(S)
        corpus = [ self.dataset.metadata[i]['utterance'] for i in S_bar ]
        score_list = [ (i, np.sum(vectorizer.transform(corpus[i]))**2) for i in S_bar ]
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
    if mode == 'random':
        index = np.random.permutation(np.arrange(len(dataset)))[:size]

    elif mode == 'he':
        alg = Histogram_Entropy(size, dataset)
        index = alg.get_subset_ind()

    elif mode == 'sm':
        alg = Submodular(size, dataset)
        index = alg.get_subset_ind()
    
    dataset.select_subset(index)
    return dataset
