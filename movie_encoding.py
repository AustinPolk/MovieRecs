from spacy import Language, load
from gensim import downloader
from sklearn.cluster import KMeans
import numpy as np

class MovieEncoder:
        
    def __init__(self):
        self.LanguageModel: Language = load("en_core_web_sm")
        self.WordClasses: list[str] = ["NOUN", "VERB", "ADJ", "ADV"]
        self.Word2Vec = downloader.load("word2vec-google-news-300")
        self.Clusterings: dict[str, KMeans]
        self.ClusterSizes: dict[str, int]
        pass

    def get_single_keyword_sets(self, plot_str: str):
        keyword_sets = {}
        for word_class in self.WordClasses:
            keyword_sets[word_class] = set()

        tokens = self.LanguageModel(plot_str)
        for token in tokens:
            if token.pos_ in self.WordClasses:
                lem = token.lemma_
                keyword_sets[token.pos_].add(lem)

        return keyword_sets

    def get_many_keyword_lists(self, plots: dict[str, str]):
        keyword_sets = {}
        for word_class in self.WordClasses:
            keyword_sets[word_class] = set()

        for _, plot_str in plots.items():
            these_keyword_sets = self.get_single_keyword_sets(plot_str)
            for word_class in keyword_sets:
                keyword_sets[word_class] |= these_keyword_sets[word_class]

        keyword_lists = {}
        for word_class, keyword_set in keyword_sets.items():
            keyword_lists[word_class] = list(keyword_set)

        return keyword_lists

    def train_clusterings(self, keyword_lists: dict, cluster_sizes: dict):
        all_vector_embeddings = {}
        self.ClusterSizes = cluster_sizes

        for word_class in keyword_lists:
            all_vector_embeddings[word_class] = []
            for word in keyword_lists[word_class]:
                if word in self.Word2Vec:
                    all_vector_embeddings[word_class].append(self.Word2Vec[word])
            all_vector_embeddings[word_class] = np.array(all_vector_embeddings[word_class])

        for word_class in all_vector_embeddings:
            clustering = KMeans(n_clusters=cluster_sizes[word_class], random_state=26)
            clustering.fit(all_vector_embeddings[word_class])
            self.Clusterings[word_class] = clustering

    def cluster_encode_word(self, word: str, pos: str):
        clustering = self.Clusterings[pos]
        vector = self.Word2Vec[word]
        which_cluster = clustering.predict(np.array([vector]))
        return which_cluster[0]
