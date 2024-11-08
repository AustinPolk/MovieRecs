from spacy import Language, load
from gensim import downloader
from sklearn.cluster import KMeans
import numpy as np
from autoencoder import Autoencoder

class Trainer:
        
    def __init__(self):
        self.LanguageModel: Language = load("en_core_web_sm")
        self.WordClasses: list[str] = ["NOUN", "VERB", "ADJ", "ADV"]
        self.Word2Vec = downloader.load("word2vec-google-news-300")
        self.Clusterings: dict[str, KMeans] = {}
        self.ClusterCounts: dict[str, int] = {}
        self.ContextWindowSize: int = 5
        self.AutoEncoder: Autoencoder = None
        self.TrainedAutoEncoder: bool = False
        self.ClusterEncodeCache: dict[str, dict[str, int]] = {}

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

        for id, plot_str in plots.items():
            print(f"Getting keyword sets for {id}")
            these_keyword_sets = self.get_single_keyword_sets(plot_str)
            for word_class in keyword_sets:
                keyword_sets[word_class] |= these_keyword_sets[word_class]

        keyword_lists = {}
        for word_class, keyword_set in keyword_sets.items():
            keyword_lists[word_class] = list(keyword_set)

        return keyword_lists

    def train_clusterings(self, plots: dict[str, str], cluster_sizes: dict):
        
        keyword_lists = self.get_many_keyword_lists(plots)
        
        all_vector_embeddings = {}
        self.ClusterCounts = cluster_sizes

        for word_class in keyword_lists:
            all_vector_embeddings[word_class] = []
            for word in keyword_lists[word_class]:
                if word in self.Word2Vec:
                    all_vector_embeddings[word_class].append(self.Word2Vec[word])
            all_vector_embeddings[word_class] = np.array(all_vector_embeddings[word_class])

        for word_class in all_vector_embeddings:
            print(f"Clustering for {word_class} class")
            clustering = KMeans(n_clusters=cluster_sizes[word_class], random_state=26)
            clustering.fit(all_vector_embeddings[word_class])
            self.Clusterings[word_class] = clustering

    def cluster_encode_word(self, word: str, pos: str):
        if word not in self.Word2Vec:
            return -1

        if pos not in self.ClusterEncodeCache:
            self.ClusterEncodeCache[pos] = {}

        if word not in self.ClusterEncodeCache[pos]:
            clustering = self.Clusterings[pos]
            vector = self.Word2Vec[word]
            which_cluster = clustering.predict(np.array([vector]))
            self.ClusterEncodeCache[pos][word] = which_cluster[0]

        return self.ClusterEncodeCache[pos][word]

    # use the context window method to assemble matrices, then flatten and concatenate them in to a single vector
    def assemble_plot_encoding(self, plot_str: str):
        noun = np.zeros((self.ClusterCounts["NOUN"]))
        verb = np.zeros((self.ClusterCounts["VERB"]))
        adj = np.zeros((self.ClusterCounts["ADJ"]))
        adv = np.zeros((self.ClusterCounts["ADV"]))
        noun_noun = np.zeros((self.ClusterCounts["NOUN"], self.ClusterCounts["NOUN"]))
        noun_verb = np.zeros((self.ClusterCounts["NOUN"], self.ClusterCounts["VERB"]))
        noun_adj = np.zeros((self.ClusterCounts["NOUN"], self.ClusterCounts["ADJ"]))
        verb_adv = np.zeros((self.ClusterCounts["VERB"], self.ClusterCounts["ADV"]))

        tokenized_plot = self.LanguageModel(plot_str)
        count = len(tokenized_plot)

        # some helpers to check if a pair of words has already been assessed
        hash = lambda x, y: x * count + y
        checked = set()
        checked_already = lambda h: (h in checked)

        for starting_index in range(count - self.ContextWindowSize + 1):
            ending_index = starting_index + self.ContextWindowSize
            window = tokenized_plot[starting_index:ending_index + 1]
            window_end = self.ContextWindowSize - 1
            for window_s_index in range(window_end):
                for window_t_index in range(window_s_index + 1, window_end):
                    
                    s_index = starting_index + window_s_index
                    t_index = starting_index + window_t_index

                    # check if this pair has already been assessed, skip if so
                    pair_hash = hash(s_index, t_index)
                    if checked_already(pair_hash):
                        continue
                    else:
                        checked.add(pair_hash)

                    # s and t represent the two halves of each word pair in the window
                    s = window[window_s_index]
                    t = window[window_t_index]
                    
                    # if neither word is in a class we care about, move on
                    if s.pos_ not in self.WordClasses and t.pos_ not in self.WordClasses:
                        continue
                    
                    # if s is in an open word class, set it in the appropriate vector
                    if s.pos_ in self.WordClasses:
                        s_encode = self.cluster_encode_word(s.lemma_, s.pos_)
                        if s_encode != -1:
                            if s.pos_ == "NOUN":
                                noun[s_encode] = 1
                            elif s.pos_ == "VERB":
                                verb[s_encode] = 1
                            elif s.pos_ == "ADJ":
                                adj[s_encode]
                            elif s.pos_ == "ADV":
                                adv[s_encode]

                    # if t is in an open word class, set it in the appropriate vector
                    if t.pos_ in self.WordClasses:
                        t_encode = self.cluster_encode_word(t.lemma_, t.pos_)
                        if t_encode != -1:
                            if t.pos_ == "NOUN":
                                noun[t_encode] = 1
                            elif t.pos_ == "VERB":
                                verb[t_encode] = 1
                            elif t.pos_ == "ADJ":
                                adj[t_encode] = 1
                            elif t.pos_ == "ADV":
                                adv[t_encode] = 1

                    # unless both are in open word classes, loop
                    if s.pos_ not in self.WordClasses or t.pos_ not in self.WordClasses:
                        continue

                    # if one of them didn't map to a cluster, loop
                    if s_encode == -1 or t_encode == -1:
                        continue

                    # set the appropriate cell in the appropriate matrix
                    if s.pos_ == "NOUN" and t.pos_ == "NOUN":
                        noun_noun[s_encode, t_encode] = 1
                        noun_noun[t_encode, s_encode] = 1

                    elif s.pos_ == "NOUN" and t.pos_ == "VERB":
                        noun_verb[s_encode, t_encode] = 1

                    elif s.pos_ == "VERB" and t.pos_ == "NOUN":
                        noun_verb[t_encode, s_encode] = 1

                    elif s.pos_ == "NOUN" and t.pos_ == "ADJ":
                        noun_adj[s_encode, t_encode] = 1

                    elif s.pos_ == "ADJ" and t.pos_ == "NOUN":
                        noun_adj[t_encode, s_encode] = 1

                    elif s.pos_ == "VERB" and t.pos_ == "ADV":
                        verb_adv[s_encode, t_encode] = 1

                    elif s.pos_ == "ADV" and t.pos_ == "VERB":
                        verb_adv[t_encode, s_encode] = 1
                    
                    else:
                        continue
                    
        nn_vec = noun_noun.flatten()
        nv_vec = noun_verb.flatten()
        na_vec = noun_adj.flatten()
        va_vec = verb_adv.flatten()

        plot_vec = np.concatenate([noun, verb, adj, adv, nn_vec, nv_vec, na_vec, va_vec])

        return plot_vec

    def assemble_plot_encodings(self, plots: dict[str, str], window_size: int = 5):
        plot_encodings = {}
        self.ContextWindowSize = window_size

        for id, plot_str in plots.items():
            print(f"Assembling plot encoding for {id}")
            encoding = self.assemble_plot_encoding(plot_str)
            plot_encodings[id] = encoding

        return plot_encodings
                    
    def train_autoencoder(self, plots: dict[str, str], contextWindow: int, layer_sizes: list[int], layer_activations: list[str]):
        plot_encodings = self.assemble_plot_encodings(plots, contextWindow)

        input_features = []
        for _, vec in plot_encodings.items():
            input_features.append(vec)
        input_features = np.array(input_features)            

        print("Training autoencoder")
        self.AutoEncoder = Autoencoder(input_features[0].shape[0], layer_sizes, layer_activations)

        self.AutoEncoder.fit(input_features, input_features, epochs=20, shuffle=True)

        self.TrainedAutoEncoder = True

    def plot_autoencoding(self, plot_str: str, no_auto: bool = False):
        plot_encoding = self.assemble_plot_encoding(plot_str)
        if no_auto:
            return plot_encoding

        input_features = [plot_encoding]
        input_features = np.array(input_features)

        encoded = self.AutoEncoder.encoder(input_features).numpy()

        return encoded[0]