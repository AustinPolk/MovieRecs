from spacy import Language, load
from gensim import downloader
from sklearn.cluster import KMeans
import numpy as np

_VERBOSE_ = False

class Verboser:
    def __init__(self):
        _VERBOSE_ = True

    def Yes(self):
        _VERBOSE_ = True

    def No(self):
        _VERBOSE_ = False

class MovieEncoding:
    def __init__(self):
        self.Nouns: set[int] = set()
        self.Verbs: set[int] = set()
        self.Adjectives: set[int] = set()
        self.Adverbs: set[int] = set()
        self.Noun_Noun: set[tuple[int, int]] = set()
        self.Noun_Adjective: set[tuple[int, int]] = set()
        self.Noun_Verb: set[tuple[int, int]] = set()
        self.Verb_Adverb: set[tuple[int, int]] = set()

    def add_single(self, word_cluster: int, word_pos: str):
        if word_cluster == -1:
            return

        if word_pos == "NOUN":
            self.Nouns.add(word_cluster)
        elif word_pos == "VERB":
            self.Verbs.add(word_cluster)
        elif word_pos == "ADJ":
            self.Adjectives.add(word_cluster)
        elif word_pos == "ADV":
            self.Adverbs.add(word_cluster)
        else:
            pass

        if _VERBOSE_:
            print(f"{word_pos}-{word_cluster} added")

    def add_pair(self, word_cluster: int, word_pos: str, other_cluster: int, other_pos: str):
        if word_cluster == -1 or other_cluster == -1:
            return

        # apply a deliberate ordering to the pair
        pair = (word_cluster, other_cluster) if word_cluster < other_cluster else (other_cluster, word_cluster)
        
        if word_pos == "NOUN" and other_pos == "NOUN":
            self.Noun_Noun.add(pair)
        elif word_pos == "NOUN" and other_pos == "VERB":
            self.Noun_Verb.add(pair)
        elif word_pos == "VERB" and other_pos == "NOUN":
            self.Noun_Verb.add(pair)
        elif word_pos == "NOUN" and other_pos == "ADJ":
            self.Noun_Adjective.add(pair)
        elif word_pos == "ADJ" and other_pos == "NOUN":
            self.Noun_Adjective.add(pair)
        elif word_pos == "VERB" and other_pos == "ADV":
            self.Verb_Adverb.add(pair)
        elif word_pos == "ADV" and other_pos == "VERB":
            self.Verb_Adverb.add(pair)
        else:
            pass

        if _VERBOSE_:
            print(f"({word_pos}-{word_cluster}, {other_pos}-{other_cluster}) added")

    def self_describe(self):
        print(f"Nouns: {self.Nouns}")
        print(f"Verbs: {self.Verbs}")
        print(f"Adjectives: {self.Adjectives}")
        print(f"Adverbs: {self.Adverbs}")
        print(f"Noun-noun pairs: {self.Noun_Noun}")
        print(f"Noun-adjective pairs: {self.Noun_Adjective}")
        print(f"Noun-verb pairs: {self.Noun_Verb}")
        print(f"Verb-adverb pairs: {self.Verb_Adverb}")      

class EncodingSimilarity:
    def __init__(self):
        self.Nouns_Weight = 0.0
        self.Verbs_Weight = 0.0
        self.Adjectives_Weight = 0.0
        self.Adverbs_Weight = 0.0
        self.Noun_Noun_Weight = 0.0
        self.Noun_Adjective_Weight = 0.0
        self.Noun_Verb_Weight = 0.0
        self.Verb_Adverb_Weight = 0.0

    def set_weights(self, weights: dict[str, float]):
        self.Nouns_Weight = weights["Nouns"]
        self.Verbs_Weight = weights["Verbs"]
        self.Adjectives_Weight = weights["Adjectives"]
        self.Adverbs_Weight = weights["Adverbs"]
        self.Noun_Noun_Weight = weights["NounNoun"]
        self.Noun_Adjective_Weight = weights["NounAdjective"]
        self.Noun_Verb_Weight = weights["NounVerb"]
        self.Verb_Adverb_Weight = weights["VerbAdverb"]

    def _verbose_similarity_score(self, encoding: MovieEncoding, other_encoding: MovieEncoding):
        nouns_inter = encoding.Nouns & other_encoding.Nouns
        print(f"Similar nouns ({len(nouns_inter)}): {nouns_inter}")
        
        nouns_score = self.Nouns_Weight * len(nouns_inter)
        print(f"Noun score -> {nouns_score} = {self.Nouns_Weight} * {len(nouns_inter)}")

        verbs_inter = encoding.Verbs & other_encoding.Verbs
        print(f"Similar verbs ({len(verbs_inter)}): {verbs_inter}")

        verbs_score = self.Verbs_Weight * len(verbs_inter)
        print(f"Verb score -> {verbs_score} = {self.Verbs_Weight} * {len(verbs_inter)}")

        adjectives_inter = encoding.Adjectives & other_encoding.Adjectives
        print(f"Similar adjectives ({len(adjectives_inter)}): {adjectives_inter}")

        adjectives_score = self.Adjectives_Weight * len(adjectives_inter)
        print(f"Adjectives score -> {adjectives_score} = {self.Adjectives_Weight} * {len(adjectives_inter)}")

        adverbs_inter = encoding.Adverbs & other_encoding.Adverbs
        print(f"Similar verbs ({len(adverbs_inter)}): {adverbs_inter}")

        adverbs_score = self.Adverbs_Weight * len(adverbs_inter)
        print(f"Adverb score -> {adverbs_score} = {self.Adverbs_Weight} * {len(adverbs_inter)}")

        noun_noun_inter = encoding.Noun_Noun & other_encoding.Noun_Noun
        print(f"Similar noun-noun pairs ({len(noun_noun_inter)}): {noun_noun_inter}")

        noun_noun_score = self.Noun_Noun_Weight * len(noun_noun_inter)
        print(f"Noun-noun score -> {noun_noun_score} = {self.Noun_Noun_Weight} * {len(noun_noun_inter)}")

        noun_adjective_inter = encoding.Noun_Adjective & other_encoding.Noun_Adjective
        print(f"Similar noun-adjective pairs ({len(noun_adjective_inter)}): {noun_adjective_inter}")

        noun_adjective_score = self.Noun_Adjective_Weight * len(noun_adjective_inter)
        print(f"Noun-adjective score -> {noun_adjective_score} = {self.Noun_Adjective_Weight} * {len(noun_adjective_inter)}")

        noun_verb_inter = encoding.Noun_Verb & other_encoding.Noun_Verb
        print(f"Similar noun-verb pairs ({len(noun_verb_inter)}): {noun_verb_inter}")

        noun_verb_score = self.Noun_Verb_Weight * len(noun_verb_inter)
        print(f"Noun-verb score -> {noun_verb_score} = {self.Noun_Verb_Weight} * {len(noun_verb_inter)}")

        verb_adverb_inter = encoding.Verb_Adverb & other_encoding.Verb_Adverb
        print(f"Similar verb-adverb pairs ({len(verb_adverb_inter)}): {verb_adverb_inter}")

        verb_adverb_score = self.Verb_Adverb_Weight * len(verb_adverb_inter)
        print(f"Verb-adverb score -> {verb_adverb_score} = {self.Verb_Adverb_Weight} * {len(verb_adverb_inter)}")

        total_score = (nouns_score + 
                       verbs_score + 
                       adjectives_score + 
                       adverbs_score + 
                       noun_noun_score + 
                       noun_adjective_score + 
                       noun_verb_score + 
                       verb_adverb_score)

        print(f"\nTotal similarity score: {total_score}")

        return total_score

    def similarity_score(self, encoding: MovieEncoding, other_encoding: MovieEncoding):
        if _VERBOSE_:
            return self._verbose_similarity_score(encoding, other_encoding)
        
        nouns_score = self.Nouns_Weight * len(encoding.Nouns & other_encoding.Nouns)
        verbs_score = self.Verbs_Weight * len(encoding.Verbs & other_encoding.Verbs)
        adjectives_score = self.Adjectives_Weight * len(encoding.Adjectives & other_encoding.Adjectives)
        adverbs_score = self.Adverbs_Weight * len(encoding.Adverbs & other_encoding.Adverbs)
        noun_noun_score = self.Noun_Noun_Weight * len(encoding.Noun_Noun & other_encoding.Noun_Noun)
        noun_adjective_score = self.Noun_Adjective_Weight * len(encoding.Noun_Adjective & other_encoding.Noun_Adjective)
        noun_verb_score = self.Noun_Verb_Weight * len(encoding.Noun_Verb & other_encoding.Noun_Verb)
        verb_adverb_score = self.Verb_Adverb_Weight * len(encoding.Verb_Adverb & other_encoding.Verb_Adverb)

        total_score = (nouns_score + 
                       verbs_score + 
                       adjectives_score + 
                       adverbs_score + 
                       noun_noun_score + 
                       noun_adjective_score + 
                       noun_verb_score + 
                       verb_adverb_score)

        return total_score
    
class WordClusterer:
    def __init__(self):
        self.LanguageModel: Language = load("en_core_web_sm")
        self.WordClasses: list[str] = ["NOUN", "VERB", "ADJ", "ADV"]
        self.Word2Vec = downloader.load("word2vec-google-news-300")
        self.Clusterings: dict[str, KMeans] = {}
        self.ClusterCounts: dict[str, int] = {}
        self.ClusteringMemo: dict[(str, str), int] = {}

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

        index = 0
        for id, plot_str in plots.items():
            print(f"{index} - Getting keyword sets for {id}")
            these_keyword_sets = self.get_single_keyword_sets(plot_str)
            for word_class in keyword_sets:
                keyword_sets[word_class] |= these_keyword_sets[word_class]
            index += 1

        keyword_lists = {}
        for word_class, keyword_set in keyword_sets.items():
            keyword_lists[word_class] = list(keyword_set)

        return keyword_lists

    def train_clusterings(self, plots: dict[str, str], cluster_sizes: dict[str, int]):
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

    def assign_cluster(self, word: str, pos: str):
        if word not in self.Word2Vec:
            return -1
        if pos not in self.WordClasses:
            return -1
        
        p = (word, pos)
        if p not in self.ClusteringMemo:
            wv = self.Word2Vec[word]
            cluster = self.Clusterings[pos].predict(np.array([wv]))
            self.ClusteringMemo[p] = cluster[0]
        
        return self.ClusteringMemo[p]

class MovieEncoder:
    def __init__(self):
        self.Clusterer: WordClusterer
        self.LanguageModel: Language = load("en_core_web_sm")
        self.WordClasses: list[str] = ["NOUN", "VERB", "ADJ", "ADV"]
    
    def set_clusterer(self, clusterer: WordClusterer):
        self.Clusterer = clusterer

    def encode(self, plot_str: str, context_window: int):
        encoding = MovieEncoding()

        tokenized_plot = self.LanguageModel(plot_str)
        count = len(tokenized_plot)

        # some helpers to check if a pair of words has already been assessed
        hash = lambda x, y: x * count + y if y < x else y * count + x
        checked = set()
        checked_already = lambda h: (h in checked)

        for starting_index in range(count - context_window + 1):
            ending_index = starting_index + context_window
            window = tokenized_plot[starting_index:ending_index + 1]
            window_end = context_window - 1
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

                    if _VERBOSE_:
                        print(f"Considering the word pair {s.text}-{t.text}")

                    s_cluster = self.Clusterer.assign_cluster(s.lemma_, s.pos_)
                    if _VERBOSE_ and s_cluster != -1:
                        print(f"{s.lemma_}({s.text}) assigned to cluster {s.pos_}-{s_cluster}")
                    t_cluster = self.Clusterer.assign_cluster(t.lemma_, t.pos_)
                    if _VERBOSE_ and t_cluster != -1:
                        print(f"{t.lemma_}({t.text}) assigned to cluster {t.pos_}-{t_cluster}")

                    encoding.add_single(s_cluster, s.pos_)
                    encoding.add_single(t_cluster, t.pos_)
                    encoding.add_pair(s_cluster, s.pos_, t_cluster, t.pos_)

        return encoding

    def digest(self, plots: dict[str, str], context_window: int):
        encodings = {}
        for plot_id, plot_str in plots.items():
            print(f"Encoding movie with id {plot_id}")
            encodings[plot_id] = self.encode(plot_str, context_window)
        return encodings