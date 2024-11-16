from spacy import Language, load
from gensim import downloader
from sklearn.cluster import KMeans
import numpy as np
import loader
from fuzzywuzzy import fuzz

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

        # TODO: maintain an order like noun before verb before adjective before adverb
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
    def __init__(self, weights: dict[str, float]):
        self.Nouns_Weight = weights["Nouns"]
        self.Verbs_Weight = weights["Verbs"]
        self.Adjectives_Weight = weights["Adjectives"]
        self.Adverbs_Weight = weights["Adverbs"]
        self.Noun_Noun_Weight = weights["NounNoun"]
        self.Noun_Adjective_Weight = weights["NounAdjective"]
        self.Noun_Verb_Weight = weights["NounVerb"]
        self.Verb_Adverb_Weight = weights["VerbAdverb"]
        self.alpha = weights["alpha"]

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

    def similarity_score(self, encoding: MovieEncoding, other_encoding: MovieEncoding, verbose: bool = False):
        if verbose:
            return self._verbose_similarity_score(encoding, other_encoding)
        
        match_less_miss = lambda x, y: len(x & y) - len(max(x | y - x, x | y - y)) * self.alpha

        nouns_score = self.Nouns_Weight * match_less_miss(encoding.Nouns, other_encoding.Nouns)
        verbs_score = self.Verbs_Weight * match_less_miss(encoding.Verbs, other_encoding.Verbs)
        adjectives_score = self.Adjectives_Weight * match_less_miss(encoding.Adjectives, other_encoding.Adjectives)
        adverbs_score = self.Adverbs_Weight * match_less_miss(encoding.Adverbs, other_encoding.Adverbs)
        noun_noun_score = self.Noun_Noun_Weight * match_less_miss(encoding.Noun_Noun, other_encoding.Noun_Noun)
        noun_adjective_score = self.Noun_Adjective_Weight * match_less_miss(encoding.Noun_Adjective, other_encoding.Noun_Adjective)
        noun_verb_score = self.Noun_Verb_Weight * match_less_miss(encoding.Noun_Verb, other_encoding.Noun_Verb)
        verb_adverb_score = self.Verb_Adverb_Weight * match_less_miss(encoding.Verb_Adverb, other_encoding.Verb_Adverb)

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

    def train_clusterings(self):
        plots = loader.load_cached_plots()
        keyword_lists = self.get_many_keyword_lists(plots)

        print("Keywords found:")        
        for word_class, keyword_list in keyword_lists.items():
            print(f"{word_class}: {len(keyword_list)}")

        all_vector_embeddings = {}
        for word_class in keyword_lists:
            all_vector_embeddings[word_class] = []
            for word in keyword_lists[word_class]:
                if word in self.Word2Vec:
                    all_vector_embeddings[word_class].append(self.Word2Vec[word])
            all_vector_embeddings[word_class] = np.array(all_vector_embeddings[word_class])

        print("Keywords with valid vector embeddings:")
        for word_class in all_vector_embeddings:
            print(f"{word_class}: {all_vector_embeddings[word_class].shape[0]}")

        print("Class cluster sizes:")
        for word_class in all_vector_embeddings:
            print(f"{word_class}: {all_vector_embeddings[word_class].shape[0] * 0.75}")

        for word_class in all_vector_embeddings:
            print(f"Clustering for {word_class} class")
            clustering = KMeans(n_clusters=int(all_vector_embeddings[word_class].shape[0] * 0.75), random_state=26)
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

    def save(self):
        loader.save(self.Clusterings, "clusterings")

    def load(self):
        self.Clusterings = loader.load("clusterings")

class MovieEncoder:
    def __init__(self):
        self.Clusterer: WordClusterer = WordClusterer()
        if loader.exists("clusterings"):
            self.Clusterer.load()
        else:
            self.Clusterer.train_clusterings()
            self.Clusterer.save()

        self.LanguageModel: Language = load("en_core_web_sm")
        self.WordClasses: list[str] = ["NOUN", "VERB", "ADJ", "ADV"]
        self.ContextWindow: int = 10

    def encode(self, to_encode: str, verbose: str = ''):
        encoding = MovieEncoding()

        tokenized_plot = self.LanguageModel(to_encode)
        count = len(tokenized_plot)

        # adjust
        context_window = self.ContextWindow + 1

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

                    if 'p' in verbose:
                        print(f"Considering the word pair {s.text}-{t.text}")

                    s_cluster = self.Clusterer.assign_cluster(s.lemma_, s.pos_)
                    if 'c' in verbose and s_cluster != -1:
                        print(f"{s.lemma_}({s.text}) assigned to cluster {s.pos_}-{s_cluster}")
                    t_cluster = self.Clusterer.assign_cluster(t.lemma_, t.pos_)
                    if 'c' in verbose and t_cluster != -1:
                        print(f"{t.lemma_}({t.text}) assigned to cluster {t.pos_}-{t_cluster}")

                    encoding.add_single(s_cluster, s.pos_)
                    encoding.add_single(t_cluster, t.pos_)
                    encoding.add_pair(s_cluster, s.pos_, t_cluster, t.pos_)

        return encoding

    def digest(self, context_window: int):
        self.ContextWindow = context_window
        plots = loader.load_cached_plots()

        if loader.exists(f"digest-{context_window}"):
            return loader.load(f"digest-{context_window}")

        encodings = {}
        index = 1
        for plot_id, plot_str in plots.items():
            print(f"{index} - Encoding movie with id {plot_id}")
            index += 1
            encodings[plot_id] = self.encode(plot_str)

        loader.save(encodings, f"digest-{context_window}")
        return encodings
    
class MovieComparison:
    def __init__(self, sim: EncodingSimilarity,movieBank: dict[str, MovieEncoding]):
        self.MovieBank: dict[str, MovieEncoding] = movieBank
        self.Similarity: EncodingSimilarity = sim

    def closest(self, encoding: MovieEncoding, top_n: int = 5):
        scores: dict[str, float] = {id: self.Similarity.similarity_score(encoding, self.MovieBank[id]) for id in self.MovieBank}
        ids = list(self.MovieBank.keys())
        sorted_ids = sorted(ids, key=lambda x: scores[x], reverse=True)
        return sorted_ids[:top_n]
    
class UserModel:
    def __init__(self, movie_bank: dict[str, MovieEncoding], similarity: EncodingSimilarity):
        self.similarity: EncodingSimilarity = similarity
        self.movie_bank: dict[str, MovieEncoding] = movie_bank
        self.liked_encodings: dict[str, MovieEncoding] = {}
        self.disliked_encodings: dict[str, MovieEncoding] = {}
        self.movie_titles: dict[str, str] = loader.load_cached_titles()

    def search_for(self, movie_title: str):
        ids = list(self.movie_bank.keys())
        fz = lambda x, y: fuzz.QRatio(x.lower(), y.lower())
        sorted_ids = sorted(ids, key = lambda x: fz(movie_title, self.movie_titles[x]), reverse=True)
        return sorted_ids

    def title_of(self, movie_id: str):
        return self.movie_titles[movie_id]

    def likedness(self, encoding: MovieEncoding):
        total = 0
        for _, liked in self.liked_encodings.items():
            total += self.similarity.similarity_score(encoding, liked)
        for _, disliked in self.disliked_encodings.items():
            total -= self.similarity.similarity_score(encoding, disliked)
    
    def like(self, movie_id: str):
        self.liked_encodings[movie_id] = self.movie_bank[movie_id]
        
    def dislike(self, movie_id: str):
        self.disliked_encodings[movie_id] = self.movie_bank[movie_id]

    def recommend(self, how_many: int = 5):
        scores = {id: self.likedness(self.movie_bank[id]) for id in self.movie_bank if (id not in self.liked_encodings and id not in self.disliked_encodings)}
        ids = list(self.movie_bank.keys())
        sorted_ids = sorted(ids, key=lambda x: scores[x], reverse=True)
        return sorted_ids[:how_many]
    