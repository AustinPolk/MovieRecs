from spacy import Language, load
from gensim import downloader
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import loader
from fuzzywuzzy import fuzz
    
class WordClusterer:
    def __init__(self):
        self.LanguageModel: Language = load("en_core_web_sm")
        self.WordClasses: list[str] = ["NOUN", "VERB", "ADJ", "ADV"]
        self.Word2Vec = downloader.load("word2vec-google-news-300")
        self.Clustering: KMeans = None
        self.ClusteringMemo: dict[(str, str), int] = {}
        self.ClusterDescriptions: dict[int, list[str]] = {}

    def get_single_keyword_set(self, plot_str: str):
        keywords = set()
        tokens = self.LanguageModel(plot_str)
        for token in tokens:
            if token.pos_ in self.WordClasses and not token.is_stop:
                lem = token.lemma_
                keywords.add(lem)
        return keywords

    def get_many_keyword_lists(self, plots: dict[str, str]):
        keywords = set()
        index = 0
        for id, plot_str in plots.items():
            print(f"{index} - Getting keyword set for {id}")
            these_keywords = self.get_single_keyword_set(plot_str)
            keywords |= these_keywords
            index += 1
        return list(keywords)

    def train_clustering(self, n_clusters: int):
        if loader.exists("keywords"):
            keyword_list = loader.load("keywords")
        else:
            plots = loader.load_cached_plots()
            keyword_list = self.get_many_keyword_lists(plots)
            loader.save(keyword_list, "keywords")

        print(f"# of keywords found: {len(keyword_list)}")        

        all_vector_embeddings = []
        all_vector_keywords = []
        for word in keyword_list:
            if word in self.Word2Vec:
                all_vector_embeddings.append(self.Word2Vec[word])
                all_vector_keywords.append(word)
        all_vector_embeddings = np.array(all_vector_embeddings)

        print(f"# of keywords with embeddings: {len(all_vector_embeddings)}")   
        
        if loader.exists(f"clusterings-{n_clusters}"):
            self.Clustering = loader.load(f"clusterings-{n_clusters}")
        else:
            print(f"Clustering keywords into {n_clusters} clusters")
            self.Clustering = KMeans(n_clusters=n_clusters, random_state=26)
            self.Clustering.fit(all_vector_embeddings)

        print("Assembling the most descriptive words for each cluster")
        words_by_cluster = {}
        for word, cluster in zip(all_vector_keywords, self.Clustering.labels_):
            if cluster not in words_by_cluster:
                words_by_cluster[cluster] = []
            words_by_cluster[cluster].append(word)
        
        cluster_centers = self.Clustering.cluster_centers_
        self.ClusterDescriptions = {}
        for cluster in words_by_cluster:
            this_center = cluster_centers[cluster]
            these_words = words_by_cluster[cluster]
            sorted_words = sorted(these_words, key = lambda word: np.linalg.norm(this_center - self.Word2Vec[word], ord = 2))
            self.ClusterDescriptions[cluster] = sorted_words[:5]

        self.save()

        score = silhouette_score(all_vector_embeddings, self.Clustering.labels_)
        print(f"Silhouette score for N={n_clusters}: {score}")

        return score

    def assign_cluster(self, word: str, pos: str):
        if word not in self.Word2Vec:
            return -1
        if pos not in self.WordClasses:
            return -1
        
        if word not in self.ClusteringMemo:
            wv = self.Word2Vec[word]
            cluster = self.Clustering.predict(np.array([wv]))
            self.ClusteringMemo[word] = cluster[0]
        
        return self.ClusteringMemo[word]

    def save(self):
        loader.save(self.Clustering, f"clusterings-{len(self.Clustering.cluster_centers_)}")

class MovieEncoder2:
    def __init__(self, encoding_size: int):
        self.EncodingSize: int = encoding_size
        self.Clusterer: WordClusterer = WordClusterer()
        self.Clusterer.train_clustering(encoding_size)
        self.LanguageModel: Language = load("en_core_web_sm")

    def encode(self, to_encode: str, normed: bool):
        encoding = np.zeros(self.EncodingSize)
        tokenized = self.LanguageModel(to_encode)

        for token in tokenized:
            t_cluster = self.Clusterer.assign_cluster(token.lemma_, token.pos_)
            encoding[t_cluster] += 1

        return encoding / (np.linalg.norm(encoding) if normed else 1)

    def digest(self, normed: bool):
        if normed and loader.exists(f"digest-n-{self.EncodingSize}"):
            return loader.load(f"digest-n-{self.EncodingSize}")
        if not normed and loader.exists(f"digest-{self.EncodingSize}"):
            return loader.load(f"digest-{self.EncodingSize}")
        
        encodings = {}
        index = 1
        for plot_id, plot_str in loader.load_cached_plots().items():
            print(f"{index} - Encoding movie with id {plot_id}", sep="")
            index += 1
            encodings[plot_id] = self.encode(plot_str, normed)
            print("... Done")

        if normed:
            loader.save(encodings, f"digest-n-{self.EncodingSize}")
        if not normed:
            loader.save(encodings, f"digest-{self.EncodingSize}")
        
        return encodings
    
class MovieComparison:
    def __init__(self, movieBank: dict[str, np.ndarray]):
        self.MovieBank: dict[str, np.ndarray] = movieBank

    def cosine_similarity(self, A, B):
        return np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))

    def closest(self, encoding: np.ndarray, top_n: int = 5):
        scores: dict[str, float] = {id: self.cosine_similarity(encoding, self.MovieBank[id]) for id in self.MovieBank}
        ids = list(self.MovieBank.keys())
        sorted_ids = sorted(ids, key=lambda x: scores[x], reverse=True)
        return sorted_ids[:top_n]
    
# class UserModel:
#     def __init__(self, movie_bank: dict[str, MovieEncoding], similarity: EncodingSimilarity):
#         self.similarity: EncodingSimilarity = similarity
#         self.movie_bank: dict[str, MovieEncoding] = movie_bank
#         self.liked_encodings: dict[str, MovieEncoding] = {}
#         self.disliked_encodings: dict[str, MovieEncoding] = {}
#         self.movie_titles: dict[str, str] = loader.load_cached_titles()

#     def search_for(self, movie_title: str):
#         ids = list(self.movie_bank.keys())
#         fz = lambda x, y: fuzz.QRatio(x.lower(), y.lower())
#         sorted_ids = sorted(ids, key = lambda x: fz(movie_title, self.movie_titles[x]), reverse=True)
#         return sorted_ids

#     def title_of(self, movie_id: str):
#         return self.movie_titles[movie_id]

#     def likedness(self, encoding: MovieEncoding):
#         total = 0
#         for _, liked in self.liked_encodings.items():
#             total += self.similarity.similarity_score(encoding, liked)
#         for _, disliked in self.disliked_encodings.items():
#             total -= self.similarity.similarity_score(encoding, disliked)
    
#     def like(self, movie_id: str):
#         self.liked_encodings[movie_id] = self.movie_bank[movie_id]
        
#     def dislike(self, movie_id: str):
#         self.disliked_encodings[movie_id] = self.movie_bank[movie_id]

#     def recommend(self, how_many: int = 5):
#         scores = {id: self.likedness(self.movie_bank[id]) for id in self.movie_bank if (id not in self.liked_encodings and id not in self.disliked_encodings)}
#         ids = list(self.movie_bank.keys())
#         sorted_ids = sorted(ids, key=lambda x: scores[x], reverse=True)
#         return sorted_ids[:how_many]
    