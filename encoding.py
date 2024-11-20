from spacy import Language, load
from gensim import downloader
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import loader
from movie import Movie, SparseVectorEncoding

class WordClusterer:
    def __init__(self):
        self.WordClasses: list[str] = ["NOUN", "VERB", "ADJ", "ADV"]
        self.Word2Vec = downloader.load("word2vec-google-news-300")
        self.Clustering: KMeans = None
        self.ClusteringMemo: dict[(str, str), int] = {}

    def train_clustering(self, all_tokens: list[tuple[str, str]], n_clusters: int):
        print(f"Total tokens: {len(all_tokens)}")        
        distinct_tokens = set(all_tokens)
        print(f"Distinct tokens: {len(distinct_tokens)}")
        open_class_tokens = set([x.lower() for x, y in distinct_tokens if y in self.WordClasses])
        print(f"Total keywords (nouns, verbs, adjectives, and adverbs): {len(open_class_tokens)}")

        all_vector_embeddings = []
        for word in open_class_tokens:
            if word in self.Word2Vec:
                all_vector_embeddings.append(self.Word2Vec[word])
        all_vector_embeddings = np.array(all_vector_embeddings)

        print(f"Keywords with Word2Vec embeddings: {len(all_vector_embeddings)}")   

        print(f"Clustering keywords into {n_clusters} clusters")
        self.Clustering = KMeans(n_clusters=n_clusters, random_state=26)
        self.Clustering.fit(all_vector_embeddings)

        score = silhouette_score(all_vector_embeddings, self.Clustering.labels_)
        print(f"Silhouette score for {n_clusters} clusters: {score}")

        return score

    def assign_cluster(self, word: str, pos: str):
        word = word.lower()
        if word not in self.Word2Vec:
            return -1
        if pos not in self.WordClasses:
            return -1
        
        if word not in self.ClusteringMemo:
            wv = self.Word2Vec[word]
            cluster = self.Clustering.predict(np.array([wv]))
            self.ClusteringMemo[word] = cluster[0]
        
        return self.ClusteringMemo[word]

class Encoder:
    def __init__(self, encoding_size: int):
        self.EncodingSize: int = encoding_size
        self.Clusterer: WordClusterer = WordClusterer()
        self.LanguageModel: Language = load("en_core_web_lg")

    def encode_query(self, query: str):
        encoding = SparseVectorEncoding()
        tokenized = self.LanguageModel(query)

        for token in tokenized:
            t_cluster = self.Clusterer.assign_cluster(token.lemma_, token.pos_)
            if t_cluster == -1:
                continue
            encoding[t_cluster] += 1
                
        encoding.normalize()
        return encoding

    def encode(self, movie_to_encode: Movie, normed: bool):
        encoding = SparseVectorEncoding()

        for token, pos in movie_to_encode.Tokens:
            t_cluster = self.Clusterer.assign_cluster(token, pos)
            if t_cluster == -1:
                continue
            encoding[t_cluster] += 1

        encoding.normalize()
        return encoding