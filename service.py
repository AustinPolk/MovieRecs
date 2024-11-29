import pandas as pd
from sklearn.cluster import KMeans
import os
import pickle
import spacy
from accept import TokenAccepter, EntityAccepter
from encode import SparseVectorEncoding, MovieEncoding
import numpy as np
from fuzzywuzzy import fuzz
from inform import MovieInfo

class MovieService:
    def __init__(self):
        self.MovieInfo: dict[int, MovieInfo] = {}
        self.MovieEncodings: dict[int, MovieEncoding] = {}
        self.ClusterModel: KMeans = None

    def load_setup_data(self):
        data_folder = "data"
        movie_info_bin = os.path.join(data_folder, "movie_info.bin")
        cluster_model_bin = os.path.join(data_folder, "cluster_model.bin")
        movie_encodings_bin = os.path.join(data_folder, "movie_encodings.bin")

        movieInfo = pd.read_pickle(movie_info_bin)
        ids = list(movieInfo['Id'].values)
        movieInfos = list(movieInfo['Movie'].values)
        self.MovieInfo = {id: movie_info for id, movie_info in zip(ids, movieInfos)}

        movieEncoding = pd.read_pickle(movie_encodings_bin)
        ids = list(movieEncoding['Id'].values)
        movieEncodings = list(movieEncoding['MovieEncoding'].values)
        self.MovieEncodings = {id: movie_encoding for id, movie_encoding in zip(ids, movieEncodings)}

        with open(cluster_model_bin, 'rb') as cluster_model_file:
            self.ClusterModel = pickle.load(cluster_model_file)

    def encode_plot_theme_query(self, plot_query: str):
        language = spacy.load("en_core_web_lg")
        tok_accepter = TokenAccepter()

        query_encoding = SparseVectorEncoding()

        tokenized = language(plot_query)
        for token in tokenized:
            if not tok_accepter.accept(token):
                continue
            dim = int(self.ClusterModel.predict(np.array([token.vector]))[0])
            query_encoding[dim] += 1
        query_encoding.normalize()

        return query_encoding
    
    def get_movie_info_by_id(self, id: int):
        return self.MovieInfo[id]

    def query_movies_by_title(self, title: str, top_n: int = 10):
        ids = list(self.MovieInfo.keys())
        similarities = {}
        for id, movieInfo in self.MovieInfo.items():
            title = movieInfo.Title
            similarities[id] = fuzz.partial_ratio(title, title)
        return sorted(ids, key = lambda x: similarities[x], reverse=True)[:top_n]

    def query_movies_by_director(self, director: str, from_ids: list[str]):
        director_names = {id: [ent.EntityName for ent in self.MovieEncodings[id].EntityEncodings if ent.EntityLabel.lower() == 'director'] for id in from_ids}
        ids = []
        for id in director_names:
            for director_name in director_names[id]:
                if fuzz.ratio(director_name, director) > 90:
                    ids.append(id)
                    break
        return ids

    def query_movies_by_actor(self, actor: str, from_ids: list[str]):
        actor_names = {id: [ent.EntityName for ent in self.MovieEncodings[id].EntityEncodings if ent.EntityLabel.lower() == 'cast'] for id in from_ids}
        ids = []
        for id in actor_names:
            for actor_name in actor_names[id]:
                if fuzz.ratio(actor_name, actor) > 90:
                    ids.append(id)
                    break
        return ids

    def query_movies_by_genre(self, genre: str, from_ids: list[str]):
        genres = {id: [ent.EntityName for ent in self.MovieEncodings[id].EntityEncodings if ent.EntityLabel.lower() == 'genre'] for id in from_ids}
        ids = []
        for id in genres:
            for genre_name in genres[id]:
                if fuzz.ratio(genre_name, genre) > 95:
                    ids.append(id)
                    break
        return ids

    def query_movies_with_similar_plot_themes(self, similar_to: str|int, from_ids: list[str], similarity_threshold: float, cosine: bool = True):
        if similar_to in self.MovieEncodings: # it's an id
            similar_encoding = self.MovieEncodings[similar_to].PlotEncoding
        else: # it's a plot string to encode
            similar_encoding = self.encode_plot_theme_query(similar_to)

        plot_encodings = {id: self.MovieEncodings[id].PlotEncoding for id in from_ids}

        similar_ids = []
        for id, encoding in plot_encodings.items():
            if cosine:
                similarity = encoding.normed_cosine_similarity(similar_encoding)
            else: # number of similar themes, regardless of intensity, over number of themes in the similar encoding
                these_themes = set(similar_encoding.Dimensions.keys())
                those_themes = set(encoding.Dimensions.keys())
                similar_themes = len(these_themes & those_themes)
                similarity = similar_themes / len(these_themes)
            if similarity >= similarity_threshold:
                similar_ids.append(id)
        
        return similar_ids

    def query_movies_with_similar_cast(self, similar_to: str|int, from_ids: list[str], similarity_threshold: float):
        all_cast_members = {id: [ent.EntityName for ent in self.MovieEncodings[id].EntityEncodings if ent.EntityLabel.lower() == 'CAST'] for id in from_ids}
        if similar_to in self.MovieEncodings: # id for a movie
            cast_members = all_cast_members[similar_to]
        else:
            cast_members = [member.strip() for member in similar_to.split(',')]
        ideal_matches = len(cast_members)

        ids = []
        for id, cast in all_cast_members.items():
            matches = 0
            # continue from here by counting how many desired cast members are here, sim score is that divided by ideal


    def query_movies_before(self, year: int, from_ids: list[str]):
        return [id for id in from_ids if int(self.MovieInfo[id].Year) < year]

    def query_movies_after_or_on(self, year: int, from_ids: list[str]):
        return [id for id in from_ids if int(self.MovieInfo[id].Year) >= year]
    
    def query_american_movies(self, from_ids: list[str]):
        return [id for id in from_ids if self.MovieInfo[id].Origin.lower() == 'american']
    
    def query_bollywood_movies(self, from_ids: list[str]):
        return [id for id in from_ids if self.MovieInfo[id].Origin.lower() == 'bollywood']