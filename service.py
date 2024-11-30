import pandas as pd
from sklearn.cluster import KMeans
import os
import pickle
import spacy
from accept import TokenAccepter
from encode import SparseVectorEncoding, MovieEncoding
import numpy as np
from fuzzywuzzy import fuzz
from inform import MovieInfo

# it is the chatbot's job to divine this information
class UserPreferences:
    def __init__(self):
        self.DescribedPlots: list[str] = None
        self.Directors: list[str] = None
        self.Actors: list[str] = None
        self.KnownTitles: list[str] = None
        self.KnownMovies: list[int] = None
        self.Genres: list[str] = None
        self.MoviesBefore: int = None
        self.MoviesAfter: int = None
        self.AllowAmericanOrigin: bool = None
        self.AllowBollywoordOrigin: bool = None
        self.AllowOtherOrigin: bool = None

class MovieRecommendation:
    def __init__(self, movie: MovieInfo):
        self.RecommendedMovie: MovieInfo = movie          # information for the actual movie being recommended 
        self.SimilarThemesToDescribed: list[str] = []     # list of user-provided descriptions that it has similar themes to 
        self.SimilarThemesToMovies: list[str] = []        # list of known liked movie titles that it has similar themes to
        self.SimilarGenresToMovies: list[str] = []        # list of known liked movie titles that it has similar genres to
        self.SimilarActorsToMovies: list[str] = []        # list of known liked movie titles that it has a similar cast to
        self.ExpressedLikeDirectors: list[str] = []       # list of director(s) for this movie that the user likes
        self.ExpressedLikeActors: list[str] = []          # list of actors in this movie that the user likes
        self.ExpressedLikeGenres: list[str] = []          # list of genres for this movie that the user likes
        self.WithinDesiredTimePeriod: bool = True         # does the movie fall within the desired time period
        self.HasDesiredOrigin: bool = True                # does the movie have the right origin
    # return a recommendation score based on the volume of criteria matching the user preferences
    def score(self):
        score = 0
        if self.SimilarThemesToDescribed:
            score += (1 + 0.1 * (len(self.SimilarThemesToDescribed) - 1))
        if self.SimilarThemesToMovies:
            score += (1 + 0.1 * (len(self.SimilarThemesToMovies) - 1))
        if self.SimilarGenresToMovies:
            score += (1 + 0.1 * (len(self.SimilarGenresToMovies) - 1))
        if self.SimilarActorsToMovies:
            score += (1 + 0.1 * (len(self.SimilarActorsToMovies) - 1))
        if self.ExpressedLikeDirectors:
            score += (1 + 0.1 * (len(self.ExpressedLikeDirectors) - 1))
        if self.ExpressedLikeActors:
            score += (1 + 0.1 * (len(self.ExpressedLikeActors) - 1))
        if self.ExpressedLikeGenres:
            score += (1 + 0.1 * (len(self.ExpressedLikeGenres) - 1))
        if self.WithinDesiredTimePeriod:
            score += 1
        if self.HasDesiredOrigin:
            score += 1
        return score

class MovieService:
    def __init__(self):
        self.MovieInfo: dict[int, MovieInfo] = {}
        self.MovieEncodings: dict[int, MovieEncoding] = {}
        self.ClusterModel: KMeans = None
        self.Recommendations: dict[int, MovieRecommendation] = {}
        self.load_setup_data()

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

        self.Recommendations = {id: MovieRecommendation(self.MovieInfo[id]) for id in ids}

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
                # use partial ratio for directors in case only the last name is specified (e.g. Tarantino)
                if fuzz.partial_ratio(director_name, director) > 90:
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
            for desired_member in cast_members:
                for member in cast:
                    if fuzz.ratio(desired_member, member) > 90:
                        matches += 1
            similarity = matches / ideal_matches
            if similarity > similarity_threshold:
                ids.append(id)
        
        return ids

    def query_movies_with_similar_genre(self, similar_to: str|int, from_ids: list[str], similarity_threshold: float):
        all_genres = {id: [ent.EntityName for ent in self.MovieEncodings[id].EntityEncodings if ent.EntityLabel.lower() == 'GENRE'] for id in from_ids}
        if similar_to in self.MovieEncodings: # id for a movie
            movie_genres = all_genres[similar_to]
        else:
            movie_genres = [member.strip() for member in similar_to.split(',')]
        ideal_matches = len(movie_genres)

        ids = []
        for id, genres in all_genres.items():
            matches = 0
            for desired_genre in movie_genres:
                for genre in genres:
                    if fuzz.ratio(desired_genre, genre) > 90:
                        matches += 1
            similarity = matches / ideal_matches
            if similarity > similarity_threshold:
                ids.append(id)
        
        return ids

    def query_movies_before(self, year: int, from_ids: list[str]):
        return [id for id in from_ids if int(self.MovieInfo[id].Year) < year]

    def query_movies_after_or_on(self, year: int, from_ids: list[str]):
        return [id for id in from_ids if int(self.MovieInfo[id].Year) >= year]
    
    def query_american_movies(self, from_ids: list[str]):
        return [id for id in from_ids if self.MovieInfo[id].Origin.lower() == 'american']
    
    def query_bollywood_movies(self, from_ids: list[str]):
        return [id for id in from_ids if self.MovieInfo[id].Origin.lower() == 'bollywood']

    def query_other_foreign_movies(self, from_ids: list[str]):
        return [id for id in from_ids if self.MovieInfo[id].Origin.lower() != 'bollywood' and self.MovieInfo[id].Origin.lower() != 'american']

    # works more like a filter, does not rank
    def query_from_user_preferences(self, user_preferences: UserPreferences):
        pass

    def recommend_movies_by_director(self, director: str):
        director_names = {id: [ent.EntityName for ent in self.MovieEncodings[id].EntityEncodings if ent.EntityLabel.lower() == 'director'] for id in self.Recommendations}
        for id in director_names:
            for director_name in director_names[id]:
                # use partial ratio for directors in case only the last name is specified (e.g. Tarantino)
                if fuzz.partial_ratio(director_name, director) > 90:
                    self.Recommendations[id].ExpressedLikeDirectors.append(director_name)

    def recommend_movies_by_actor(self, actor: str):
        actor_names = {id: [ent.EntityName for ent in self.MovieEncodings[id].EntityEncodings if ent.EntityLabel.lower() == 'cast'] for id in self.Recommendations}
        for id in actor_names:
            for actor_name in actor_names[id]:
                if fuzz.ratio(actor_name, actor) > 90:
                    self.Recommendations[id].ExpressedLikeActors.append(actor_name)

    def recommend_movies_by_genre(self, genre: str):
        genres = {id: [ent.EntityName for ent in self.MovieEncodings[id].EntityEncodings if ent.EntityLabel.lower() == 'genre'] for id in self.Recommendations}
        for id in genres:
            for genre_name in genres[id]:
                if fuzz.ratio(genre_name, genre) > 95:
                    self.Recommendations[id].ExpressedLikeGenres.append(genre_name)

    def recommend_movies_with_similar_plot_themes(self, similar_to: str|int, similarity_threshold: float, cosine: bool = True):
        described = False
        if similar_to in self.MovieEncodings: # it's an id
            similar_encoding = self.MovieEncodings[similar_to].PlotEncoding
        else: # it's a plot string to encode
            similar_encoding = self.encode_plot_theme_query(similar_to)
            described = True

        plot_encodings = {id: self.MovieEncodings[id].PlotEncoding for id in self.Recommendations}

        for id, encoding in plot_encodings.items():
            if cosine:
                similarity = encoding.normed_cosine_similarity(similar_encoding)
            else: # number of similar themes, regardless of intensity, over number of themes in the similar encoding
                these_themes = set(similar_encoding.Dimensions.keys())
                those_themes = set(encoding.Dimensions.keys())
                similar_themes = len(these_themes & those_themes)
                similarity = similar_themes / len(these_themes)
            if similarity >= similarity_threshold:
                if described:
                    self.Recommendations[id].SimilarThemesToDescribed.append(similar_to)
                else:
                    title = self.MovieInfo[id].Title
                    self.Recommendations[id].SimilarThemesToMovies.append(title)

    def recommend_movies_with_similar_cast(self, similar_to: str|int, similarity_threshold: float):
        all_cast_members = {id: [ent.EntityName for ent in self.MovieEncodings[id].EntityEncodings if ent.EntityLabel.lower() == 'CAST'] for id in self.Recommendations}
        # for now assume that it always is an id
        if similar_to in self.MovieEncodings: # id for a movie
            cast_members = all_cast_members[similar_to]
        #else:
        #    cast_members = [member.strip() for member in similar_to.split(',')]
        ideal_matches = len(cast_members)

        for id, cast in all_cast_members.items():
            matches = 0
            for desired_member in cast_members:
                for member in cast:
                    if fuzz.ratio(desired_member, member) > 90:
                        matches += 1
            similarity = matches / ideal_matches
            if similarity > similarity_threshold:
                title = self.MovieInfo[id].Title
                self.Recommendations[id].SimilarActorsToMovies.append(title)

    def recommend_movies_with_similar_genre(self, similar_to: str|int, similarity_threshold: float):
        all_genres = {id: [ent.EntityName for ent in self.MovieEncodings[id].EntityEncodings if ent.EntityLabel.lower() == 'GENRE'] for id in self.Recommendations}
        # for now assume that it is always an id
        if similar_to in self.MovieEncodings: # id for a movie
            movie_genres = all_genres[similar_to]
        #else:
        #    movie_genres = [member.strip() for member in similar_to.split(',')]
        ideal_matches = len(movie_genres)

        for id, genres in all_genres.items():
            matches = 0
            for desired_genre in movie_genres:
                for genre in genres:
                    if fuzz.ratio(desired_genre, genre) > 90:
                        matches += 1
            similarity = matches / ideal_matches
            if similarity > similarity_threshold:
                title = self.MovieInfo[id].Title
                self.Recommendations[id].SimilarGenresToMovies.append(title)

    def recommend_movies_before(self, year: int):
        for _, recommendation in self.Recommendations.items():
            if recommendation.RecommendedMovie.Year >= year:
                recommendation.WithinDesiredTimePeriod = False

    def recommend_movies_after_or_on(self, year: int):
        for _, recommendation in self.Recommendations.items():
            if recommendation.RecommendedMovie.Year < year:
                recommendation.WithinDesiredTimePeriod = False
    
    def recommend_american_movies(self):
        for _, recommendation in self.Recommendations.items():
            if recommendation.RecommendedMovie.Origin.lower() == 'american':
                recommendation.HasDesiredOrigin = True
    
    def recommend_bollywood_movies(self):
        for _, recommendation in self.Recommendations.items():
            if recommendation.RecommendedMovie.Origin.lower() == 'bollywood':
                recommendation.HasDesiredOrigin = True

    def recommend_other_foreign_movies(self):
        for _, recommendation in self.Recommendations.items():
            if recommendation.RecommendedMovie.Origin.lower() != 'american' and recommendation.RecommendedMovie.Origin.lower() != 'bollywood':
                recommendation.HasDesiredOrigin = True

    # does not filter, but will rank
    def recommend_from_user_preferences(self, user_preferences: UserPreferences, top_n: int = 10):
        pass