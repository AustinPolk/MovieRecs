import numpy as np
import pandas as pd
import spacy
import loader

class Movie:
    def __init__(self):
        self.Title: str = None
        self.Plot: str = None
        self.Year: int = None
        self.Cast: list[str] = None
        self.Id: int = None
        self.Tokens: list[(str, str)] = []
        self.Entities: list[str] = []
        self.VectorEncoding: np.ndarray = None

class MovieBank:
    def __init__(self):
        self.Movies: dict[int, Movie] = {}

    def __getitem__(self, index: int):
        return self.Movies[index]
    
    def __setitem__(self, index: int, value: Movie):
        self.Movies[index] = value

    def load_movies_from_csv(self, csv_filepath: str):
        language = spacy.load("en_core_web_lg")
        movie_frame = pd.read_csv(csv_filepath)

        index = 0
        for _, row in movie_frame.iterrows():
            movie = Movie()
            movie.Year = int(row['Release Year'])
            movie.Title = str(row['Title'])
            movie.Plot = str(row['Plot'])
            cast = str(row['Cast'])
            movie.Cast = [member.strip() for member in cast.split(",")]
            movie.Id = index

            tokenized = language(movie.Plot)
            for token in tokenized:
                movie.Tokens.append((token.lemma_, token.pos_))
            for ent in tokenized.ents:
                movie.Entities.append(ent.text)

            print(f"{movie.Id}. {movie.Title} ({movie.Year}) - {movie.Plot[:20]}... ({len(movie.Tokens)} tokens, {len(movie.Entities)} entities)")

            self[index] = movie
            index += 1

        self.save()

    def save(self):
        loader.save(self.Movies, "movie_bank")

    def load(self):
        self.Movies = loader.load("movie_bank")

