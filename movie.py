import numpy as np
import pandas as pd
import spacy
import loader

class SparseVectorEncoding:
    def __init__(self):
        self.Dimensions: int = 0            # dimensionality of the vector
        self.Values: dict[int, float] = {}  # actual values in the vector, indexed by dimension
        self.Norm: float = 1                # magnitude of the vector, used in normalization/cosine similarity
    def __getitem__(self, index: int):
        if index not in self.Values:    # any value not recorded is assumed 0 in the vector
            return 0
        return self.Values[index]
    def __setitem__(self, index: int, value: float):
        self.Values[index] = value

class Movie:
    def __init__(self):
        self.Title: str = None                      # title of the movie
        self.Plot: str = None                       # plot of the movie
        self.Year: int = None                       # year that the movie was released
        self.Director: str = None                   # director of the movie, if known (otherwise None)
        self.Origin: str = None                     # "origin" of the movie (e.g. American, Tamil)
        self.Cast: list[str] = None                 # list of cast members in the movie
        self.Id: int = None                         # unique number to refer to this particular movie
        self.Tokens: list[(str, str)] = []          # list of tokens in the movie plot, where each entry is the token text and part of speech
        self.Entities: list[(str, str)] = []        # list of entities in the movie plot, where each entry is the entity text and entity label
        self.Encoding: SparseVectorEncoding = None  # vector encoding for the movie, using a sparse data structure to reduce space

class MovieBank:
    def __init__(self):
        self.Movies: dict[int, Movie] = {}

    def __getitem__(self, index: int):
        return self.Movies[index]
    
    def __setitem__(self, index: int, value: Movie):
        self.Movies[index] = value

    def load_movies_from_csv(self, csv_filepath: str, pickup: bool = True):
        language = spacy.load("en_core_web_lg")
        movie_frame = pd.read_csv(csv_filepath)

        if pickup:
            loaded = self.load()
            if loaded:
                start = max(id for id in self.Movies)
            else:
                start = 0
        else:
            start = 0

        index = 0
        for _, row in movie_frame.iterrows():
            if index < start:
                index += 1
                continue

            try:
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
            except:
                continue
            index += 1

            # save progress every 200 rows
            if index % 200 == 0:
                self.save()

        self.save()

    def save(self):
        loader.save(self.Movies, "movie_bank")

    def load(self):
        if loader.exists("movie_bank"):
            self.Movies = loader.load("movie_bank")
            return True
        else:
            return False