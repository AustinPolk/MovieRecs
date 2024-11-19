import numpy as np
import pandas as pd
import spacy
import loader

class SparseVectorEncoding:
    def __init__(self, from_str: str|None = None):
        self.Values: dict[int, float] = {}  # actual values in the vector, indexed by dimension
        self.Norm: float = None             # magnitude of the vector, used in normalization/cosine similarity
        
        if from_str:
            for entry in from_str.split(','):
                idx, val = entry.split(':')
                self.Values[int(idx)] = float(val)
    def __getitem__(self, index: int):
        if index not in self.Values:    # any value not recorded is assumed 0 in the vector
            return 0
        return self.Values[index]
    def __setitem__(self, index: int, value: float):
        self.Values[index] = value
    def __str__(self):
        s_rep = ''
        for idx, val in self.Values.items():
            s_rep += f'{idx}:{val},'
        return s_rep[:-1] # remove trailing comma

class Movie:
    def __init__(self, from_str: str|None = None):
        self.Title: str = None                      # title of the movie
        self.Plot: str = None                       # plot of the movie
        self.Year: int = None                       # year that the movie was released
        self.Director: str = None                   # director of the movie, if known (otherwise None)
        self.Origin: str = None                     # "origin" of the movie (e.g. American, Tamil)
        self.Cast: list[str] = None                 # list of cast members in the movie, if known (otherwise None)
        self.Id: int = None                         # unique number to refer to this particular movie
        self.Tokens: list[(str, str)] = []          # list of tokens in the movie plot, where each entry is the token text and part of speech
        self.Entities: list[(str, str)] = []        # list of entities in the movie plot, where each entry is the entity text and entity label
        self.Encoding: SparseVectorEncoding = None  # vector encoding for the movie, using a sparse data structure to reduce space
        
        if from_str:
            strip = lambda s: s.strip("\"\'\n\r ")
            none_if_null = lambda x: None if x == ".NULL" else x
            components = [none_if_null(x) for x in from_str.strip("|").split("|")]
            self.Title = components[0]
            self.Plot = components[1]
            self.Year = int(components[2]) if components[2] else components[2]
            self.Director = components[3]
            self.Origin = components[4]
            self.Cast = [strip(x) for x in components[5].strip("[]").split(",")] if components[5] else components[5]
            self.Id = int(components[6]) if components[6] else components[6]

            if components[7]:
                token_list = components[7].split(",")
                for entry in token_list:
                    text, pos = [strip(x) for x in entry.split(":")]
                    self.Tokens.append((text, pos))
            
            if components[8]:
                entity_list = components[8].split(",")
                for entry in entity_list:
                    text, label = [strip(x) for x in entry.split(":")]
                    self.Entities.append((text, label))

            self.Encoding = SparseVectorEncoding(from_str=components[9]) if components[9] else components[9]
    def set(self, attr: str, value: str):
        # remove any quotes or whitespace from the beginning and end of the string
        strip = lambda s: s.strip("\"\'\n\r ")
        value = strip(value)
        
        # remove the pipe character if present, as it has a special purpose in the string representation of the movie
        value = value.replace("|", " ")
        
        # if the value is unknown, set it to none
        none_if_unknown = lambda s: None if not s or s.isspace() or s.lower() == 'unknown' else s
        value = none_if_unknown(value)

        if attr == 'Title':
            self.Title = value
        elif attr == 'Plot':
            self.Plot = value
        elif attr == 'Year':
            self.Year = int(value)
        elif attr == 'Director':
            self.Director = value
        elif attr == 'Origin':
            self.Origin = value
        elif attr == 'Cast':
            if not value:
                self.Cast = value
            self.Cast = [strip(member) for member in value.split(",")]
        elif attr == 'Id':
            self.Id = int(value)
        else:
            raise Exception()
    def __str__(self):
        token_list_str = ""
        for token, pos in self.Tokens:
            token_list_str += f"{token}:{pos},"
        token_list_str = token_list_str[:-1]

        ent_list_str = ""
        for entity, label in self.Entities:
            ent_list_str += f"{entity}:{label},"
        ent_list_str = ent_list_str[:-1]

        null_if_none = lambda x: ".NULL" if not x or str(x).isspace() else x
        s_rep = f"||{null_if_none(self.Title)}|{null_if_none(self.Plot)}|{null_if_none(self.Year)}|{null_if_none(self.Director)}|{null_if_none(self.Origin)}|{null_if_none(self.Cast)}|{null_if_none(self.Id)}|{null_if_none(token_list_str)}|{null_if_none(ent_list_str)}|{null_if_none(self.Encoding)}||"
        return s_rep
    
    def describe(self, short: bool):
        desc = f"{self.Title} ({self.Year})"
        if short:
            return desc    
        if self.Director:
            desc = f"{desc}, directed by {self.Director}"
        if self.Cast:
            desc = f"{desc}, starring "
            if len(self.Cast) == 1:
                desc += self.Cast[0]
            else:
                for name in self.Cast[:-1]:
                    desc += f"{name}, "
                desc += f"and {self.Cast[-1]}"
        return desc


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
                movie.set("Title", row['Title'])
                movie.set("Plot", row['Plot'])
                movie.set("Year", str(row['Release Year']))
                movie.set("Director", row['Director'])
                movie.set("Origin", row['Origin/Ethnicity'])
                movie.set("Cast", row['Cast'])
                movie.Id = index

                tokenized = language(movie.Plot)
                for token in tokenized:
                    movie.Tokens.append((token.lemma_, token.pos_))
                for ent in tokenized.ents:
                    movie.Entities.append((ent.text, ent.label_))

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