import pickle
import pandas as pd
import spacy
from sklearn.cluster import KMeans

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
    def normalize(self):
        self.Norm = sum(A * A for A in self.Values.values()) ** 0.5
        for idx in self.Values:
            self.Values[idx] /= self.Norm
    def normed_cosine_similarity(self, other):
        return sum(A * B for A, B in zip(self.Values.values(), other.Values.values()))

class MovieInfo:
    def __init__(self):
        self.Title: str = None                      # title of the movie
        self.Plot: str = None                       # plot of the movie
        self.Year: int = None                       # year that the movie was released
        self.Director: str = None                   # director of the movie, if known (otherwise None)
        self.Origin: str = None                     # "origin" of the movie (e.g. American, Tamil)
        self.Cast: list[str] = None                 # list of cast members in the movie, if known (otherwise None)
        self.Id: int = None                         # unique number to refer to this particular movie
        self.Genre: str = None                      # genre attributed to the movie, if known (otherwise None)
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
            else:
                self.Cast = [strip(member) for member in value.split(",")]
        elif attr == 'Genre':
            self.Genre = value
        elif attr == 'Id':
            self.Id = int(value)
        else:
            raise Exception() 
    def describe(self, short: bool):
        desc = f"{self.Title} ({self.Year})"
        if short:
            return desc
        if self.Genre:
            desc = f"{desc[:-1]}, {self.Genre})"
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

class TokenizedPlot:
    def __init__(self):
        self.Tokens: list[(str, str)] = []
        self.Entities: list[(str, str)] = []

class TokenAccepter:
    def __init__(self):
        pass

    def accept(self, token):
        if token.pos_ not in ["NOUN", "VERB", "ADJ", "ADV"]:    # only accept words in an open class
            return False
        if not token.has_vector:    # only accept words with available vector embeddings
            return False
        return True
    
class EntityAccepter:
    def __init__(self):
        pass

    def accept(self, entity):
        if entity.label_ in ["TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]:    # do not accept these entity types, they aren't very useful
            return False
        return True

class MovieServiceSetup:
    def __init__(self):
        pass

    # source is the csv file containing movie info, 
    # sink is a binary file containing formatted movie info
    def load_all_movie_info(self, source: str, sink: str):
        index = 0
        source_frame = pd.read_csv(source)
        movie_infos = []
        for row in source_frame.iterrows():
            try:
                movie = MovieInfo()
                movie.set("Title", row['Title'])
                movie.set("Plot", row['Plot'])
                movie.set("Year", str(row['Release Year']))
                movie.set("Director", row['Director'])
                movie.set("Origin", row['Origin/Ethnicity'])
                movie.set("Cast", row['Cast'])
                movie.set("Genre", row['Genre'])
                movie.Id = index

                movie_infos.append((movie.Id, movie))
                print(movie.Id, movie.describe(False))
            except:
                continue
            index += 1
        sink_frame = pd.DataFrame(movie_infos, columns=['Id', 'Movie'])
        sink_frame.to_pickle(sink)

    # source is a binary file containing formatted movie info, 
    # sink is a binary file containing tokenized plots, 
    # vector_sink is a binary file of word vector embeddings
    def tokenize_all_plots_plus_vectors(self, source: str, sink: str, vector_sink: str):
        language = spacy.load("en_core_web_lg")
        source_frame = pd.read_pickle(source)
        tokenized_plots = []
        word_vectors = {}

        tok_accepter = TokenAccepter()
        ent_accepter = EntityAccepter()
        for row in source_frame.iterrows():
            try:
                movie = row['Movie']
                plot = movie.Plot
                tokenized_plot = TokenizedPlot()

                tokenized = language(plot)
                for token in tokenized:
                    if not tok_accepter.accept(token):
                        continue
                    if (token.lemma_, token.pos_) not in word_vectors:
                        word_vectors[(token.lemma_, token.pos_)] = (token.vector, token.vector_norm)
                    tokenized_plot.Tokens.append((token.lemma_, token.pos_))
                for entity in tokenized.ents:
                    if not ent_accepter.accept(entity):
                        continue
                    tokenized_plot.Entities.append((entity.text, entity.label_))
                
                # add special entities pertaining to the movie's production
                if movie.Director:
                    tokenized_plot.Entities.append((movie.Director, "DIRECTOR"))
                if movie.Cast:
                    for member in movie.Cast:
                        tokenized_plot.Entities.append((member, "CAST"))
                if movie.Genre:
                    tokenized_plot.Entities.append((movie.Genre, "GENRE"))
                if movie.Origin:
                    tokenized_plot.Entities.append((movie.Origin, "ORIGIN"))
                
                tokenized_plots.append((movie.Id, tokenized_plot))
            except:
                continue
        with open(vector_sink, "wb+") as vector_file:
            pickle.dump(word_vectors, vector_file)

        sink_frame = pd.DataFrame(tokenized_plots, columns=['Id', 'TokenizedPlot'])
        sink_frame.to_pickle(sink)

    # vector_source is a binary file containing word vector embeddings, 
    # sink is a binary file containing a clustering model
    def train_cluster_model_on_vectors(self, vector_source: str, sink: str, n_clusters: int):
        pass