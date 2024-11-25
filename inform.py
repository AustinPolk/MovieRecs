import pandas as pd
from sklearn.metrics import silhouette_score
import numpy as np

class MovieInfo:
    def __init__(self):
        self.Title: str = None                      # title of the movie
        self.Plot: str = None                       # plot of the movie
        self.Year: int = None                       # year that the movie was released
        self.Director: list[str] = None             # director of the movie, if known (otherwise None)
        self.Origin: str = None                     # "origin" of the movie (e.g. American, Tamil)
        self.Cast: list[str] = None                 # list of cast members in the movie, if known (otherwise None)
        self.Id: int = None                         # unique number to refer to this particular movie
        self.Genre: list[str] = None                # genres attributed to the movie, if known (otherwise None)
    def set(self, attr: str, value: str):
        # replace an actual missing value with empty string
        if pd.isna(value):
            value = ""

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
            if not value:
                self.Director = value
            else:
                self.Director = []
                # remove quotes from nicknames
                dequote = lambda x: x.replace('\'', '').replace('\"', '')
                list_elements = [dequote(strip(director)) for director in value.split(',')]
                for element in list_elements:
                    # if this element looks like "and John Smith", make it "John Smith"
                    no_and = element[4:] if element.startswith('and ') else element
                    # split a list separated by an 'and' instead of a','
                    for sub_element in element.split(' and '):
                        self.Director.append(sub_element)
        elif attr == 'Origin':
            self.Origin = value
        elif attr == 'Cast':
            if not value:
                self.Cast = value
            else:
                self.Cast = []
                list_elements = [strip(member) for member in value.split(',')]
                for element in list_elements:
                    # handle the case when cast members are separated by newlines
                    for sub_element in element.split('\n'):
                        self.Cast.append(sub_element)
        elif attr == 'Genre':
            if not value:
                self.Genre = value
            else:
                self.Genre = []
                # split the genre into a list of genres
                list_elements = [strip(genre) for genre in value.split(',')]
                # try to get as granular of data as possible on the genre
                for element in list_elements:
                    self.Genre.append(element)
                    # separate genres like "romantic comedy" into "romantic" and "comedy"
                    for sub_element in element.split():
                        self.Genre.append(sub_element)
                        # separate genres like "drama/thriller" into "drama" and "thriller"
                        for sub_sub_element in sub_element.split('-/'):
                            self.Genre.append(sub_sub_element)
        elif attr == 'Id':
            self.Id = int(value)
        else:
            raise Exception() 
    def describe(self, short: bool):
        desc = f"{self.Title} ({self.Year})"
        if short:
            return desc
        if self.Origin:
            desc = f"{desc[:-1]}, {self.Origin})"
        if self.Director:
            desc = f"{desc}, directed by "
            if len(self.Director) == 1:
                desc += self.Director[0]
            else:
                for name in self.Director[:-1]:
                    desc += f"{name}, "
                desc += f"and {self.Director[-1]}"
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
        self.Vectors: dict[(str, str), np.ndarray] = {}
        self.Entities: list[(str, str)] = []