from typing import Literal
from imdb import Cinemagoer
import numpy as np
from spacy import Language, load

def get_movie_ids(movies: dict[str, int]) -> dict[str, str]:
    movies_by_id = {}
    ia = Cinemagoer()
    for movie, year in movies.items():
        search_results = ia.search_movie(movie)
        if year == -1:
            result = search_results[0]
        else:
            same_year = [x for x in search_results if int(x['year']) == int(year)]
            result = same_year[0]
        id = result.movieID
        movies_by_id[id] = movie
    return movies_by_id

def get_movie_plots(movie_ids: list[str]) -> dict[str, str]:
    plots_by_id = {}
    ia = Cinemagoer()
    for movie_id in movie_ids:
        for i in range(10):
            try:
                movie = ia.get_movie(movie_id)
                synopsis = movie['synopsis']
                plot = ""
                for s in synopsis:
                    plot += f"{s} "
            except:
                continue
            break
        else:
            print(f"Could not retrieve plot information for {movie_id}")
            continue
        plots_by_id[movie_id] = plot
    return plots_by_id

def get_plot_keywords(nlp: Language, movie_plot: str, adjectives: bool) -> set[str]:

    doc = nlp(movie_plot)
    word_classes = ["NOUN", "VERB"]
    if adjectives:
        word_classes.append("ADJ")

    keywords = set()
    for token in doc:
        if token.pos_ not in word_classes:
            continue
        lem = token.lemma_
        keywords.add(lem)

    return keywords

def get_plots_keywords(movie_plots: dict[str, str], adjectives: bool = False) -> dict[str, set[str]]:
    nlp = load("en_core_web_sm")
    
    keywords_by_id = {}
    for id, plot in movie_plots.items():
        keywords = get_plot_keywords(nlp, plot, adjectives)
        keywords_by_id[id] = keywords
    
    return keywords_by_id

def get_keyword_counts(movie_keywords: dict[str, str]) -> dict[str, int]:
    keyword_counts = {}
    for id, keywords in movie_keywords.items():
        for keyword in keywords:
            if keyword not in keyword_counts:
                keyword_counts[keyword] = 0
            keyword_counts[keyword] += 1
    return keyword_counts

def trim_keyword_list(keyword_counts: dict[str, int], min_occurrences: int, max_occurrences: int) -> list[str]:
    trimmed = []
    for keyword, count in keyword_counts.items():
        if count >= min_occurrences and count <= max_occurrences:
            trimmed.append(keyword)
    return trimmed

def get_movie_keyword_vector(master_keyword_list: list[str], movie_keywords: set[str], idx_by_keyword: dict[str, int]) -> np.ndarray:
    n = len(master_keyword_list)
    vec = np.zeros(n)

    for keyword in movie_keywords:
        if keyword in master_keyword_list:
            idx = idx_by_keyword[keyword]
            vec[idx] = 1
    
    return vec

def get_movie_keyword_vectors(master_keyword_list: list[str], movies_keywords: dict[str, set[str]]) -> dict[str, np.ndarray]:
    idx_by_keyword = {}
    for idx, keyword in enumerate(master_keyword_list):
        idx_by_keyword[keyword] = idx
    
    vectors_by_id = {}
    for id, keywords in movies_keywords.items():
        vec = get_movie_keyword_vector(master_keyword_list, keywords, idx_by_keyword)
        vectors_by_id[id] = vec

    return vectors_by_id

def get_training_features(movie_ids: list[str], method: Literal['random', 'genre', 'imdb_keyword'], dimensions: int = 0):
    pass

def train_keyword_to_feature_model(keyword_vectors: dict[str, np.ndarray], feature_vectors: dict[str, np.ndarray]):
    pass

def train_user_model():
    pass

