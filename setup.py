import pickle
import pandas as pd
import spacy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import os

from inform import MovieInfo, TokenizedPlot
from accept import TokenAccepter, EntityAccepter
from encode import MovieEncoding

class MovieServiceSetup:
    def __init__(self):
        self.setup()

    # source is the csv file containing movie info, 
    # sink is a binary file containing formatted movie info
    def load_all_movie_info(self, source: str, sink: str):
        # this step has already been completed
        if os.path.exists(sink):
            print("Movies already loaded")
            return

        source_frame = pd.read_csv(source)
        movie_infos = []
        for idx, row in source_frame.iterrows():
            try:
                movie = MovieInfo()
                movie.set("Title", row['Title'])
                movie.set("Plot", row['Plot'])
                movie.set("Year", str(row['Release Year']))
                movie.set("Director", row['Director'])
                movie.set("Origin", row['Origin/Ethnicity'])
                movie.set("Cast", row['Cast'])
                movie.set("Genre", row['Genre'])
                movie.Id = idx

                movie_infos.append((movie.Id, movie))
                print(movie.Id, movie.describe(False))
            except Exception as e:
                print('Fuck, ', e)
                continue
        sink_frame = pd.DataFrame(movie_infos, columns=['Id', 'Movie'])
        sink_frame.to_pickle(sink)

    # source is a binary file containing formatted movie info, 
    # sink is a binary file containing tokenized plots, 
    # vector_sink is a binary file of word vector embeddings
    def tokenize_all_plots_plus_vectors(self, source: str, sink: str, start_idx: int, end_idx: int, blacklist: list[int]):
        # this step has already been completed
        if os.path.exists(sink):
            print(f"Plots {start_idx} to {end_idx-1} alraedy tokenized")
            return

        language = spacy.load("en_core_web_lg")
        source_frame = pd.read_pickle(source)
        tokenized_plots = []

        tok_accepter = TokenAccepter()
        ent_accepter = EntityAccepter()
        for idx, row in source_frame.iterrows():
            # note that if end_idx < start_idx, it will go from start_idx all the way through (this is intended)
            if idx < start_idx or idx in blacklist:
                continue
            elif idx == end_idx:
                break
            
            try:
                movie = row['Movie']
                plot = movie.Plot
                tokenized_plot = TokenizedPlot()

                tokenized = language(plot)
                for token in tokenized:
                    if not tok_accepter.accept(token):
                        continue
                    if (token.lemma_, token.pos_) not in tokenized_plot.Vectors:
                        tokenized_plot.Vectors[(token.lemma_, token.pos_)] = token.vector
                    tokenized_plot.Tokens.append((token.lemma_, token.pos_))
                for entity in tokenized.ents:
                    if not ent_accepter.accept(entity):
                        continue
                    tokenized_plot.Entities.append((entity.text, entity.label_))
                
                # add special entities pertaining to the movie's production
                if movie.Director:
                    for director in movie.Director:
                        tokenized_plot.Entities.append((director, "DIRECTOR"))
                if movie.Cast:
                    for member in movie.Cast:
                        tokenized_plot.Entities.append((member, "CAST"))
                if movie.Genre:
                    for genre in movie.Genre:
                        tokenized_plot.Entities.append((genre, "GENRE"))
                if movie.Origin:
                    tokenized_plot.Entities.append((movie.Origin, "ORIGIN"))
                
                print(movie.Id, movie.describe(False), f"({len(tokenized_plot.Tokens)}, {len(tokenized_plot.Entities)})")
                tokenized_plots.append((movie.Id, tokenized_plot))
            except Exception as e:
                print('Fuck, ', e)
                continue

        sink_frame = pd.DataFrame(tokenized_plots, columns=['Id', 'TokenizedPlot'])
        sink_frame.to_pickle(sink)

    # sources is a list of binary files containing tokenized plots
    # vector sources is a list of binary files containing word vector embeddings
    # sink is the combined file containing all tokenized plots
    # vector_sink is the combined file containing all word vector embeddings
    def combine_tokenized_results(self, sources: list[str], sink: str):
        if not os.path.exists(sink):
            print("Combining tokenized plots")
            all_tokenized = pd.read_pickle(sources[0])
            for source in sources[1:]:
                this_tokenized = pd.read_pickle(source)
                all_tokenized = pd.concat([all_tokenized, this_tokenized], ignore_index=True)
            all_tokenized.to_pickle(sink)
        else:
            print("Tokenized plots already combined")

    # source is a binary file containing word vector embeddings, 
    # sink is a binary file containing a clustering model,
    # score_sink is a binary file containing scores for different cluster sizes
    def train_cluster_model_on_vectors(self, source: str, sink: str, score_sink:str, min_clusters: int, max_clusters: int, cluster_tries: int = 30):
        if os.path.exists(sink) and os.path.exists(score_sink):
            print("Clustering model already chosen")
            return
        
        # get all distinct word vectors
        all_tokenized = pd.read_pickle(source)
        occurrences = {}
        word_vectors = {}
        for _, row in all_tokenized.iterrows():
            tokenized_plot = row['TokenizedPlot']
            word_vectors.update(tokenized_plot.Vectors)
            # count how many times each distinct token appears
            for token_pos in tokenized_plot.Vectors:
                if token_pos not in occurrences:
                    occurrences[token_pos] = 0
                occurrences[token_pos] += 1

        # only include a word vector if it occurs in at least 200 movie plots,
        # attempt to remove strange, outlier words that end up becoming their own cluster
        trimmed_vectors = {k:i for k, i in word_vectors.items() if occurrences[k] > 100}
        word_vectors = np.array(list(trimmed_vectors.values()))

        print(f"Performing clustering on {len(word_vectors)} word vectors (down from {len(occurrences)})")
        
        best_cluster_model = None
        best_score = -1
        all_scores = {}
        cluster_statistics = {}

        max_clusters = min(max_clusters, len(word_vectors) - 1)
        cluster_step = (max_clusters - min_clusters) // (cluster_tries - 1)

        for n_clusters in range(min_clusters, max_clusters + 1, cluster_step):
            cluster_model = KMeans(n_clusters=n_clusters, random_state=26)
            cluster_model.fit(word_vectors)
            score = silhouette_score(word_vectors, cluster_model.labels_)
            print(f"Silhouette score for {n_clusters} clusters: {score}")

            cluster_sizes = np.bincount(cluster_model.labels_)
            min_size = min(cluster_sizes)
            max_size = max(cluster_sizes)
            avg_size = np.average(cluster_sizes)
            cluster_statistics[n_clusters] = {
                'Minimum': min_size,
                'Maximum': max_size,
                'Average': avg_size
            }
            print(f"Max: {max_size}\tMin: {min_size}\tAvg: {avg_size}")

            if score > best_score:
                best_score = score
                best_cluster_model = cluster_model

            all_scores[n_clusters] = score

        print(f"{len(best_cluster_model.cluster_centers_)} clusters achieved the highest silhouette score ({best_score})")

        with open(sink, "wb+") as cluster_file:
            pickle.dump(best_cluster_model, cluster_file)
        
        scores = {
            'Scores': all_scores,
            'Statistics': cluster_statistics
        }
        with open(score_sink, "wb+") as score_file:
            pickle.dump(scores, score_file)

    # source is a binary file containing tokenized plots
    # cluster_source is a binary file containing a clustering model
    # sink is a binary file containing the final encodings
    def encode_all_movies(self, source: str, cluster_source: str, sink: str):
        if os.path.exists(sink):
            print("All movies already encoded")
            return
        
        source_frame = pd.read_pickle(source)

        with open(cluster_source, "rb") as cluster_file:
            cluster_model = pickle.load(cluster_file)

        print("Encoding all movies")

        all_encodings = []
        clustering_memo = {}
        for _, row in source_frame.iterrows():
            try:
                id = row['Id']
                tokenized_plot = row['TokenizedPlot']

                movie_encoding = MovieEncoding()

                for token, pos in tokenized_plot.Tokens:
                    # to speed up computation, memoize clustering results
                    if (token, pos) not in clustering_memo:
                        word_vector = tokenized_plot.Vectors[(token, pos)]
                        dim = int(cluster_model.predict(np.array([word_vector]))[0])
                        clustering_memo[(token, pos)] = dim
                    else:
                        dim = clustering_memo[(token, pos)]
                    movie_encoding.PlotEncoding[dim] += 1
                movie_encoding.PlotEncoding.normalize()

                for entity, label in tokenized_plot.Entities:
                    movie_encoding.add_entity(entity, label)

                print(f"Encoded id={id}")

                all_encodings.append((id, movie_encoding))
            except Exception as e:
                print('Fuck, ', e)
                continue
        
        sink_frame = pd.DataFrame(all_encodings, columns=['Id', 'MovieEncoding'])
        sink_frame.to_pickle(sink)

    def setup(self):
        import os

        data_folder = "data"
        movie_data_csv = os.path.join(data_folder, "wiki_movie_info.csv")
        movie_info_bin = os.path.join(data_folder, "movie_info.bin")
        tokenized_plots_bin = os.path.join(data_folder, "tokenized_plots.bin")
        cluster_model_bin = os.path.join(data_folder, "cluster_model.bin")
        cluster_scores_bin = os.path.join(data_folder, "cluster_scores.bin")
        movie_encodings_bin = os.path.join(data_folder, "movie_encodings.bin")

        self.load_all_movie_info(source=movie_data_csv, sink=movie_info_bin)
        movie_blacklist = []
        
        tokenized_sinks = []
        rough_count = 35000
        step = 100
        for start, end in zip(range(0, rough_count+1, step), range(step, rough_count+step+1, step)):
            this_tokenized_sink = os.path.join(data_folder, "tokenized_plots", f"{start}_{end-1}.bin")
            tokenized_sinks.append(this_tokenized_sink)
            self.tokenize_all_plots_plus_vectors(source=movie_info_bin, sink=this_tokenized_sink, start_idx=start, end_idx=end, blacklist=movie_blacklist)
        self.combine_tokenized_results(sources=tokenized_sinks, sink=tokenized_plots_bin)

        self.train_cluster_model_on_vectors(source=tokenized_plots_bin, sink=cluster_model_bin, score_sink=cluster_scores_bin, min_clusters=500, max_clusters=10000, cluster_tries=30)
        self.encode_all_movies(source=tokenized_plots_bin, cluster_source=cluster_model_bin, sink=movie_encodings_bin)

    def encodings(self):
        data_folder = "data"
        movie_encodings_bin = os.path.join(data_folder, "movie_encodings.bin")
        return pd.read_pickle(movie_encodings_bin)
    
    def movieinfos(self):
        data_folder = "data"
        movie_info_bin = os.path.join(data_folder, "movie_info.bin")
        return pd.read_pickle(movie_info_bin)