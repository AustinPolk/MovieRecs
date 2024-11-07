from trainer import Trainer
from loader import Loader
import pickle
import numpy as np

class Recommender:
    def __init__(self):
        self.Trainer: Trainer = None
        self.Loader: Loader = None
        self.EncodedMovies: dict = {}
        self.MovieTitles: dict

        with open("cache\\movies_by_id.bin", "rb") as f:
            self.MovieTitles = pickle.load(f)

    def Save(self):
        with open("cache\\trainer.bin", "wb+") as f:
            pickle.dump(self.Trainer, f)
        with open("cache\\encoded_movies.bin", "wb+") as f:
            pickle.dump(self.EncodedMovies, f)

    def Load(self):
        with open("cache\\trainer.bin", "rb") as f:
            self.Trainer = pickle.load(f)
        with open("cache\\encoded_movies.bin", "rb+") as f:
            self.EncodedMovies = pickle.load(f)
        self.Loader = Loader()

    def TrainNew(self, hyperparameters: dict):
        self.Loader = Loader()
        cached_plots = self.Loader.load_cached_plots()

        self.Trainer = Trainer()
        self.Trainer.train_clusterings(cached_plots, 
                                       hyperparameters['Clusters'])
        self.Trainer.train_autoencoder(cached_plots, 
                                       hyperparameters['ContextWindow'], 
                                       hyperparameters['AutoEncoderLayers'], 
                                       hyperparameters['AutoEncoderActivations'])
        
        # also encode all the known movie plots
        for id, plot_str in cached_plots.items():
            self.EncodedMovies[id] = self.Trainer.plot_autoencoding(plot_str)

    def LoadOrTrainAndSave(self, hyperparameters: dict):
        self.Loader = Loader()
        cached_plots = self.Loader.load_cached_plots()
        
        try:
            with open("cache\\trainer.bin", "rb") as f:
                self.Trainer = pickle.load(f)
            print("Loading existing trainer")
        except:
            print("Creating new trainer")

            self.Trainer = Trainer()
            self.Trainer.train_clusterings(cached_plots, 
                                        hyperparameters['Clusters'])
            self.Trainer.train_autoencoder(cached_plots, 
                                        hyperparameters['ContextWindow'], 
                                        hyperparameters['AutoEncoderLayers'], 
                                        hyperparameters['AutoEncoderActivations'])
        
            with open("cache\\trainer.bin", "wb+") as f:
                pickle.dump(self.Trainer, f)
        
        try:
            with open("cache\\encoded_movies.bin", "rb+") as f:
                self.EncodedMovies = pickle.load(f)
            print("Loading existing movie embeddings")
        except:
            print("Creating movie embeddings")
            for id, plot_str in cached_plots.items():
                self.EncodedMovies[id] = self.Trainer.plot_autoencoding(plot_str)

            with open("cache\\encoded_movies.bin", "wb+") as f:
                pickle.dump(self.EncodedMovies, f)

    # return the similarity between two vectors using the given method, the higher the more similar
    def _encoding_similarity(self, e_s: np.ndarray, e_t: np.ndarray, method: str):
        smoothing = 0.001   # ensures no division by zero errors
        if method == "euclid":
            dist_squared = smoothing
            for i in range(e_s.shape[0]):
                dist_squared += (e_s[i] - e_t[i])**2
            return dist_squared**(-1/2)
        if method == "euclid_squared":
            dist_squared = smoothing
            for i in range(e_s.shape[0]):
                dist_squared += (e_s[i] - e_t[i])**2
            return dist_squared**(-1)
        if method == "average":
            total_deviation = smoothing
            for i in range(e_s.shape[0]):
                total_deviation += abs(e_s[i] - e_t[i])
            return (total_deviation / e_s.shape[0])
        if method == "maximum":
            max_deviation = -1
            for i in range(e_s.shape[0]):
                dev = abs(e_s[i] - e_t[i])
                if dev > max_deviation:
                    max_deviation = dev
            return 1 / (max_deviation + smoothing)
        if method == "cosine":
            dot_prod = 0
            s_mag = smoothing
            t_mag = smoothing
            for i in range(e_s.shape[0]):
                dot_prod += e_s[i] * e_t[i]
                s_mag += e_s[i] * e_s[i]
                t_mag += e_t[i] * e_t[i]
            return dot_prod / (s_mag * t_mag)**(1/2)

    def Describe(self, description: str, similarity_method: str, top_n: int = 5): 
        described = self.Trainer.plot_autoencoding(description)[0]

        sim_scores = {}
        for id, encoding in self.EncodedMovies.items():
            similarity = self._encoding_similarity(described, encoding[0], method=similarity_method)
            sim_scores[id] = similarity

        ranked = list(self.EncodedMovies.keys())
        ranked.sort(key=lambda x: sim_scores[x], reverse=True)

        top_ids = ranked[:top_n]
        top_titles = [self.MovieTitles[id] for id in top_ids]

        return top_titles
