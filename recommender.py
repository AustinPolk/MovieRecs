from trainer import Trainer
from loader import Loader
import pickle

class Recommender:
    def __init__(self):
        self.Trainer: Trainer = None
        self.Loader: Loader = None
        self.EncodedMovies: dict = {}

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

        
