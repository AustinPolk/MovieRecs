from keras import Model, Sequential, layers
from encode import SparseVectorEncoding, MovieEncoding
import numpy as np
import random

class Autoencoder(Model):
    def __init__(self, input_size: int, hidden_size: int, latent_size: int, activation: str):
        super(Autoencoder, self).__init__()
        self.input_size = input_size
        self.activation = activation
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.denormalized: bool = False
        self.encoder = Sequential([
            layers.Dense(hidden_size, activation=activation),
            layers.Dense(latent_size, activation=activation),
        ])
        self.decoder = Sequential([
            layers.Dense(hidden_size, activation=activation),
            layers.Dense(input_size, activation=activation),
            layers.Reshape((input_size,))
        ])
        self.compile(optimizer='adam', loss='mse')
        
    def sparse_encoding_to_numpy(self, sparse: SparseVectorEncoding):
        vec = np.zeros(self.input_size)
        for dim in sparse.Dimensions:
            vec[dim] = sparse.Dimensions[dim]
        return vec

    def get_config(self):
        return {
            'input_size': self.input_size,
            'activation': self.activation,
            'latent_size': self.latent_size,
            'hidden_size': self.hidden_size
        }
    
    def call(self, x):
        if isinstance(x, SparseVectorEncoding):
            vec = self.sparse_encoding_to_numpy(x)
            if self.denormalized and x.Norm:
                vec *= x.Norm
            x = vec.reshape(1, -1)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def train_on_movie_encodings(self, movie_encodings: dict[int, MovieEncoding], denormalize: bool):
        self.denormalized = denormalize

        # take a random sample from the movie encodings
        random_encodings = [movie_encodings[id] for id in movie_encodings]
        random.shuffle(random_encodings)
        train_encodings = random_encodings[:5000]
        test_encodings = random_encodings[5000:6500]

        # convert the movie encodings to numpy vecs
        vecs = []
        for movie_encoding in train_encodings:
            plot_encoding = movie_encoding.PlotEncoding
            plot_vector = self.sparse_encoding_to_numpy(plot_encoding)
            if denormalize and plot_encoding.Norm:
                plot_vector *= plot_encoding.Norm
            vecs.append(plot_vector)
        vecs = np.array(vecs)

        # then train on this data
        self.fit(vecs, vecs, epochs=10, verbose=0)
        
        # report the loss on the test data
        vecs = []
        for movie_encoding in test_encodings:
            plot_encoding = movie_encoding.PlotEncoding
            plot_vector = self.sparse_encoding_to_numpy(plot_encoding)
            if denormalize and plot_encoding.Norm:
                plot_vector *= plot_encoding.Norm
            vecs.append(plot_vector)
        vecs = np.array(vecs)

        final_loss = self.evaluate(vecs, vecs)

        return final_loss