from keras import Model, Sequential, layers, losses

class Autoencoder(Model):
    def __init__(self, input_size: int, layer_sizes: list[int], layer_activations: list[str]):
        super(Autoencoder, self).__init__()
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.layer_activations = layer_activations
        self.encoder = Sequential([
            layers.Dense(layer_sizes[0], activation=layer_activations[0]),
            layers.Dense(layer_sizes[1], activation=layer_activations[1]),
            layers.Dense(layer_sizes[2], activation=layer_activations[2]),
        ])
        self.decoder = Sequential([
            layers.Dense(layer_sizes[3], activation=layer_activations[3]),
            layers.Dense(layer_sizes[4], activation=layer_activations[4]),
            layers.Dense(input_size, activation=layer_activations[5]),
            layers.Reshape((input_size,))
        ])
        self.compile(optimizer='adam', loss=losses.MeanSquaredError())

    def get_config(self):
        return {
            'input_size': self.input_size,
            'layer_sizes': self.layer_sizes,
            'layer_activations': self.layer_activations
        }
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded