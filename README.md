# MovieReviews

Given a list of movies the user likes, a model will be trained to identify whether the user will like a given movie based on the available plot information. This can save the user time researching a movie to determine whether they will like it, and also avoids the issue of spoilers by keeping the actual plot information hidden from the user.

## Data Sources

This project aims to take qualitative information about the plot of a movie and generate quantitative data that can be used to identify that story. The main transformation from words to data comes in the form of word vector embeddings, such as those supplied by a pre-trained model from Google (https://code.google.com/archive/p/word2vec/). Alternatively, text may be compiled from many movie plots and a new model trained on them for this specific purpose. This text can be drawn from plot summaries on Wikipedia, The Movie Spoiler, etc. These sources will also be used when evaluating individual movie plots, both to train a user preference model and to test a given movie against that model.

## Methodology

Via vector embeddings of text from a plot summary, a single high-dimensional vector will be produced which embeds key information about that plot. A vector embedding model will derive this final vector through analysis of individual components of the plot, and vector embeddings of the text therein. 

### Benchmarking

To benchmark the efficacy of the plot vector embedding, an analysis of the clustering of the results can be performed. It would be expected that similar movies, such as those within a series or saga, would have similar vector embeddings clustered near each other. In the same way, it would be expected that very different movies would not cluster near each other. At the same time, it may be expected that movies of a similar genre would cluster closer together than those of different genres. Analyzing movies with very clearly defined similarities and differences will show how the results of a given embedding model may differ from expectation.

## User Preference Model

As mentioned above, the user will supply a list of movies they like. Each of these will be evaluated using their plot information, resulting in vectors in a higher dimensional space that represent those movies. These vectors can now be used as training data for a model, such as an RBF machine or a neural network. Then, given a new movie plot vector, this newly trained model can assess whether or not the user will like a movie.
