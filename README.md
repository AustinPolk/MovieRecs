# MovieRecs

_MovieRecs_ is a personalized movie recommendation agent that can gather conversational information from the user to suggest movies that they might enjoy. It has access to information for around 35,000 movies, spanning many genres, years, and countries of origin, making for a diverse range of movies to recommend from. Built entirely in Python, the source code and any required packages can be downloaded and run on any machine.

### Video Demo

A video demonstrating an example MovieRecs session can be found [here](https://youtu.be/IZpuDGD65HM)

## To Run

To run this on a Windows machine, first clone the repository. Ensure that Python is installed on the machine, and use Pip to install all packages in *requirements.txt*. On your machine, navigate to the repository folder and open the PowerShell. To start _MovieRecs_, use the following command:

> python run.py

## Data Sources

The raw movie information was sourced from a dataset on Kaggle, [Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots). The specific information for each movie, including items like the genre and plot, was then parsed and structured into a more usable format. Multiple stages of data processing were used, the results of which are cached and used in _MovieRecs_.

## Methodology

_MovieRecs_ uses a recommendation system that recommends the movies with the largest amount of favorable traits. Traits include the genre, plot themes, cast, and other identifying information about the movie. The traits for each movie are evaluated against user preferences and a recommendation score is generated based on the number of preferred traits. The top 3 movies with the highest scores are then returned as recommendations to the user.

### Evaluating Similar Movies

_MovieRecs_ also implements a system for comparing themes found in the plots of movies to enhance recommendations based on known movies. This involves generating a vector embedding for each movie plot and calculating the Euclidean distance between them to measure similarity. To generate these embeddings, the plots are tokenized, and each word is assigned to a precomputed theme "cluster." These clustering results are stored in a vector, which is then encoded to reduce it from several thousand to around a dozen dimensions. This is the vector that is used when comparing the themes identified in the plot of a given movie.
