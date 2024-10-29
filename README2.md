# The Process So Far

1. Tokenize movie plot synopsis, pull out all nouns and verbs
2. Assemble a vector based on keywords found within the tokenized plot
3. Run that vector through an encoder to reduce the dimensionality
4. Use a Random Forest classifier to classify the movie into two categories: Will Like (1), or Won't Like (0)
