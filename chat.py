from service import UserPreferences, MovieRecommendation, MovieService
from transformers import pipeline
import random
import spacy

class RequestChecklist:
    def __init__(self):
        self.CheckedDirectors = False
        self.CheckedActors = False
        self.CheckedMovies = False
        self.CheckedDescriptions = False
        self.CheckedGenres = False
        self.CheckedPeriod = False
        self.CheckedAmerican = False
        self.CheckedForeign = False
        self.CheckedBollywood = False
    def done_yet(self):
        done = self.CheckedDirectors \
                and self.CheckedActors \
                and self.CheckedMovies \
                and self.CheckedDescriptions \
                and self.CheckedGenres \
                and self.CheckedPeriod \
                and self.CheckedAmerican \
                and self.CheckedForeign \
                and self.CheckedBollywood
        return done
    def next_request(self):
        # randomly select the next request
        odds = 2
        if not self.CheckedDirectors:
            if random.randint(0, odds) == 0:
                self.CheckedDirectors = True
                return 'LikedDirectors'
        if not self.CheckedActors:
            if random.randint(0, odds) == 0:
                self.CheckedActors = True
                return 'LikedActors'
        if not self.CheckedMovies:
            if random.randint(0, odds) == 0:
                self.CheckedMovies = True
                return 'LikedMovies'
        if not self.CheckedDescriptions:
            if random.randint(0, odds) == 0:
                self.CheckedDescriptions = True
                return 'DescribedMovies'
        if not self.CheckedGenres:
            if random.randint(0, odds) == 0:
                self.CheckedGenres = True
                return 'LikedGenres'
        if not self.CheckedPeriod:
            if random.randint(0, odds) == 0:
                self.CheckedPeriod = True
                return 'TimePeriod'
        if not self.CheckedAmerican:
            if random.randint(0, odds) == 0:
                self.CheckedAmerican = True
                return 'AmericanFilms'
        if not self.CheckedForeign:
            if random.randint(0, odds) == 0:
                self.CheckedForeign = True
                return 'ForeignFilms'
        if not self.CheckedBollywood:
            if random.randint(0, odds) == 0:
                self.CheckedBollywood = True
                return 'BollywoodFilms'

        # if by random chance we didn't return a request, try again
        if not self.done_yet():
            return self.next_request()
        
        # if there are no requests left, we need to process the info
        return 'Waiting'

        

class Chatter:
    def __init__(self):
        self.Prompts = {
            'Intro': 'Hello! I hope you\'re doing well, let\'s find you some movies to watch.',
            'Enter': 'Alright, let\'s get started then!',
            'LikedMovies': 'Are there any movies that you really loved?',
            'DescribedMovies': 'Tell me about the type of movies you want to see. What would they be about?',
            'LikedDirectors': 'Do you have any directors whose work you really enjoyed?',
            'LikedGenres': 'Are there any genres in particular that you are looking for?',
            'LikedActors': 'Are there any actors or actresses that you want to watch?',
            'TimePeriod': 'Is there any particular time period you want to watch movies from?',
            'AmericanFilms': 'Are you interested in watching American films?',
            'ForeignFilms': 'Have you ever had an interest in foreign films?',
            'BollywoodFilms': 'Would you like to watch any Bollywood movies?',
            'Redirect': 'I\'m not sure that was on topic, could we steer back to our movie discussion?',
            'SaidYes': 'Perfect, I\'ll make a note of that',
            'SaidNo': 'Alright, no worries',
            'Waiting': 'Please wait a moment while I consider the information you\'ve provided me...',
            'Recommend': "Here are some movies that I thought you might enjoy.",
            'Outro': 'I hope that my recommendations will be of service to you. Come back soon!'
        }
        self.Paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")
        self.Language = spacy.load("en_core_web_lg")
        self.Checklist = RequestChecklist()
        self.MovieService = MovieService()
        self.UserPreferences = UserPreferences()

    def get_liked_movies(self, user_response):
        liked_movies = []
        for entity in user_response.ents:
            if entity.label_ == 'WORK_OF_ART':
                liked_movies.append(entity.text)
        return liked_movies if liked_movies else None

    def get_described_movies(self, user_response):
        described = []
        for sent in user_response.sents:
            described.append(sent.text)
        return described if described else None

    def get_liked_directors(self, user_response):
        liked_directors = []
        for entity in user_response.ents:
            if entity.label_ == 'PERSON':
                liked_directors.append(entity.text)
        return liked_directors if liked_directors else None

    def get_liked_genres(self, user_response):
        liked_genres = []
        for token in user_response:
            if token.pos_ in ['NOUN', 'ADJ']:
                liked_genres.append(token.text)
        return liked_genres if liked_genres else None

    def get_liked_actors(self, user_response):
        liked_actors = []
        for entity in user_response.ents:
            if entity.label_ == 'PERSON':
                liked_actors.append(entity.text)
        return liked_actors if liked_actors else None

    def get_time_period(self, user_response):
        before = None
        after = None
        for token in user_response:
            if token.lemma_.lower() == 'before' and not before:
                before = True
            elif token.lemma_.lower() == 'after' and not after:
                after = True
            elif token.lemma_.lower() == 'from' and not after:
                after = True
            elif token.lemma_.lower() == 'to' and not before:
                before = True
            elif token.like_num and before:
                before = int(token.text)
            elif token.like_num and after:
                after = int(token.text)
        return (before, after)

    def get_american_films(self, user_response):
        for token in user_response:
            if token.lemma_.lower() == 'no':
                return False
        return True

    def get_foreign_films(self, user_response):
        for token in user_response:
            if token.lemma_.lower() == 'no':
                return False
        return True

    def get_bollywood_films(self, user_response):
        for token in user_response:
            if token.lemma_.lower() == 'no':
                return False
        return True

    def get_and_parse_user_response(self, next_request: str):
        self.generate_prompt(next_request)
        
        response = input(">>> ")

        if self.detect_off_topic(response):
            self.generate_prompt('Redirect')
            return self.get_and_parse_user_response(next_request)
        elif self.detect_refusal(response) and next_request not in ['AmericanFilms', 'ForeignFilms', 'BollywoodFilms']:
            self.generate_prompt('SaidNo')
            return False
        elif self.detect_ready_for_recommendation(response):
            return True
        
        self.generate_prompt('SaidYes')

        tokenized_response = self.Language(response)

        if next_request == 'LikedMovies':
            self.UserPreferences.KnownLikedTitles = self.get_liked_movies(tokenized_response)
        if next_request == 'DescribedMovies':
            self.UserPreferences.DescribedPlots = self.get_described_movies(tokenized_response)
        if next_request == 'LikedDirectors':
            self.UserPreferences.Directors = self.get_liked_directors(tokenized_response)
        if next_request == 'LikedGenres':
            self.UserPreferences.Genres = self.get_liked_genres(tokenized_response)
        if next_request == 'LikedActors':
            self.UserPreferences.Actors = self.get_liked_actors(tokenized_response)
        if next_request == 'TimePeriod':
            self.UserPreferences.MoviesBefore, self.UserPreferences.MoviesAfter = self.get_time_period(tokenized_response)
        if next_request == 'AmericanFilms':
            self.UserPreferences.AllowAmericanOrigin = self.get_american_films(tokenized_response)
        if next_request == 'ForeignFilms':
            self.UserPreferences.AllowOtherOrigin = self.get_foreign_films(tokenized_response)
        if next_request == 'BollywoodFilms':
            self.UserPreferences.AllowBollywoordOrigin = self.get_bollywood_films(tokenized_response)

        return False

    # detect if the user has gone off topic (currently ignore this one)
    def detect_off_topic(self, user_input: str):
        return False

    # detect if the user wants a recommendation given the current information (currently ignore this one)
    def detect_ready_for_recommendation(self, user_input: str):
        return False

    # detect if the user declines to answer a question
    def detect_refusal(self, user_input: str):
        tokenized = self.Language(user_input)
        for token in tokenized:
            if token.lemma_.lower() == 'no':
                return True
        return False

    def generate_prompt(self, part_of_conversation: str):
        original_prompt = self.Prompts[part_of_conversation]
        paraphrased = self.Paraphraser(original_prompt, max_length = 100)[0]
        new_prompt = paraphrased['generated_text']
        print(f'\n{new_prompt}\n')
    
    def recommend_movie(self, movie_recommendation: MovieRecommendation):
        recommended_because = movie_recommendation.explain(True, True)
        
        print(f"- {movie_recommendation.RecommendedMovie.describe(True)}")
        
        because = True
        for line in recommended_because:
            paraphrased = self.Paraphraser(line, max_length = 100)[0]
            new_line = paraphrased['generated_text']
            if because:
                print(new_line)
                because = False
            else:
                print(f'\t- {new_line}')
            
    def chat(self):
        self.generate_prompt('Intro')

        while not self.Checklist.done_yet():
            next_request = self.Checklist.next_request()
            breakout = self.get_and_parse_user_response(next_request)
            if breakout:
                break

        waiting = self.Checklist.next_request()
        self.generate_prompt(waiting)

        recommendations = self.MovieService.recommend_from_user_preferences(self.UserPreferences, top_n=3, similarity_threshold=0.5)

        self.generate_prompt('Recommend')
        
        for recommendation in recommendations:
            self.recommend_movie(recommendation)

        self.generate_prompt('Outro')

        input("Press enter to quit...")
