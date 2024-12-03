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

    def get_liked_movies(self, user_response, debug: bool):
        liked_movies = []
        if debug:
            print('- get_liked_movies(DEBUG) -')
        for entity in user_response.ents:
            if debug:
                print(f'Detected entity: {entity.text}, {entity.label_}')
            if entity.label_ == 'WORK_OF_ART':
                liked_movies.append(entity.text)
        if debug:
            print(f'Detected movies: {liked_movies}')
        return liked_movies if liked_movies else None

    def get_described_movies(self, user_response, debug: bool):
        described = []
        for sent in user_response.sents:
            described.append(sent.text)
        if debug:
            print('- get_described_movies(DEBUG) -')
            print(f'Detected described plots: {described}')
        return described if described else None

    def get_liked_directors(self, user_response, debug: bool):
        liked_directors = []
        if debug:
            print('- get_liked_directors(DEBUG) -')
        for entity in user_response.ents:
            if debug:
                print(f'Detected entity: {entity.text}, {entity.label_}')
            if entity.label_ == 'PERSON':
                liked_directors.append(entity.text)
        if debug:
            print(f'Detected directors: {liked_directors}')
        return liked_directors if liked_directors else None

    def get_liked_genres(self, user_response, debug: bool):
        liked_genres = []
        for token in user_response:
            if token.pos_ in ['NOUN', 'ADJ']:
                liked_genres.append(token.text)
        if debug:
            print('- get_liked_genres(DEBUG) -')
            print(f'Detected genres: {liked_genres}')
        return liked_genres if liked_genres else None

    def get_liked_actors(self, user_response, debug: bool):
        liked_actors = []
        if debug:
            print('- get_liked_actors(DEBUG) -')
        for entity in user_response.ents:
            if debug:
                print(f'Detected entity: {entity.text}, {entity.label_}')
            if entity.label_ == 'PERSON':
                liked_actors.append(entity.text)
        if debug:
            print(f'Detected actors: {liked_actors}')
        return liked_actors if liked_actors else None

    def get_time_period(self, user_response, debug: bool):
        need_before = False
        need_after = False
        before = None
        after = None
        for token in user_response:
            if token.lemma_.lower() == 'before' and not need_before:
                need_before = True
            elif token.lemma_.lower() == 'after' and not need_after:
                need_after = True
            # elif token.lemma_.lower() == 'from' and not after:
            #     after = True
            # elif token.lemma_.lower() == 'to' and not before:
            #     before = True
            elif token.like_num and need_before:
                before = int(token.text)
                need_before = False
            elif token.like_num and need_after:
                after = int(token.text)
                need_after = False
        if debug:
            print('- get_time_period(DEBUG) -')
            if before:
                print(f'Detected before = {before}')
            if after:
                print(f'Detected after = {after}')
        return (before, after)

    def get_and_parse_user_response(self, next_request: str, debug: bool):
        self.generate_prompt(next_request, debug)
        
        response = input(">>> ")

        if self.detect_off_topic(response):
            self.generate_prompt('Redirect', debug)
            return self.get_and_parse_user_response(next_request)
        elif self.detect_refusal(response) and next_request not in ['AmericanFilms', 'ForeignFilms', 'BollywoodFilms']:
            self.generate_prompt('SaidNo', debug)
            return False
        elif self.detect_ready_for_recommendation(response):
            return True
        
        self.generate_prompt('SaidYes', debug)

        tokenized_response = self.Language(response)

        if next_request == 'LikedMovies':
            self.UserPreferences.KnownLikedTitles = self.get_liked_movies(tokenized_response, debug)
        if next_request == 'DescribedMovies':
            self.UserPreferences.DescribedPlots = self.get_described_movies(tokenized_response, debug)
        if next_request == 'LikedDirectors':
            self.UserPreferences.Directors = self.get_liked_directors(tokenized_response, debug)
        if next_request == 'LikedGenres':
            self.UserPreferences.Genres = self.get_liked_genres(tokenized_response, debug)
        if next_request == 'LikedActors':
            self.UserPreferences.Actors = self.get_liked_actors(tokenized_response, debug)
        if next_request == 'TimePeriod':
            self.UserPreferences.MoviesBefore, self.UserPreferences.MoviesAfter = self.get_time_period(tokenized_response, debug)
        if next_request == 'AmericanFilms':
            self.UserPreferences.AllowAmericanOrigin = True
        if next_request == 'ForeignFilms':
            self.UserPreferences.AllowOtherOrigin = True
        if next_request == 'BollywoodFilms':
            self.UserPreferences.AllowBollywoordOrigin = True

        return False

    # detect if the user has gone off topic (currently ignore this one)
    def detect_off_topic(self, user_input: str):
        return False

    # detect if the user wants a recommendation given the current information (currently ignore this one)
    def detect_ready_for_recommendation(self, user_input: str):
        return False

    # detect if the user declines to answer a question
    def detect_refusal(self, user_input: str):
        # just detect if 'no' was one of the first words in the sentence
        tokenized = self.Language(user_input)
        for token in tokenized[:3]:
            if token.lemma_.lower() == 'no':
                return True
        return False

    def generate_prompt(self, part_of_conversation: str, debug: bool):
        original_prompt = self.Prompts[part_of_conversation]
        paraphrased = self.Paraphraser(original_prompt, max_length = 100)[0]
        new_prompt = paraphrased['generated_text']
        if debug:
            print('- generate_prompt(DEBUG) -')
            print(f'Original prompt: {original_prompt}')
            print(f'New prompt: {new_prompt}')
        print(f'\n{new_prompt}\n')
    
    def recommend_movie(self, movie_recommendation: MovieRecommendation, debug: bool):
        recommended_because = movie_recommendation.explain(True, True)
        
        print(f"- {movie_recommendation.RecommendedMovie.describe(True)}")
        
        because = True
        for line in recommended_because:
            paraphrased = self.Paraphraser(line, max_length = 100)[0]
            new_line = paraphrased['generated_text']
            if debug:
                print('- recommend_movie(DEBUG) -')
                print(f'Original explanation: {line}')
                print(f'New explanation: {new_line}')
            if because:
                print(new_line)
                because = False
            else:
                print(f'\t- {new_line}')
            
    def chat(self, debug: bool = False):
        self.generate_prompt('Intro', debug)

        while not self.Checklist.done_yet():
            next_request = self.Checklist.next_request()
            breakout = self.get_and_parse_user_response(next_request, debug)
            if breakout:
                break

        waiting = self.Checklist.next_request()
        self.generate_prompt(waiting, debug)

        recommendations = self.MovieService.recommend_from_user_preferences(self.UserPreferences, top_n=3, mixed_results=True, similarity_threshold=0.5)

        self.generate_prompt('Recommend', debug)
        
        for recommendation in recommendations:
            self.recommend_movie(recommendation, debug)
            print()

        self.generate_prompt('Outro', debug)

        input("Press enter to quit...")
