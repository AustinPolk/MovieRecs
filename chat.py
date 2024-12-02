from service import UserPreferences, MovieRecommendation, MovieService
from transformers import pipeline
import random

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
            'Intro': 'Hello! I hope you\'re doing well, would you like me to help you find a movie to watch?',
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
        self.Checklist = RequestChecklist()
        self.MovieService = MovieService()
        self.UserPreferences = UserPreferences()

    def get_and_parse_user_response(self):
        pass

    # detect if the user wants a recommendation given the current information
    def detect_ready_for_recommendation(self, user_input: str):
        pass

    # detect if the user declines to answer a question
    def detect_refusal(self, user_input: str):
        pass

    def generate_prompt(self, part_of_conversation: str):
        original_prompt = self.Prompts[part_of_conversation]
        paraphrased = self.Paraphraser(original_prompt, max_length = 100)
        new_prompt = paraphrased['generated_text']
        print(f'\n{new_prompt}\n')
    
    def recommend_movie(self, movie_recommendation: MovieRecommendation):
        recommended_because = movie_recommendation.explain(True, True)
        
        print(f"- {movie_recommendation.RecommendedMovie.describe(True)}")
        
        because = True
        for line in recommended_because:
            paraphrased = self.Paraphraser(line, max_length = 100)
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
            self.generate_prompt()
            self.get_and_parse_user_response()

        waiting = self.Checklist.next_request()
        self.generate_prompt(waiting)

        recommendations = self.MovieService.recommend_from_user_preferences(self.UserPreferences, top_n=3, similarity_threshold=0.5)

        self.generate_prompt('Recommend')
        
        for recommendation in recommendations:
            self.recommend_movie(recommendation)

        self.generate_prompt('Outro')
