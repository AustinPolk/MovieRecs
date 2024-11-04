from spacy import Language, load

class MovieEncoder:
        
    def __init__(self):
        self.LanguageModel: Language = load("en_core_web_sm")
        self.WordClasses: list[str] = ["NOUN", "VERB", "ADJ", "ADV"]
        pass

    def get_single_keyword_sets(self, plot_str: str):
        keyword_sets = {}
        for word_class in self.WordClasses:
            keyword_sets[word_class] = set()

        tokens = self.LanguageModel(plot_str)
        for token in tokens:
            if token.pos_ in self.WordClasses:
                lem = token.lemma_
                keyword_sets[token.pos_].add(lem)

        return keyword_sets

    def get_many_keyword_lists(self, plots: dict[str, str]):
        keyword_sets = {}
        for word_class in self.WordClasses:
            keyword_sets[word_class] = set()

        for _, plot_str in plots.items():
            these_keyword_sets = self.get_single_keyword_sets(plot_str)
            for word_class in keyword_sets:
                keyword_sets[word_class] |= these_keyword_sets[word_class]

        keyword_lists = {}
        for word_class, keyword_set in keyword_sets.items():
            keyword_lists[word_class] = list(keyword_set)

        return keyword_lists
