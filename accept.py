class TokenAccepter:
    def __init__(self):
        pass
    def accept(self, token):
        if token.pos_ not in ["NOUN", "VERB", "ADJ", "ADV"]:    # only accept words in an open class
            return False
        if not token.has_vector:    # only accept words with available vector embeddings
            return False
        return True
    
class EntityAccepter:
    def __init__(self):
        pass
    def accept(self, entity):
        if entity.label_ in ["TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]:    # do not accept these entity types, they aren't very useful
            return False
        return True