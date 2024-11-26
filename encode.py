from fuzzywuzzy import fuzz

class SparseVectorEncoding:
    def __init__(self):
        self.Dimensions: dict[int, float] = {}  # actual values in the vector, indexed by dimension
        self.Norm: float = None                 # magnitude of the vector, used in normalization/cosine similarity
    def __getitem__(self, index: int):
        if index not in self.Dimensions:    # any value not recorded is assumed 0 in the vector
            return 0
        return self.Dimensions[index]
    def __setitem__(self, index: int, value: float):
        self.Dimensions[index] = value
    def normalize(self):
        self.Norm = sum(A * A for A in self.Dimensions.values()) ** 0.5
        for dim in self.Dimensions:
            self[dim] /= self.Norm
    def normed_cosine_similarity(self, other):
        if not self.Norm:
            self.normalize()
        if not other.Norm:
            other.normalize()
        common_dims = set(self.Dimensions.keys()) & set(other.Dimensions.keys())
        similarity = 0
        for dim in common_dims:
            similarity += self[dim] * other[dim]
        return similarity

# for now just relies on string similarity, in the future could be name vectors
class EntityEncoding:
    def __init__(self, entity: str, label: str):
        self.EntityName: str = entity.lower()
        self.EntityLabel: str = label.lower()
    def similarity(self, other):
        if self.EntityLabel != other.EntityLabel:
            return 0
        return fuzz.ratio(self.EntityName, other.EntityName) / 100

class MovieEncoding:
    def __init__(self):
        self.PlotEncoding: SparseVectorEncoding = SparseVectorEncoding()
        self.EntityEncodings: list[EntityEncoding] = []
    def add_entity(self, entity: str, label: str):
        entity_encoding = EntityEncoding(entity, label)
        if self.EntityEncodings:
            max_similarity = max(entity_encoding.similarity(x) for x in self.EntityEncodings)
        else:
            max_similarity = 0.0
        if max_similarity < 0.95:    # don't add if it is too similar, it is likely a repeat
            self.EntityEncodings.append(entity_encoding)
    def estimate_entity_matches(self, other):
        these_entities = list(self.EntityEncodings)
        those_entities = list(other.EntityEncodings)
        matches = 0

        # attempt to make a 1 to 1 matching from these entities to those entities
        while these_entities and those_entities:
            ent = these_entities.pop()
            best_match = None
            best_similarity = 0
            for other_ent in those_entities:
                similarity = ent.similarity(other_ent)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = other_ent
            if best_similarity > 0.85:  # threshold for what counts as a match
                matches += 1
                those_entities.remove(best_match)
        
        return matches
    def similarity(self, other):
        max_matches = min(len(self.EntityEncodings), len(other.EntityEncodings)) + 1    # +1 just to make the resultant similarity a little smaller
        ent_sim_score = self.estimate_entity_matches(other) / max_matches
        plot_sim_score = self.PlotEncoding.normed_cosine_similarity(other.PlotEncoding)
        return 0.65 * plot_sim_score + 0.35 * ent_sim_score     # give the plot score a higher weight, but still let the entity score have some say
