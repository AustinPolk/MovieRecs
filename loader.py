import pickle

class Loader:
    def __init__(self):
        pass

    def dopickle(filepath: str, thing):
        try:
            with open(filepath, "wb+") as f:
                p = pickle.dump(thing, f)
        except:
            return

    def _unpickle(self, filepath: str):
        p = None
        try:
            with open(filepath, "rb") as f:
                p = pickle.load(f)
        except:
            p = None
        return p

    def load(self, from_file: str):
        thing = self._unpickle(f"cache\\{from_file}.bin")
        return thing
    
    def save(self, thing, to_file: str):
        self.dopickle(f"cache\\{to_file}.bin", thing)
        return

    def load_cached_plots(self):
        plots_by_id = self._unpickle("cache\\plots_by_id.bin")
        return plots_by_id
    
    def load_cached_titles(self):
        movies_by_id = self._unpickle("cache\\movies_by_id.bin")
        return movies_by_id