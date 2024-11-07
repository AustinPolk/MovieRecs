import pickle

class Loader:
    def __init__(self):
        pass

    def _unpickle(self, filepath: str):
        p = None
        try:
            with open(filepath, "rb") as f:
                p = pickle.load(f)
        except:
            p = None
        return p

    def load_cached_plots(self):
        plots_by_id = self._unpickle("cache\\plots_by_id.bin")
        return plots_by_id