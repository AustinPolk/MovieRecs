import pickle
import os

def dopickle(filepath: str, thing):
    try:
        with open(filepath, "wb+") as f:
            p = pickle.dump(thing, f)
    except:
        return

def _unpickle(filepath: str):
    p = None
    try:
        with open(filepath, "rb") as f:
            p = pickle.load(f)
    except:
        p = None
    return p

def exists(file: str):
    return os.path.exists(f"cache\\{file}.bin")

def load(from_file: str):
    thing = _unpickle(f"cache\\{from_file}.bin")
    return thing

def save(thing, to_file: str):
    dopickle(f"cache\\{to_file}.bin", thing)
    return

def load_cached_plots():
    plots_by_id = _unpickle("cache\\plots_by_id.bin")
    return plots_by_id

def load_cached_titles():
    movies_by_id = _unpickle("cache\\movies_by_id.bin")
    return movies_by_id