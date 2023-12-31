from safetensors import safe_open
from .utils import populate_snapshot


class DreamBoothModel:
    def __init__(self, name: str):
        snapshot = populate_snapshot("dream_booth/" + name)

        # Load the model states
        self.states = {}
        with safe_open(snapshot, framework="pt", device="cpu") as model:
            for key in model.keys():
                self.states[key] = model.get_tensor(key)
