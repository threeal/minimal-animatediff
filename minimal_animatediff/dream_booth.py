from safetensors import safe_open


class DreamBoothModel:
    def __init__(self, model_file):
        self.state_dict = {}
        with safe_open(model_file, framework="pt", device="cpu") as model:
            for key in model.keys():
                self.state_dict[key] = model.get_tensor(key)
