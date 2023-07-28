import os

from huggingface_hub import snapshot_download
from safetensors import safe_open


class DreamBoothSnapshot:
    def __init__(self, name: str):
        # Populate the model snapshot
        snapshot_download(
            repo_id="threeal/AnimateDiffMirrors",
            local_dir="snapshots",
            allow_patterns=["DreamBooth/" + name],
        )

        # Load the model states
        self.states = {}
        path = os.path.join("snapshots/DreamBooth", name)
        with safe_open(path, framework="pt", device="cpu") as model:
            for key in model.keys():
                self.states[key] = model.get_tensor(key)
