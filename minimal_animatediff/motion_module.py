import torch

from .utils import populate_snapshot


class MotionModuleModel:
    def __init__(self, name: str):
        snapshot = populate_snapshot("motion_modules/" + name)
        self.states = torch.load(snapshot, map_location="cpu")
