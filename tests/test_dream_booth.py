from minimal_animatediff.dream_booth import DreamBoothModel
from minimal_animatediff import utils


def test_load_dream_booth_model():
    path = utils.get_model_path("toonyou_beta3.safetensors")
    model = DreamBoothModel(path)
    assert len(model.states.keys()) == 1133
