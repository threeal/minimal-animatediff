from minimal_animatediff.dream_booth import DreamBoothModel


def test_init_dream_booth_model():
    model = DreamBoothModel("toonyou_beta3.safetensors")
    assert len(model.states.keys()) == 1133
