from minimal_animatediff.dream_booth import DreamBoothModel


def test_init_dream_booth_model():
    model = DreamBoothModel("toonyou_beta3.safetensors")
    assert len(model.states.keys()) == 1133


def test_init_other_dream_booth_model():
    model = DreamBoothModel("lyriel_v16.safetensors")
    assert len(model.states.keys()) == 1131


def test_init_non_existing_dream_booth_model():
    try:
        DreamBoothModel("invalid.safetensors")
        assert False
    except FileNotFoundError:
        pass
