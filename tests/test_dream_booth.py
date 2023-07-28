from minimal_animatediff.dream_booth import DreamBoothSnapshot


def test_init_dream_booth_snapshot():
    model = DreamBoothSnapshot("toonyou_beta3.safetensors")
    assert len(model.states.keys()) == 1133
