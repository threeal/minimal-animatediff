from minimal_animatediff.motion_module import MotionModule


def test_init_motion_module():
    model = MotionModule("mm_sd_v15.ckpt")
    assert len(model.states.keys()) == 560


def test_init_other_motion_module():
    model = MotionModule("mm_sd_v14.ckpt")
    assert len(model.states.keys()) == 560


def test_init_non_existing_motion_module():
    try:
        MotionModule("invalid.ckpt")
        assert False
    except FileNotFoundError:
        pass
