from minimal_animatediff.motion_module import MotionModuleModel


def test_init_motion_module_model():
    model = MotionModuleModel("mm_sd_v15.ckpt")
    assert len(model.states.keys()) == 560


def test_init_other_motion_module_model():
    model = MotionModuleModel("mm_sd_v14.ckpt")
    assert len(model.states.keys()) == 560


def test_init_non_existing_motion_module_model():
    try:
        MotionModuleModel("invalid.ckpt")
        assert False
    except FileNotFoundError:
        pass
