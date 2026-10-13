from src.semhand.pixel_feature_student import build_model


def test_build_model_is_random_init_and_has_feature_projector():
    model = build_model(teacher_dim=32)
    assert model.classifier.out_features == 1
    assert model.projector.out_features == 32
