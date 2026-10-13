from scripts.expand_handness_temporal_negatives import nearest_frames


def test_nearest_frames_limits_radius_and_preserves_time_order():
    frames = [80, 90, 94, 98, 100, 104, 108, 120]
    assert nearest_frames(frames, 100, limit=5, radius=10) == [94, 98, 100, 104, 108]
