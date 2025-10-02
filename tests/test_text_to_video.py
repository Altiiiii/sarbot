from text_to_video.pipeline import StoryboardBuilder, FrameRenderer, StoryboardSegment


def test_storyboard_builder_splits_long_text():
    builder = StoryboardBuilder(seconds_per_segment=2.0, max_chars=40)
    text = "Yapay zeka metinleri sahnelere dönüştürür. Her sahne video içinde belirli bir süre kalır."
    segments = builder.build(text)
    assert len(segments) == 2
    assert all(segment.duration == 2.0 for segment in segments)


def test_frame_renderer_output_shape():
    renderer = FrameRenderer(width=640, height=360)
    segment = StoryboardSegment(text="Merhaba dünya", duration=1.0)
    frame = renderer.render(segment)
    assert frame.shape == (360, 640, 3)
