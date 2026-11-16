import numpy as np
import pytest
from scipy.ndimage import find_objects, label

from biodetect import segmentation as seg
from biodetect.features import box_cells


def test_gaussian_kernel_matches_imgaussfilt_default():
    k = seg.gaussian_kernel()
    assert k.shape == (3, 3)
    assert k.sum() == pytest.approx(1.0)
    assert k[1, 1] == pytest.approx(0.6193, abs=1e-4)          # MATLAB fspecial('gaussian', 3, 0.5)
    np.testing.assert_allclose(k, k.T)
    np.testing.assert_allclose(k, k[::-1, ::-1])


def _brute_otsu_level(img):
    """Otsu over 256 quantised levels by trying every threshold; ties averaged."""
    lo, hi = img.min(), img.max()
    q = np.round((img - lo) / (hi - lo) * 255).astype(int).ravel()
    best, ks = -1.0, []
    for k in range(255):
        c0, c1 = q[q <= k], q[q > k]
        if len(c0) == 0 or len(c1) == 0:
            continue
        w0, w1 = len(c0) / len(q), len(c1) / len(q)
        sb = w0 * w1 * (c0.mean() - c1.mean()) ** 2
        if sb > best + 1e-12:
            best, ks = sb, [k]
        elif abs(sb - best) <= 1e-12:
            ks.append(k)
    return lo + np.mean(ks) / 255 * (hi - lo)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_graythresh_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    img = np.concatenate([rng.gamma(2.0, 1.0, 3000), rng.normal(12, 2, 800)]).reshape(40, 95)
    _, thr = seg.graythresh_binarize(img)
    assert thr == pytest.approx(_brute_otsu_level(img), rel=1e-9)


def test_graythresh_separates_two_levels():
    img = np.full((20, 30), 0.2)
    img[5:9, 10:20] = 0.8
    mask, thr = seg.graythresh_binarize(img)
    assert 0.2 <= thr < 0.8
    np.testing.assert_array_equal(mask, img > 0.5)


def test_graythresh_constant_image():
    mask, thr = seg.graythresh_binarize(np.ones((4, 4)))
    assert not mask.any() and np.isnan(thr)


def test_noise_reduced_is_non_negative_and_mostly_zero(soundscape):
    x, _ = soundscape
    f, t, s1 = seg.noise_reduced(x[: 48000 * 10], 48000)
    assert s1.shape == (seg.N_FFT // 2 + 1, len(t))
    assert s1.min() >= 0
    assert (s1 == 0).mean() > 0.3                                # about half of each row sits below its mode


def _overlaps(ev, boxes):
    return ((boxes.begin_s < ev.end_s) & (boxes.end_s > ev.begin_s)
            & (boxes.low_hz <= ev.high_hz + 100) & (boxes.high_hz >= ev.low_hz - 100)).any()


def test_every_synthetic_event_is_boxed(soundscape):
    x, events = soundscape
    boxes, info = seg.segment(x, 48000)
    assert info["segments"] == len(boxes) > 0
    missed = [e.kind for e in events.itertuples() if not _overlaps(e, boxes)]
    assert not missed, f"events without a box: {missed}"


def test_boxes_lie_inside_the_spectrogram(soundscape):
    x, _ = soundscape
    b, _ = seg.segment(x, 48000, t0=100.0)
    assert (b.begin_s >= 100.0 - 1e-9).all() and (b.end_s <= 160.0 + 1e-9).all()
    assert (b.low_hz >= 0).all() and (b.high_hz <= 24000).all()
    assert (b.end_s > b.begin_s).all() and (b.high_hz > b.low_hz).all()
    assert (b.n_cells >= 1).all() and (b.mean_db > 0).all()


def test_white_noise_gives_mostly_speckle():
    x = np.random.default_rng(0).standard_normal(48000 * 60) * 0.05
    b, _ = seg.segment(x, 48000)
    assert len(b) > 100
    assert (b.n_cells < 50).mean() > 0.99


def test_box_cells_inverts_segment(soundscape):
    """features.box_cells must recover each segment's exact cell range from its box."""
    x, _ = soundscape
    x = x[: 48000 * 20]
    f, t, s1 = seg.noise_reduced(x, 48000)
    mask = seg.clean_mask(seg.graythresh_binarize(s1)[0])
    lab, _ = label(mask, structure=seg.CONN8)
    boxes, _ = seg.segment(x, 48000)
    dt, df = (seg.WIN - seg.OVERLAP) / 48000, 48000 / seg.N_FFT
    for box, sl in zip(boxes.itertuples(), find_objects(lab), strict=True):
        assert box_cells(box, 0.0, t[0], dt, df) == (sl[0].start, sl[0].stop, sl[1].start, sl[1].stop)


@pytest.mark.parametrize("duration, chunk, expected", [
    (60.0, 60.0, [0.0, 60.0]),
    (150.0, 60.0, [0.0, 60.0, 120.0, 150.0]),
    (125.0, 60.0, [0.0, 60.0, 125.0]),                           # 5 s tail joins the piece before
    (125.0, 0.0, [0.0, 125.0]),                                  # chunk 0 = whole file
    (30.0, 60.0, [0.0, 30.0]),
])
def test_piece_edges(duration, chunk, expected):
    np.testing.assert_allclose(seg.piece_edges(duration, chunk), expected)


def test_find_wavs_rejects_repeated_names(tmp_path):
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "rec.wav").write_bytes(b"")
    with pytest.raises(ValueError, match="rec"):
        seg.find_wavs([tmp_path])


def test_find_wavs_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        seg.find_wavs([tmp_path / "nope.wav"])
