import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from biodetect import features as feat
from biodetect import segmentation as seg
from biodetect import speckle_null
from biodetect.lamda import gad, lamda, scale


# ---------------------------------------------------------------- features
def test_cepstra_matches_explicit_sum():
    rng = np.random.default_rng(0)
    for nu in (1, 5, 30):
        q = rng.normal(size=(nu, 3))
        got = feat.cepstra(q)
        want = np.zeros((24, 3))
        for k in range(1, 25):
            npk = np.sqrt(1 / nu) if k == 1 else np.sqrt(2 / nu)
            for i in range(1, nu + 1):
                want[k - 1] += npk * q[i - 1] * np.cos(np.pi / nu * (i - 1) * (k - 0.5))
        np.testing.assert_allclose(got, want, atol=1e-12)


def test_frames_average_cepstra():
    s1 = np.random.default_rng(1).uniform(0, 20, (40, 60))
    f = np.arange(40) * 46.875
    one = feat.box_features(s1, f, 5, 25, 10, 50, frame_cols=0)
    framed = feat.box_features(s1, f, 5, 25, 10, 50, frame_cols=10)
    assert one.shape == framed.shape == (25,)
    assert one[0] == framed[0]                                   # f_peak does not depend on w
    assert not np.allclose(one[1:], framed[1:])


def test_f_peak_of_a_tone():
    sr = 48000
    rng = np.random.default_rng(2)
    t = np.arange(sr * 10) / sr
    x = 0.01 * rng.standard_normal(len(t))
    on = (t >= 4) & (t < 5)
    # bin 150 of a 1024-point FFT, Hann-enveloped: a constant-level tone would become the bin's modal
    # "noise" level and be subtracted away
    x[on] += 0.2 * np.hanning(on.sum()) * np.sin(2 * np.pi * 7031.25 * t[on])
    f, tt, s1 = seg.noise_reduced(x, sr)
    boxes, _ = seg.segment(x, sr)
    big = boxes.loc[boxes.n_cells.idxmax()]
    assert big.begin_s < 4.5 < big.end_s
    r0, r1, c0, c1 = feat.box_cells(big, 0.0, tt[0], (seg.WIN - seg.OVERLAP) / sr, sr / seg.N_FFT)
    y = feat.box_features(s1, f, r0, r1, c0, c1)
    assert y[0] == pytest.approx(7031.25, abs=sr / seg.N_FFT)
    assert np.isfinite(y).all()


# ---------------------------------------------------------------- LAMDA
def test_gad_formula():
    rng = np.random.default_rng(3)
    rho, x = rng.uniform(0.05, 0.95, 7), rng.uniform(0, 1, 7)
    m = rho ** x * (1 - rho) ** (1 - x)
    assert gad(rho, x) == pytest.approx(m.prod() / (m.prod() + (1 - m).prod()))
    assert gad(np.full(7, 0.5), x) == pytest.approx(0.5)         # the NIC is indifferent


def test_single_datum_prototype():
    x = np.array([[0.2, 0.7, 1.0]])
    idx, g, cp, n = lamda(x, n_iter=1)
    np.testing.assert_allclose(cp, (x + 0.5) / 2)
    assert idx.tolist() == [0] and n.tolist() == [1]


def test_three_blobs_give_three_clusters():
    rng = np.random.default_rng(4)
    centres = np.array([[0.1, 0.1, 0.1, 0.9], [0.9, 0.9, 0.1, 0.1], [0.1, 0.9, 0.9, 0.9]])
    truth = np.repeat(np.arange(3), 40)
    data = np.clip(centres[truth] + rng.normal(0, 0.03, (120, 4)), 0, 1)
    perm = rng.permutation(120)
    idx, g, cp, _ = lamda(data[perm])
    assert len(cp) == 3
    assert adjusted_rand_score(truth[perm], idx) == 1.0
    assert np.all((g > 0) & (g <= 1))


def test_lamda_is_deterministic_and_order_dependent():
    data = np.random.default_rng(5).uniform(0, 1, (300, 6))
    a = lamda(data)
    b = lamda(data.copy())
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_allclose(a[2], b[2])


def test_lamda_rejects_unscaled_data():
    with pytest.raises(ValueError):
        lamda(np.array([[0.5, 2.0]]))


def test_scalings_are_in_unit_interval():
    raw = np.random.default_rng(6).normal(size=(50, 4)) * [1, 10, 100, 0]
    for m in ("minmax", "rank"):
        s = scale(raw, m)
        assert s.min() >= 0 and s.max() <= 1
    r = scale(raw, "rank")
    np.testing.assert_allclose(np.median(r[:, :3], 0), 0.5, atol=0.02)
    with pytest.raises(ValueError):
        scale(raw, "zscore")


# ---------------------------------------------------------------- speckle null
def test_phase_randomise_keeps_the_magnitude_spectrum():
    x = np.random.default_rng(7).standard_normal(4801)
    y = speckle_null.phase_randomise(x, np.random.default_rng(8))
    np.testing.assert_allclose(np.abs(np.fft.rfft(y)), np.abs(np.fft.rfft(x)), rtol=1e-8, atol=1e-8)
    assert not np.allclose(x, y)


def test_cutoff_table():
    comp = pd.DataFrame({"kind": ["real"] * 4 + ["null_paper"] * 2 + ["null_same_thr"],
                         "n_cells": [5, 60, 80, 200, 10, 70, 55]})
    t = speckle_null.cutoff_table(comp, n_pieces=2, cuts=[0, 50, 100]).set_index("min_cells")
    assert t.loc[0, "real_per_piece"] == 2.0
    assert t.loc[50, "noise_share_paper"] == pytest.approx(0.5 / 1.5)
    assert t.loc[100, "real_kept_frac"] == 0.25
