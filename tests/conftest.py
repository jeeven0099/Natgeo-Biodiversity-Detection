import matplotlib
import pytest

from biodetect import synthetic

matplotlib.use("Agg")


@pytest.fixture(scope="session")
def soundscape():
    """60 s synthetic soundscape at 48 kHz and its event table."""
    return synthetic.soundscape(60.0, 48000, seed=1)


@pytest.fixture(scope="session")
def wav_dir(tmp_path_factory):
    """Two short synthetic recordings on disk."""
    d = tmp_path_factory.mktemp("wavs")
    synthetic.write(d, n_files=2, duration_s=70.0, seed=3)
    return d
