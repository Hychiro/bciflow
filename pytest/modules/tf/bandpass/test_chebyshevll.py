import numpy as np
import pytest
from scipy.signal import welch

from bciflow.modules.tf.bandpass.chebyshevII import chebyshevII


# ==========================================================
# Test Suite: chebyshevII
# ==========================================================
#
# Coverage:
#   1. Input validation (dict, required keys)
#   2. Signal validation (array, dimensions, samples)
#   3. sfreq validation
#   4. Cutoff frequencies validation
#   5. Filter parameters validation (btype, order, rs)
#   6. Filter functionality and numerical stability
#   7. Shape preservation (single, multi, high-dimensional)
#   8. Internal processing (reshape + filtering loop)
#   9. Filter types (bandpass, lowpass, highpass, bandstop)
#   10. Automatic rs configuration
#   11. inplace behavior (copy vs overwrite)
#
# ==========================================================

class TestChebyshevII:

    # ==========================================================
    # Fixtures
    # ==========================================================

    @pytest.fixture
    def dummy_eeg(self):
        sfreq = 256
        t = np.arange(0, 2, 1 / sfreq)

        signal = (
            np.sin(2 * np.pi * 10 * t) +   # dentro da banda
            np.sin(2 * np.pi * 20 * t)     # fora da banda
        )

        return {
            "X": signal[np.newaxis, :],   
            "sfreq": sfreq
        }

    @pytest.fixture
    def multi_eeg(self):
        sfreq = 256
        t = np.arange(0, 2, 1 / sfreq)

        signal = np.sin(2 * np.pi * 10 * t)

        data = np.stack([
            signal,
            signal,
            signal
        ])

        return {
            "X": data,
            "sfreq": sfreq
        }

    # ==========================================================
    # SECTION 1 — eegdata validation
    # ==========================================================

    def test_eegdata_not_dict(self):
        with pytest.raises(ValueError):
            chebyshevII(eegdata="not a dict")

    def test_eegdata_missing_X(self, dummy_eeg):
        eegdata = dummy_eeg.copy()
        del eegdata['X']

        with pytest.raises(ValueError):
            chebyshevII(eegdata=eegdata)

    def test_eegdata_missing_sfreq(self, dummy_eeg):
        eegdata = dummy_eeg.copy()
        del eegdata['sfreq']

        with pytest.raises(ValueError):
            chebyshevII(eegdata=eegdata)

    # ==========================================================
    # SECTION 2 — X validation
    # ==========================================================

    def test_X_not_array(self, dummy_eeg):
        eegdata = dummy_eeg.copy()
        eegdata['X'] = "not an array"

        with pytest.raises(ValueError):
            chebyshevII(eegdata=eegdata)

    def test_X_not_2d(self, dummy_eeg):
        eegdata = dummy_eeg.copy()
        eegdata['X'] = np.array([1, 2, 3])

        with pytest.raises(ValueError):
            chebyshevII(eegdata=eegdata)

    def test_X_single_sample(self, dummy_eeg):
        eegdata = dummy_eeg.copy()
        eegdata['X'] = np.array([[1], [2], [3]])

        with pytest.raises(ValueError):
            chebyshevII(eegdata=eegdata)

    # ==========================================================
    # SECTION 3 — sfreq validation
    # ==========================================================

    def test_sfreq_not_numeric(self, dummy_eeg):
        eegdata = dummy_eeg.copy()
        eegdata['sfreq'] = "invalid"

        with pytest.raises(ValueError):
            chebyshevII(eegdata=eegdata)

    def test_sfreq_not_positive(self, dummy_eeg):
        eegdata = dummy_eeg.copy()
        eegdata['sfreq'] = -1

        with pytest.raises(ValueError):
            chebyshevII(eegdata=eegdata)

    # ==========================================================
    # SECTION 4 — cutoff frequencies validation
    # ==========================================================

    def test_low_cut_not_numeric(self, dummy_eeg):
        with pytest.raises(ValueError):
            chebyshevII(
                eegdata=dummy_eeg,
                low_cut="A",
                high_cut=30
            )

    def test_high_cut_not_numeric(self, dummy_eeg):
        with pytest.raises(ValueError):
            chebyshevII(
                eegdata=dummy_eeg,
                low_cut=10,
                high_cut="A"
            )

    def test_cutoff_not_positive(self, dummy_eeg):
        with pytest.raises(ValueError):
            chebyshevII(
                eegdata=dummy_eeg,
                low_cut=-1,
                high_cut=30
            )

    def test_high_cut_smaller_than_low_cut(self, dummy_eeg):
        with pytest.raises(ValueError):
            chebyshevII(
                eegdata=dummy_eeg,
                low_cut=30,
                high_cut=10
            )

    def test_high_cut_larger_than_nyquist(self, dummy_eeg):
        with pytest.raises(ValueError):
            chebyshevII(
                eegdata=dummy_eeg,
                low_cut=10,
                high_cut=2000
            )

    # ==========================================================
    # SECTION 5 — filter type, order and rs validation
    # ==========================================================

    def test_invalid_btype(self, dummy_eeg):
        with pytest.raises(ValueError):
            chebyshevII(
                eegdata=dummy_eeg,
                low_cut=10,
                high_cut=30,
                btype="invalid"
            )

    def test_invalid_order(self, dummy_eeg):
        with pytest.raises(ValueError):
            chebyshevII(
                eegdata=dummy_eeg,
                low_cut=10,
                high_cut=30,
                order="invalid"
            )

    def test_order_not_positive(self, dummy_eeg):   
        with pytest.raises(ValueError):
            chebyshevII(
                eegdata=dummy_eeg,
                low_cut=10,
                high_cut=30,
                order=0
            )

    def test_rs_not_numeric(self, dummy_eeg):
        with pytest.raises(ValueError):
            chebyshevII(
                eegdata=dummy_eeg,
                low_cut=10,
                high_cut=30,
                rs="invalid"
            )

    def test_rs_not_positive(self, dummy_eeg):
        with pytest.raises(ValueError):
            chebyshevII(
                eegdata=dummy_eeg,
                low_cut=10,
                high_cut=30,
                rs=-1
            )

    # ==========================================================
    # SECTION 6 — functional tests
    # ==========================================================

    def test_rs_auto_bandpass(self, dummy_eeg):
        out = chebyshevII(
            eegdata=dummy_eeg,
            rs='auto'
        )

        assert 'X' in out

    def test_rs_auto_lowpass(self, dummy_eeg):
        out = chebyshevII(
            eegdata=dummy_eeg,
            low_cut=1,
            high_cut=20,
            btype='lowpass',
            rs='auto'
        )

        assert out['X'].shape == dummy_eeg['X'].shape

    def test_multichannel(self, multi_eeg):
        out = chebyshevII(
            eegdata=multi_eeg,
            low_cut=8,
            high_cut=12
        )

        assert out['X'].shape == multi_eeg['X'].shape

    def test_valid_order(self, dummy_eeg):
        out = chebyshevII(
            eegdata=dummy_eeg,
            low_cut=8,
            high_cut=12,
            order=6
        )

        assert out['X'].shape == dummy_eeg['X'].shape

    def test_signal_changes(self, dummy_eeg):

        before = dummy_eeg['X'][0].copy()

        out = chebyshevII(
            eegdata=dummy_eeg,
            low_cut=5,
            high_cut=8
        )

        after = out['X'][0]

        f_before, p_before = welch(
            before,
            fs=dummy_eeg['sfreq'],
            nperseg=256
        )

        f_after, p_after = welch(
            after,
            fs=dummy_eeg['sfreq'],
            nperseg=256
        )

        band = (f_before >= 15) & (f_before <= 40)

        assert np.mean(p_after[band]) < np.mean(p_before[band])

    # ==========================================================
    # SECTION 7 — btype tests
    # ==========================================================

    def test_lowpass_execution(self, dummy_eeg):
        out = chebyshevII(
            eegdata=dummy_eeg,
            low_cut=1,
            high_cut=20,
            btype='lowpass'
        )

        assert out['X'].shape == dummy_eeg['X'].shape

    def test_highpass_execution(self, dummy_eeg):
        out = chebyshevII(
            eegdata=dummy_eeg,
            low_cut=5,
            high_cut=40,
            btype='highpass'
        )

        assert out['X'].shape == dummy_eeg['X'].shape

    def test_bandstop_execution(self, dummy_eeg):
        out = chebyshevII(
            eegdata=dummy_eeg,
            low_cut=8,
            high_cut=12,
            btype='bandstop'
        )

        assert out['X'].shape == dummy_eeg['X'].shape

    # ==========================================================
    # SECTION 8 — inplace behavior
    # ==========================================================

    def test_inplace_false_copy(self, dummy_eeg):
        out = chebyshevII(
            eegdata=dummy_eeg,
            low_cut=8,
            high_cut=12,
            inplace=False
        )

        assert out is not dummy_eeg

    def test_inplace_true(self, dummy_eeg):   
        out = chebyshevII(
            eegdata=dummy_eeg,
            low_cut=8,
            high_cut=12,
            inplace=True
        )

        assert out is dummy_eeg

    # ==========================================================
    # SECTION 9 — reshape/high-dimensional execution
    # ==========================================================

    def test_high_dimensional_input(self): 
        sfreq = 256

        data = np.random.randn(2, 3, 4, 512)

        eegdata = {
            "X": data,
            "sfreq": sfreq
        }

        out = chebyshevII(
            eegdata=eegdata,
            low_cut=8,
            high_cut=12
        )

        assert out['X'].shape == data.shape

    # ==========================================================
    # SECTION 10 — full execution
    # ==========================================================

    def test_full_execution_test(self, dummy_eeg):

        signal_before = dummy_eeg['X'].copy()

        filtered_data = chebyshevII(
            eegdata=dummy_eeg,
            low_cut=8,
            high_cut=12,
            order=4,
            rs=40
        )

        signal_after = filtered_data['X']

        assert filtered_data['X'].shape == dummy_eeg['X'].shape

        assert not np.allclose(signal_before, signal_after)

        assert np.isfinite(signal_after).all()