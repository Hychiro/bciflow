import numpy as np
import pytest

from bciflow.modules.tf.resample_cubic import cubic_resample

# ==========================================================
# Test Suite: cubic_resample
# ==========================================================
#
# Cobertura:
#   1. Validação estrutural de entrada (dict, chaves obrigatórias)
#   2. Validação do sinal X (tipo, dimensionalidade, tamanho)
#   3. Validação de sfreq
#   4. Validação de new_sfreq
#   5. Verificação de comprimento válido após resample
#   6. Comportamento do resample (shape e atualização de sfreq)
#   7. Suporte multicanal
#   8. Processamento interno (reshape + loop por canal)
#   9. Comportamento inplace vs cópia
#   10. Robustez numérica (saída finita e consistente)
#
# ==========================================================

class TestCubicResample:
    # ======================================================
    # 0. fixture
    # ======================================================

    @pytest.fixture
    def eeg(self):
        sfreq = 100
        t = np.arange(0, 2, 1/sfreq)

        signal = np.sin(2 * np.pi * 5 * t)
        signal = signal[np.newaxis, :]  # shape -> (1, samples)

        return {
            "X": signal,
            "sfreq": sfreq
        }

    # ======================================================
    # 1. eegdata validation
    # ======================================================

    def test_eegdata_not_dict(self):
        with pytest.raises(ValueError):
            cubic_resample("not dict", 50)

    def test_missing_X(self, eeg):
        data = eeg.copy()
        del data["X"]

        with pytest.raises(ValueError):
            cubic_resample(data, 50)

    def test_missing_sfreq(self, eeg):
        data = eeg.copy()
        del data["sfreq"]

        with pytest.raises(ValueError):
            cubic_resample(data, 50)

    # ======================================================
    # 2. X validation
    # ======================================================

    def test_X_not_array(self, eeg):
        data = eeg.copy()
        data["X"] = "invalid"

        with pytest.raises(ValueError):
            cubic_resample(data, 50)

    def test_X_not_2d(self):
        with pytest.raises(ValueError):
            cubic_resample({"X": np.array([1, 2]), "sfreq": 100}, 50)

    def test_X_too_small(self):
        x = np.array([[1]])

        with pytest.raises(ValueError):
            cubic_resample({"X": x, "sfreq": 100}, 50)

    # ======================================================
    # 3. sfreq validation
    # ======================================================

    def test_sfreq_not_numeric(self, eeg):
        data = eeg.copy()
        data["sfreq"] = "A"

        with pytest.raises(ValueError):
            cubic_resample(data, 50)

    def test_sfreq_negative(self, eeg):
        data = eeg.copy()
        data["sfreq"] = -10

        with pytest.raises(ValueError):
            cubic_resample(data, 50)

    # ======================================================
    # 4. new_sfreq validation
    # ======================================================

    def test_new_sfreq_not_numeric(self, eeg):
        with pytest.raises(ValueError):
            cubic_resample(eeg, "A")

    def test_new_sfreq_negative(self, eeg):
        with pytest.raises(ValueError):
            cubic_resample(eeg, -10)

    def test_new_sfreq_greater_than_original(self, eeg):
        with pytest.raises(ValueError):
            cubic_resample(eeg, 200)

    def test_new_sfreq_not_divisible(self, eeg):
        with pytest.raises(ValueError):
            cubic_resample(eeg, 30)

    def test_invalid_length_after_resample(self):
        sfreq = 100
        t = np.arange(0, 0.5, 1/sfreq)

        x = np.sin(2*np.pi*5*t)
        x = x[np.newaxis, :]

        data = {"X": x, "sfreq": sfreq}

        with pytest.raises(ValueError):
            cubic_resample(data, 40)

    def test_invalid_length_after_resample(self):
        sfreq = 100
        new_sfreq = 50

        x = np.array([[1.0, 2.0]])

        data = {
            "X": x,
            "sfreq": sfreq
        }

        with pytest.raises(ValueError):
            cubic_resample(data, new_sfreq)

    # ======================================================
    # 5. functional tests
    # ======================================================

    def test_resample_shape(self, eeg):
        data = eeg.copy()

        out = cubic_resample(data, 50)

        assert out["X"].shape[-1] == eeg["X"].shape[-1] // 2
        assert out["sfreq"] == 50

    def test_multichannel(self):
        sfreq = 100
        t = np.arange(0, 2, 1/sfreq)

        sig = np.sin(2*np.pi*5*t)
        x = np.stack([sig, sig])

        data = {"X": x, "sfreq": sfreq}

        out = cubic_resample(data, 50)

        assert out["X"].shape == (2, x.shape[-1] // 2)

    def test_inplace_false_copy(self, eeg):
        data = eeg.copy()

        out = cubic_resample(data, 50, inplace=False)

        assert out is not data

    def test_inplace_true(self, eeg):
        data = eeg.copy()

        out = cubic_resample(data, 50, inplace=True)

        assert out is data

    def test_signal_changes(self, eeg):
        data = eeg.copy()

        out = cubic_resample(data, 50)

        assert out["X"].shape[-1] < eeg["X"].shape[-1]

    def test_reconstruction_consistency(self, eeg):
        data = eeg.copy()

        out = cubic_resample(data, 50)

        assert np.isfinite(out["X"]).all()
