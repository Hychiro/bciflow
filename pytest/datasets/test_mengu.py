import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from bciflow.datasets.mengu import mengu


# ==========================================================
# Test Suite: mengu
# ==========================================================
#
# Cobertura:
#   1. Validação de tipos
#   2. Validação de ranges e valores permitidos
#   3. Normalização automática do path
#   4. Mock do carregamento HDF5
#   5. Filtragem de sessões
#   6. Filtragem de labels
#   7. Filtragem de depth
#   8. Construção correta de X e y
#   9. Estrutura e integridade do dicionário retornado
#
# ==========================================================


class TestMenGu:


    # ======================================================
    # SECTION 1 - Parameter Validation Tests
    # ======================================================

    def test_invalid_subject_type(self):
        with pytest.raises(ValueError):
            mengu(subject="1")

    def test_invalid_subject_range(self):
        with pytest.raises(ValueError):
            mengu(subject=31)

    def test_invalid_session_type(self):
        with pytest.raises(ValueError):
            mengu(session_list="s01")

    def test_invalid_session_value(self):
        with pytest.raises(ValueError):
            mengu(session_list=["invalid"])

    def test_invalid_labels_type(self):
        with pytest.raises(ValueError):
            mengu(labels="f01")

    def test_invalid_label_value(self):
        with pytest.raises(ValueError):
            mengu(labels=["invalid"])

    def test_invalid_depth_type(self):
        with pytest.raises(ValueError):
            mengu(depth="low")

    def test_invalid_depth_value(self):
        with pytest.raises(ValueError):
            mengu(depth=["invalid"])

    def test_invalid_path_type(self):
        with pytest.raises(ValueError):
            mengu(path=123)


    # ======================================================
    # SECTION 2 - Full Execution Test (mocked)
    # ======================================================

    @patch("bciflow.datasets.h5py.File")
    def test_full_execution(self, mock_h5py):

        fake_data = np.random.randn(
            12,   # sessions
            60,   # labels
            64,   # channels
            1000, # samples
            2     # depth
        )

        fake_file = MagicMock()
        fake_file.__enter__.return_value = {
            "datas": fake_data
        }

        mock_h5py.return_value = fake_file
        eeg = mengu(
            subject=1,
            session_list=["s01"],
            labels=["f01", "f02"],
            depth=["low"]
        )

        assert eeg["data_type"] == "epochs"
        assert eeg["X"].shape == (
            2,      # epochs
            64,     # channels
            1000    # samples
        )
        assert eeg["y"].shape == (2,)
        assert eeg["sfreq"] == 1000.
        assert eeg["tmin"] == 0.
        assert len(eeg["ch_names"]) == 64

    # ======================================================
    # SECTION 3 - Multi Session Test
    # ======================================================

    @patch("bciflow.datasets.h5py.File")
    def test_multiple_sessions(self, mock_h5py):

        fake_data = np.random.randn(
            12,
            60,
            64,
            1000,
            2
        )

        fake_file = MagicMock()
        fake_file.__enter__.return_value = {
            "datas": fake_data
        }

        mock_h5py.return_value = fake_file
        eeg = mengu(
            subject=1,
            session_list=["s01", "s02"],
            labels=["f01"],
            depth=["low"]
        )

        assert eeg["X"].shape[0] == 2
        assert eeg["y"].shape[0] == 2

    # ======================================================
    # SECTION 4 - Multiple Depths Test
    # ======================================================

    @patch("bciflow.datasets.h5py.File")
    def test_multiple_depths(self, mock_h5py):

        fake_data = np.random.randn(
            12,
            60,
            64,
            1000,
            2
        )

        fake_file = MagicMock()
        fake_file.__enter__.return_value = {
            "datas": fake_data
        }
        mock_h5py.return_value = fake_file

        eeg = mengu(
            subject=1,
            session_list=["s01"],
            labels=["f01"],
            depth=["low", "high"]
        )

        assert eeg["X"].shape[0] == 2
        assert eeg["y"].shape[0] == 2


    # ======================================================
    # SECTION 5 - Label Mapping Test
    # ======================================================

    @patch("bciflow.datasets.h5py.File")
    def test_y_dict(self, mock_h5py):

        fake_data = np.random.randn(
            12,
            60,
            64,
            1000,
            2
        )
        fake_file = MagicMock()
        fake_file.__enter__.return_value = {
            "datas": fake_data
        }
        mock_h5py.return_value = fake_file
        eeg = mengu(
            subject=1,
            session_list=["s01"],
            labels=["f01", "f02"],
            depth=["low"]
        )
        expected_y_dict = {
            "f01": 0,
            "f02": 1
        }

        assert eeg["y_dict"] == expected_y_dict
        assert np.array_equal(
            np.unique(eeg["y"]),
            np.array([0, 1])
        )

    # ======================================================
    # SECTION 6 - Path Handling Test
    # ======================================================

    @patch("bciflow.datasets.h5py.File")
    def test_path_handling(self, mock_h5py):

        fake_data = np.random.randn(
            12,
            60,
            64,
            1000,
            2
        )
        fake_file = MagicMock()
        fake_file.__enter__.return_value = {
            "datas": fake_data
        }
        mock_h5py.return_value = fake_file
        custom_path = "my/custom/path"
        mengu(
            subject=1,
            path=custom_path
        )
        expected_path = "my/custom/path/data_s1_64.mat"
        mock_h5py.assert_called_with(
            expected_path,
            "r"
        )