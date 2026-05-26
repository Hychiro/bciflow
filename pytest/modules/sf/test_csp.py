import numpy as np
import pytest

from bciflow.modules.sf.csp import csp


# ==========================================================
# Test Suite: CSP
# ==========================================================
#
# Coverage:
#   1. Constructor validation (m_pairs)
#   2. eegdata validation
#   3. X validation
#   4. y validation
#   5. Class validation
#   6. Electrode and CSP parameter validation
#   7. Fit execution
#   8. Transform execution
#   9. fit_transform execution
#   10. Shape preservation
#   11. Internal learned parameters
#   12. Numerical stability
#   13. 3D/4D transform behavior
#
# ==========================================================


class TestCSP:

    # ==========================================================
    # Fixtures
    # ==========================================================

    @pytest.fixture
    def dummy_eeg(self):

        np.random.seed(42)

        n_trials = 20
        n_bands = 3
        n_electrodes = 8
        n_samples = 256

        X = np.random.randn(
            n_trials,
            n_bands,
            n_electrodes,
            n_samples
        )

        y = np.array([0] * 10 + [1] * 10)

        return {
            'X': X,
            'y': y
        }

    @pytest.fixture
    def single_band_eeg(self):

        np.random.seed(42)
        X = np.random.randn(10, 8, 256)
        return {
            'X': X
        }

    # ==========================================================
    # SECTION 1 — constructor validation
    # ==========================================================

    def test_invalid_m_pairs_type(self):
        with pytest.raises(ValueError):
            csp(m_pairs='invalid')

    def test_invalid_m_pairs_negative(self):
        with pytest.raises(ValueError):
            csp(m_pairs=-1)

    def test_invalid_m_pairs_zero(self):
        with pytest.raises(ValueError):
            csp(m_pairs=0)

    # ==========================================================
    # SECTION 2 — eegdata validation
    # ==========================================================

    def test_fit_eegdata_not_dict(self):
        model = csp()
        with pytest.raises(ValueError):
            model.fit("invalid")

    def test_fit_missing_X(self, dummy_eeg):
        model = csp()
        eegdata = dummy_eeg.copy()
        del eegdata['X']
        with pytest.raises(ValueError):
            model.fit(eegdata)

    def test_fit_missing_y(self, dummy_eeg):
        model = csp()
        eegdata = dummy_eeg.copy()
        del eegdata['y']
        with pytest.raises(ValueError):
            model.fit(eegdata)

    # ==========================================================
    # SECTION 3 — X validation
    # ==========================================================

    def test_X_not_array(self, dummy_eeg):
        model = csp()
        eegdata = dummy_eeg.copy()
        eegdata['X'] = 'invalid'
        with pytest.raises(ValueError):
            model.fit(eegdata)

    def test_X_wrong_dimensions(self, dummy_eeg):
        model = csp()
        eegdata = dummy_eeg.copy()
        eegdata['X'] = np.random.randn(10, 10)
        with pytest.raises(ValueError):
            model.fit(eegdata)

    def test_X_single_trial(self, dummy_eeg):
        model = csp()
        eegdata = dummy_eeg.copy()
        eegdata['X'] = np.random.randn(1, 2, 8, 256)
        with pytest.raises(ValueError):
            model.fit(eegdata)

    def test_X_single_sample(self, dummy_eeg):
        model = csp()
        eegdata = dummy_eeg.copy()
        eegdata['X'] = np.random.randn(10, 2, 8, 1)
        with pytest.raises(ValueError):
            model.fit(eegdata)

    def test_X_nan_values(self, dummy_eeg):
        model = csp()
        eegdata = dummy_eeg.copy()
        eegdata['X'][0, 0, 0, 0] = np.nan
        with pytest.raises(ValueError):
            model.fit(eegdata)

    def test_X_inf_values(self, dummy_eeg):
        model = csp()
        eegdata = dummy_eeg.copy()
        eegdata['X'][0, 0, 0, 0] = np.inf
        with pytest.raises(ValueError):
            model.fit(eegdata)

    # ==========================================================
    # SECTION 4 — y validation
    # ==========================================================

    def test_y_not_array(self, dummy_eeg):
        model = csp()
        eegdata = dummy_eeg.copy()
        eegdata['y'] = 'invalid'
        with pytest.raises(ValueError):
            model.fit(eegdata)

    def test_y_not_1d(self, dummy_eeg):
        model = csp()
        eegdata = dummy_eeg.copy()
        eegdata['y'] = np.array([[0, 1]])
        with pytest.raises(ValueError):
            model.fit(eegdata)

    def test_y_wrong_size(self, dummy_eeg):
        model = csp()
        eegdata = dummy_eeg.copy()
        eegdata['y'] = np.array([0, 1])
        with pytest.raises(ValueError):
            model.fit(eegdata)

    def test_y_nan(self, dummy_eeg):
        model = csp()
        eegdata = dummy_eeg.copy()
        eegdata['y'] = eegdata['y'].astype(float)
        eegdata['y'][0] = np.nan
        with pytest.raises(ValueError):
            model.fit(eegdata)

    # ==========================================================
    # SECTION 5 — class validation
    # ==========================================================

    def test_single_class(self, dummy_eeg):
        model = csp()
        eegdata = dummy_eeg.copy()
        eegdata['y'] = np.zeros(20)
        with pytest.raises(ValueError):
            model.fit(eegdata)

    def test_more_than_two_classes(self, dummy_eeg):
        model = csp()
        eegdata = dummy_eeg.copy()
        eegdata['y'] = np.array(
            [0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
             1, 2, 0, 1, 2, 0, 1, 2, 0, 1]
        )
        with pytest.raises(ValueError):
            model.fit(eegdata)

    def test_empty_class_branch(self, monkeypatch):
        model = csp()
        def fake_unique(y):
            return np.array([0, 1])
        monkeypatch.setattr(np, "unique", fake_unique)
        eegdata = {
            'X': np.random.randn(10, 2, 8, 256),
            'y': np.zeros(10)
        }
        with pytest.raises(
            ValueError,
            match="Both classes must contain at least one trial."
        ):
            model.fit(eegdata)

    # ==========================================================
    # SECTION 6 — electrode validation
    # ==========================================================

    def test_single_electrode(self):
        model = csp()
        eegdata = {
            'X': np.random.randn(10, 2, 1, 256),
            'y': np.array([0] * 5 + [1] * 5)
        }
        with pytest.raises(ValueError):
            model.fit(eegdata)

    def test_invalid_m_pairs_vs_electrodes(self):
        model = csp(m_pairs=5)
        eegdata = {
            'X': np.random.randn(10, 2, 8, 256),
            'y': np.array([0] * 5 + [1] * 5)
        }
        with pytest.raises(ValueError):
            model.fit(eegdata)

    # ==========================================================
    # SECTION 7 — fit execution
    # ==========================================================

    def test_fit_execution(self, dummy_eeg):
        model = csp(m_pairs=2)
        out = model.fit(dummy_eeg)
        assert out is model

    def test_fit_creates_W(self, dummy_eeg):
        model = csp(m_pairs=2)
        model.fit(dummy_eeg)
        assert model.W is not None

    def test_W_shape(self, dummy_eeg):
        model = csp(m_pairs=2)
        model.fit(dummy_eeg)
        assert model.W.shape == (3, 8, 4)

    def test_bands_attribute(self, dummy_eeg):
        model = csp()
        model.fit(dummy_eeg)
        assert model.bands == 3

    def test_n_electrodes_attribute(self, dummy_eeg):
        model = csp()
        model.fit(dummy_eeg)
        assert model.n_electrodes == 8

    # ==========================================================
    # SECTION 8 — transform validation
    # ==========================================================

    def test_transform_before_fit(self, single_band_eeg):
        model = csp()
        with pytest.raises(ValueError):
            model.transform(single_band_eeg)

    def test_transform_invalid_input(self):
        model = csp()
        with pytest.raises(ValueError):
            model.transform("invalid")

    def test_transform_missing_X(self):
        model = csp()
        with pytest.raises(ValueError):
            model.transform({})

    def test_transform_X_not_array(self):
        model = csp()
        with pytest.raises(ValueError):
            model.transform({'X': 'invalid'})

    def test_transform_invalid_dimensions(self):
        model = csp()
        with pytest.raises(ValueError):
            model.transform({'X': np.array([1, 2, 3])})
    
    def test_transform_empty_trials(self):
        model = csp()
        model.W = np.random.randn(1, 8, 4)
        model.bands = 1
        model.n_electrodes = 8
        eegdata = {
            'X': np.empty((0, 8, 256))
        }
        with pytest.raises(
            ValueError,
            match="X must contain at least one trial."
        ):
           model.transform(eegdata)

    def test_transform_single_sample(self):
        model = csp()
        model.W = np.random.randn(1, 8, 4)
        model.bands = 1
        model.n_electrodes = 8
        eegdata = {
            'X': np.random.randn(5, 8, 1)
        }
        with pytest.raises(
            ValueError,
            match="Each trial must contain more than one sample."
        ):
            model.transform(eegdata)
    
    def test_transform_nan_values(self):
        model = csp()
        model.W = np.random.randn(1, 8, 4)
        model.bands = 1
        model.n_electrodes = 8
        eegdata = {
            'X': np.random.randn(5, 8, 256)
        }
        eegdata['X'][0, 0, 0] = np.nan
        with pytest.raises(
            ValueError,
            match="X contains NaN or infinite values."
        ):
            model.transform(eegdata)

    def test_transform_uninitialized_object(self):
        model = csp()
        model.W = np.random.randn(1, 8, 4)
        eegdata = {
            'X': np.random.randn(5, 8, 256)
        }
        with pytest.raises(
            ValueError,
            match="CSP object is not properly initialized. Call fit\\(\\) first."
        ):
            model.transform(eegdata)    

    def test_transform_wrong_number_of_bands(self):
        model = csp()
        model.W = np.random.randn(2, 8, 4)
        model.bands = 2
        model.n_electrodes = 8
        eegdata = {
            'X': np.random.randn(5, 3, 8, 256)
        }
        with pytest.raises(
            ValueError,
            match="The number of bands in the input data is different from the number of bands in the fitted data."
        ):
            model.transform(eegdata)

    def test_transform_wrong_number_of_electrodes(self):
        model = csp()
        model.W = np.random.randn(2, 8, 4)
        model.bands = 2
        model.n_electrodes = 8
        eegdata = {
            'X': np.random.randn(5, 2, 10, 256)
        }
        with pytest.raises(
            ValueError,
            match="The number of electrodes in the input data does not match the fitted CSP model."
        ):
            model.transform(eegdata)    

    # ==========================================================
    # SECTION 9 — transform execution
    # ==========================================================

    def test_transform_execution_4d(self, dummy_eeg):
        model = csp(m_pairs=2)
        model.fit(dummy_eeg)
        out = model.transform(dummy_eeg)
        assert 'X' in out

    def test_transform_shape_4d(self, dummy_eeg):
        model = csp(m_pairs=2)
        model.fit(dummy_eeg)
        out = model.transform(dummy_eeg)
        assert out['X'].shape == (20, 3, 4, 256)

    def test_transform_execution_3d(self):
        np.random.seed(42)
        train_X = np.random.randn(20, 1, 8, 256)
        train_y = np.array([0] * 10 + [1] * 10)
        model = csp(m_pairs=2)
        model.fit({
            'X': train_X,
            'y': train_y
        })
        test_X = np.random.randn(5, 8, 256)
        out = model.transform({
            'X': test_X
        })
        assert out['X'].shape == (5, 1, 4, 256)

    # ==========================================================
    # SECTION 10 — transform consistency
    # ==========================================================

    def test_transform_changes_signal(self, dummy_eeg):
        model = csp(m_pairs=2)
        model.fit(dummy_eeg)
        before = dummy_eeg['X'].copy()
        out = model.transform(dummy_eeg)
        after = out['X']
        assert not np.allclose(before[:, :, :4], after)

    def test_transform_finite_values(self, dummy_eeg):
        model = csp(m_pairs=2)
        model.fit(dummy_eeg)
        out = model.transform(dummy_eeg)
        assert np.isfinite(out['X']).all()

    # ==========================================================
    # SECTION 11 — fit_transform
    # ==========================================================

    def test_fit_transform_execution(self, dummy_eeg):
        model = csp(m_pairs=2)
        out = model.fit_transform(dummy_eeg)
        assert 'X' in out

    def test_fit_transform_shape(self, dummy_eeg):
        model = csp(m_pairs=2)
        out = model.fit_transform(dummy_eeg)
        assert out['X'].shape == (20, 3, 4, 256)

    # ==========================================================
    # SECTION 12 — numerical stability
    # ==========================================================

    def test_singular_covariance_execution(self):
        X = np.ones((10, 2, 8, 256))
        y = np.array([0] * 5 + [1] * 5)
        model = csp(m_pairs=2)
        out = model.fit({
            'X': X,
            'y': y
        })
        assert out.W is not None

    # ==========================================================
    # SECTION 13 — identity fallback
    # ==========================================================

    def test_identity_fallback(self, monkeypatch, dummy_eeg):
        def mock_eigh(*args, **kwargs):
            raise Exception("Forced failure")
        monkeypatch.setattr(
            "scipy.linalg.eigh",
            mock_eigh
        )
        model = csp(m_pairs=2)
        model.fit(dummy_eeg)
        assert model.W is not None
        assert np.isfinite(model.W).all()