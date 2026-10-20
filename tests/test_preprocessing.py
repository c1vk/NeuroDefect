from __future__ import annotations

import numpy as np

from src.data.cleaner import clean_frame
from src.data.imbalance import apply_smote, resolve_smote
from src.data.preprocessing import fit_preprocessor
from src.data.splitting import stratified_split
from src.data.validator import validate_target
from src.evaluation.validation import assert_no_test_in_train, scaler_fit_on_train_only


def test_no_index_leakage(tiny_frame):
    features, target, _, _ = clean_frame(tiny_frame, "defects")
    target = validate_target(target).astype(int)
    split = stratified_split(features, target, test_size=0.3, random_state=0)
    assert_no_test_in_train(split.train_index, split.test_index)
    assert len(set(split.train_index) & set(split.test_index)) == 0


def test_scaler_fit_on_training_only(tiny_frame):
    features, target, _, _ = clean_frame(tiny_frame, "defects")
    target = validate_target(target).astype(int)
    split = stratified_split(features, target, test_size=0.3, random_state=0)
    prep = fit_preprocessor(split.X_train)
    train_scaled = prep.transform(split.X_train).to_numpy()
    test_scaled = prep.transform(split.X_test).to_numpy()
    assert scaler_fit_on_train_only(train_scaled, test_scaled)
    assert abs(train_scaled.mean()) < 0.15
    # Test mean is allowed to differ — that is evidence the scaler did not see test data.
    assert train_scaled.shape[1] == test_scaled.shape[1]


def test_smote_policy_skips_cm1_and_jm1_only():
    assert resolve_smote("cm1", True) is False
    assert resolve_smote("CM1_shared", True) is False
    assert resolve_smote("jm1", True) is False
    assert resolve_smote("pc1", True) is True
    assert resolve_smote("NASA_CM1_sample", True) is True
    assert resolve_smote("cm1", True, force=True) is True
    assert resolve_smote("pc1", False) is False


def test_svm_baseline_is_calibrated():
    from sklearn.calibration import CalibratedClassifierCV

    from src.models.baselines import make_sklearn_baselines

    svm = make_sklearn_baselines(0)["Support Vector Machine"]
    assert isinstance(svm, CalibratedClassifierCV)
    assert svm.ensemble is False
    assert svm.estimator.probability is not True


def test_smote_training_only(tiny_frame):
    features, target, _, _ = clean_frame(tiny_frame, "defects")
    target = validate_target(target).astype(int)
    split = stratified_split(features, target, test_size=0.3, random_state=0)
    prep = fit_preprocessor(split.X_train)
    X_train = prep.transform(split.X_train)
    X_test_before = prep.transform(split.X_test).to_numpy().copy()
    X_res, y_res, report = apply_smote(X_train, split.y_train.astype(int), enabled=True, random_state=0)
    X_test_after = prep.transform(split.X_test).to_numpy()
    assert np.allclose(X_test_before, X_test_after)
    assert report.applied
    assert len(X_res) >= len(X_train)
    assert len(split.X_test) == X_test_after.shape[0]
