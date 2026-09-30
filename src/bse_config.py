"""Frozen behavioral-statistics expert configuration for the release protocol."""

from __future__ import annotations

from typing import Mapping


BSE_ESTIMATOR_CLASS = "sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier"
BSE_CONFIGURATION_SOURCE = "estimator.get_params(deep=False)"


def release_bse_parameters(seed: int) -> dict:
    """Return every constructor parameter used by the release environment."""
    return {
        "loss": "log_loss",
        "learning_rate": 0.1,
        "max_iter": 300,
        "max_leaf_nodes": 15,
        "max_depth": None,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "max_features": 1.0,
        "max_bins": 255,
        "categorical_features": "from_dtype",
        "monotonic_cst": None,
        "interaction_cst": None,
        "warm_start": False,
        "early_stopping": "auto",
        "scoring": "loss",
        "validation_fraction": 0.1,
        "n_iter_no_change": 10,
        "tol": 1e-7,
        "verbose": 0,
        "random_state": int(seed),
        "class_weight": None,
    }


def estimator_configuration(estimator, sklearn_version: str) -> dict:
    """Serialize the estimator's complete shallow parameter mapping."""
    estimator_class = f"{type(estimator).__module__}.{type(estimator).__qualname__}"
    parameters = dict(estimator.get_params(deep=False))
    return {
        "estimator": estimator_class,
        "sklearn_version": str(sklearn_version),
        "parameter_source": BSE_CONFIGURATION_SOURCE,
        "parameters": parameters,
    }


def expected_configuration(seed: int, sklearn_version: str = "1.7.2") -> dict:
    """Build a complete configuration record for tests and static validation."""
    return {
        "estimator": BSE_ESTIMATOR_CLASS,
        "sklearn_version": str(sklearn_version),
        "parameter_source": BSE_CONFIGURATION_SOURCE,
        "parameters": release_bse_parameters(seed),
    }


def configuration_mismatches(configuration: Mapping[str, object], seed: int) -> dict:
    """Return release-contract mismatches in a serialized BSE configuration."""
    mismatches = {}
    if configuration.get("estimator") != BSE_ESTIMATOR_CLASS:
        mismatches["estimator"] = {
            "actual": configuration.get("estimator"),
            "expected": BSE_ESTIMATOR_CLASS,
        }
    if configuration.get("parameter_source") != BSE_CONFIGURATION_SOURCE:
        mismatches["parameter_source"] = {
            "actual": configuration.get("parameter_source"),
            "expected": BSE_CONFIGURATION_SOURCE,
        }
    if not str(configuration.get("sklearn_version", "")).strip():
        mismatches["sklearn_version"] = {
            "actual": configuration.get("sklearn_version"),
            "expected": "non-empty version string",
        }

    actual_parameters = configuration.get("parameters")
    expected_parameters = release_bse_parameters(seed)
    if not isinstance(actual_parameters, Mapping):
        mismatches["parameters"] = {
            "actual": actual_parameters,
            "expected": expected_parameters,
        }
        return mismatches

    actual_keys = set(actual_parameters)
    expected_keys = set(expected_parameters)
    if actual_keys != expected_keys:
        mismatches["parameter_keys"] = {
            "missing": sorted(expected_keys - actual_keys),
            "unexpected": sorted(actual_keys - expected_keys),
        }
    for key, expected in expected_parameters.items():
        actual = actual_parameters.get(key)
        if actual != expected:
            mismatches[f"parameters.{key}"] = {
                "actual": actual,
                "expected": expected,
            }
    return mismatches
