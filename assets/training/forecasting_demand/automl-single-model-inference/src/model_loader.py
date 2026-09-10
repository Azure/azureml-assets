# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Safely discover and load AutoML forecasting model artifacts."""

import os
import zipfile

import skops.io as skops_io


SKOPS_FILE_POSTFIX = ".skops"
_LEGACY_MODEL_POSTFIXES = (".pkl", ".pickle", ".pt", ".pth")
# Exact inventory from a forecasting model exported by the AutoML runtime.
_TRUSTED_AUTOML_FORECASTING_TYPES = frozenset(
    {
        "azureml.automl.core.featurization.featurizationconfig.FeaturizationConfig",
        "azureml.automl.runtime.featurizer.transformer.timeseries.forecasting_pipeline.AzureMLForecastPipeline",
        "azureml.automl.runtime.featurizer.transformer.timeseries.timeseries_transformer.TimeSeriesPipelineType",
        "azureml.automl.runtime.featurizer.transformer.timeseries.timeseries_transformer.TimeSeriesTransformer",
        "azureml.automl.runtime.shared.model_wrappers.LightGBMRegressor",
        "azureml.automl.runtime.shared.model_wrappers.StandardScalerWrapper",
        "azureml.automl.runtime.shared.problem_info.ProblemInfo",
        "azureml.training.tabular.featurization._engineered_feature_names._FeatureTransformersAsJSONObject",
        "azureml.training.tabular.featurization.timeseries.category_binarizer.CategoryBinarizer",
        "azureml.training.tabular.featurization.timeseries.missingdummies_transformer.MissingDummiesTransformer",
        "azureml.training.tabular.featurization.timeseries.numericalize_transformer.NumericalizeTransformer",
        "azureml.training.tabular.featurization.timeseries.restore_dtypes_transformer.RestoreDtypesTransformer",
        "azureml.training.tabular.featurization.timeseries.short_grain_dropper.ShortGrainDropper",
        "azureml.training.tabular.featurization.timeseries.time_index_featurizer.TimeIndexFeaturizer",
        "azureml.training.tabular.featurization.timeseries.time_series_imputer.TimeSeriesImputer",
        "azureml.training.tabular.featurization.timeseries.unique_target_grain_dropper.UniqueTargetGrainDropper",
        "azureml.training.tabular.models.forecasting_pipeline_wrapper.ForecastingPipelineWrapper",
        "azureml.training.tabular.timeseries._automl_forecast_freq.AutoMLForecastFreq",
        "collections.OrderedDict",
        "lightgbm.basic.Booster",
        "lightgbm.sklearn.LGBMRegressor",
        "numpy.dtype",
        "pandas._libs.tslibs.offsets.BQuarterBegin",
        "pandas._libs.tslibs.offsets.BQuarterEnd",
        "pandas._libs.tslibs.offsets.BYearBegin",
        "pandas._libs.tslibs.offsets.BYearEnd",
        "pandas._libs.tslibs.offsets.BusinessDay",
        "pandas._libs.tslibs.offsets.BusinessHour",
        "pandas._libs.tslibs.offsets.BusinessMonthBegin",
        "pandas._libs.tslibs.offsets.BusinessMonthEnd",
        "pandas._libs.tslibs.offsets.CustomBusinessDay",
        "pandas._libs.tslibs.offsets.CustomBusinessHour",
        "pandas._libs.tslibs.offsets.CustomBusinessMonthBegin",
        "pandas._libs.tslibs.offsets.CustomBusinessMonthEnd",
        "pandas._libs.tslibs.offsets.DateOffset",
        "pandas._libs.tslibs.offsets.Day",
        "pandas._libs.tslibs.offsets.Easter",
        "pandas._libs.tslibs.offsets.FY5253",
        "pandas._libs.tslibs.offsets.FY5253Quarter",
        "pandas._libs.tslibs.offsets.Hour",
        "pandas._libs.tslibs.offsets.LastWeekOfMonth",
        "pandas._libs.tslibs.offsets.Micro",
        "pandas._libs.tslibs.offsets.Milli",
        "pandas._libs.tslibs.offsets.Minute",
        "pandas._libs.tslibs.offsets.MonthBegin",
        "pandas._libs.tslibs.offsets.MonthEnd",
        "pandas._libs.tslibs.offsets.Nano",
        "pandas._libs.tslibs.offsets.QuarterBegin",
        "pandas._libs.tslibs.offsets.QuarterEnd",
        "pandas._libs.tslibs.offsets.Second",
        "pandas._libs.tslibs.offsets.SemiMonthBegin",
        "pandas._libs.tslibs.offsets.SemiMonthEnd",
        "pandas._libs.tslibs.offsets.Week",
        "pandas._libs.tslibs.offsets.WeekOfMonth",
        "pandas._libs.tslibs.offsets.YearBegin",
        "pandas._libs.tslibs.offsets.YearEnd",
        "pandas._libs.tslibs.timestamps.Timestamp",
        "pandas.core.frame.DataFrame",
        "pandas.core.indexes.base.Index",
        "pandas.core.indexes.datetimes.DatetimeIndex",
        "pandas.core.indexes.multi.MultiIndex",
        "pandas.core.indexes.range.RangeIndex",
        "pandas.core.series.Series",
        "sklearn.preprocessing._data.StandardScaler",
    }
)


def find_model(model_path):
    """Return the single supported model artifact from a model directory."""
    if not os.path.isdir(model_path) or os.path.islink(model_path):
        raise ValueError("The model path must be a regular directory.")

    safe_models = []
    legacy_models = []
    for root, _, files in os.walk(model_path):
        for filename in sorted(files):
            model_full_path = os.path.join(root, filename)
            postfix = os.path.splitext(filename)[1].lower()
            if postfix == SKOPS_FILE_POSTFIX:
                safe_models.append(model_full_path)
            elif postfix in _LEGACY_MODEL_POSTFIXES:
                legacy_models.append(model_full_path)

    if not safe_models and legacy_models:
        raise ValueError(
            "Unsafe legacy model artifacts are not supported by this component: "
            f"{', '.join(legacy_models)}. Export the fitted forecasting model "
            "directly to .skops without loading or converting an untrusted pickle."
        )

    if len(safe_models) > 1:
        raise ValueError(
            "Expected exactly one .skops model artifact, but found "
            f"{len(safe_models)}: {', '.join(safe_models)}."
        )

    if safe_models:
        model_full_path = safe_models[0]
        if os.path.islink(model_full_path):
            raise ValueError("Symbolic links are not accepted as model artifacts.")
        return model_full_path

    raise ValueError(
        f"Unable to find a supported safe model in folder {model_path}. "
        f"Supported format: {SKOPS_FILE_POSTFIX}."
    )


def _load_skops_model(model_full_path):
    if not os.path.isfile(model_full_path) or os.path.islink(model_full_path):
        raise ValueError("The model artifact must be a regular file.")
    if not zipfile.is_zipfile(model_full_path):
        raise ValueError("The .skops model artifact is not a valid ZIP archive.")

    untrusted_types = set(skops_io.get_untrusted_types(file=model_full_path))
    unsupported_types = untrusted_types - _TRUSTED_AUTOML_FORECASTING_TYPES
    if unsupported_types:
        raise ValueError(
            "The skops model contains types that are not explicitly trusted: "
            f"{', '.join(sorted(unsupported_types))}."
        )

    trusted_types = untrusted_types & _TRUSTED_AUTOML_FORECASTING_TYPES
    return skops_io.load(model_full_path, trusted=sorted(trusted_types))


def get_model(model_full_path):
    """Load a model without allowing arbitrary Python object construction."""
    model_postfix = os.path.splitext(model_full_path)[1].lower()
    print(f"Loading the model from path: {model_full_path}")

    if model_postfix == SKOPS_FILE_POSTFIX:
        fitted_model = _load_skops_model(model_full_path)
    else:
        raise ValueError(
            f"Unsupported model format '{model_postfix}'. "
            f"Supported format: {SKOPS_FILE_POSTFIX}."
        )

    print("Model loading succeeded.")
    return fitted_model
