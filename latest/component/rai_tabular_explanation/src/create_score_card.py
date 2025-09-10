# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json
import logging
import os
import re
from datetime import datetime

import _score_card.classification_components as ClassificationComponents
import _score_card.regression_components as RegressionComponents
import mlflow
from _score_card._rai_insight_data import PdfDataGen, RaiInsightData
from _score_card.common_components import get_full_html, to_pdf
from rai_component_utilities import ensure_shim

ensure_shim()
from azureml.rai.utils.telemetry import LoggerFactory, track  # noqa: E402
from constants import (COMPONENT_NAME, DashboardInfo,  # noqa: E402
                       PropertyKeyValues, RAIToolType)
from rai_component_utilities import add_properties_to_gather_run  # noqa: E402
from rai_component_utilities import load_dashboard_info_file  # noqa: E402

from responsibleai import __version__ as responsibleai_version  # noqa: E402

threshold_reg = re.compile(r"([<>=]{1,2})([0-9.]+)")

DEFAULT_MODULE_NAME = "rai_score_card"
DEFAULT_MODULE_VERSION = "0.0.0"

_logger = logging.getLogger(__file__)
_ai_logger = None
_module_name = DEFAULT_MODULE_NAME
_module_version = DEFAULT_MODULE_VERSION


def _get_logger():
    global _ai_logger
    if _ai_logger is None:
        _ai_logger = LoggerFactory.get_logger(
            __file__, _module_name, _module_version, COMPONENT_NAME)
    return _ai_logger


def get_parser():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # Constructor arguments
    parser.add_argument(
        "--rai_insights_dashboard", type=str, help="name:version", required=True
    )
    parser.add_argument(
        "--pdf_output_path", type=str, help="pdf output path", required=True
    )
    parser.add_argument(
        "--pdf_generation_config", type=str, help="pdf config", required=True
    )
    parser.add_argument(
        "--predefined_cohorts_json", type=str, help="cohorts defintion", required=False
    )
    parser.add_argument("--local", type=bool, help="local run", required=False)
    parser.add_argument(
        "--wkhtml2pdfpath", type=str, help="path to wkhtml2pdf", required=False
    )

    # Component info
    parser.add_argument("--component_name", type=str, required=True)
    parser.add_argument("--component_version", type=str, required=True)

    return parser


def parse_threshold(threshold):
    norm_th = threshold.replace(" ", "")
    match = re.search(threshold_reg, norm_th)

    if not match:
        _logger.warning(
            "Unable to parse argument threshold {}. "
            "Please refer to documentation for acceptable input.".format(threshold)
        )
        return None, None

    target_type, target_arg = match.group(1, 2)

    try:
        target_arg = float(target_arg)
    except Exception as ex:
        _logger.warning(
            "Unable to parse argument threshold {}. "
            "Please refer to documentation for acceptable input.".format(threshold)
        )
        _logger.warning(
            "Exception message included which may be useful for your reference: {}".format(
                ex
            )
        )
    return target_type, target_arg


def add_properties_to_gather_run_score_card(dashboard_info, rai_info):
    """Local wrapper for the common add_properties_to_gather_run function."""
    included_tools = {
        RAIToolType.CAUSAL: False,
        RAIToolType.COUNTERFACTUAL: False,
        RAIToolType.ERROR_ANALYSIS: False,
        RAIToolType.EXPLANATION: False,
        RAIToolType.SCORECARD: True,
    }

    run_properties = {
        PropertyKeyValues.RAI_INSIGHTS_TYPE_KEY: "PdfGeneration",
        PropertyKeyValues.RAI_INSIGHTS_DASHBOARD_ID_KEY: dashboard_info[
            DashboardInfo.RAI_INSIGHTS_RUN_ID_KEY
        ],
        PropertyKeyValues.RAI_INSIGHTS_RESPONSIBLEAI_VERSION_KEY: responsibleai_version,
        PropertyKeyValues.RAI_INSIGHTS_MODEL_ID_KEY: dashboard_info[
            DashboardInfo.RAI_INSIGHTS_MODEL_ID_KEY
        ],
        PropertyKeyValues.RAI_INSIGHTS_SCORE_CARD_TITLE_KEY: rai_info["ScoreCardTitle"],
    }

    # Call the common function
    add_properties_to_gather_run(
        dashboard_info, run_properties, included_tools, _module_name, _module_version
    )


def validate_and_correct_config(config, insight_data):
    i_data = insight_data.get_raiinsight()
    try:
        top_n = config["FeatureImportance"]["top_n"]
        if top_n > 10:
            _logger.warning(
                "Feature importance is limited to "
                "top 10 most important feature"
                f", but top_n={top_n} was specificed."
                "Setting top_n to 10 automatically."
            )
        config["FeatureImportance"]["top_n"] = 10
    except KeyError:
        pass

    if "Fairness" in config.keys():
        fc = config["Fairness"]
        cat_features = [
            f for f in fc["sensitive_features"] if f in i_data.categorical_features
        ]
        dropped_features = [
            f for f in fc["sensitive_features"] if f not in i_data.categorical_features
        ]
        _logger.warning(
            "Non categorical features dropped for fairness assessment: {}".format(dropped_features)
        )
        fc["sensitive_features"] = cat_features
    return config


@track(_get_logger)
def main(args):
    dashboard_info = load_dashboard_info_file(args.rai_insights_dashboard)
    _logger.info("Constructor info: {0}".format(dashboard_info))

    insight_data = RaiInsightData(args.rai_insights_dashboard)
    with open(args.pdf_generation_config, "r") as json_file:
        config = json.load(json_file)

    if args.predefined_cohorts_json:
        with open(args.predefined_cohorts_json, "r") as json_file:
            cohorts_definition = json.load(json_file)
            cohorts_map = {
                c["name"]: c["cohort_filter_list"] for c in cohorts_definition
            }
            config["cohorts_definition"] = cohorts_map

    config = validate_and_correct_config(config, insight_data)

    for k, v in config["Metrics"].items():
        if "threshold" in v.keys():
            tt, ta = parse_threshold(v["threshold"])
            if tt and ta:
                config["Metrics"][k]["threshold"] = (tt, ta)
            else:
                config["Metrics"][k].pop("threshold")

    if not args.local:
        workspace_name = os.environ.get("AZUREML_ARM_WORKSPACE_NAME")
        resource_group = os.environ.get("AZUREML_ARM_RESOURCEGROUP")
        subscription_id = os.environ.get("AZUREML_ARM_SUBSCRIPTION")
        wsid = f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/\
        providers/Microsoft.MachineLearningServices/workspaces/{workspace_name}"
        dashboard_link = "https://ml.azure.com/model/analysis/{}/{}/?wsid={}".format(
            dashboard_info[DashboardInfo.RAI_INSIGHTS_MODEL_ID_KEY],
            dashboard_info[DashboardInfo.RAI_INSIGHTS_GATHER_RUN_ID_KEY],
            wsid,
        )

        # Get UTC from python datetime module if this is not available from run details
        startTimeUtc = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')

        config["runinfo"] = {
            # Leaving blank for now
            "submittedBy": '',
            "startTimeUtc": startTimeUtc,
            "dashboard_link": dashboard_link,
            "model_id": dashboard_info[DashboardInfo.RAI_INSIGHTS_MODEL_ID_KEY],
            "dashboard_title": dashboard_info[
                DashboardInfo.RAI_INSIGHTS_DASHBOARD_TITLE_KEY
            ],
        }

    if config["Model"]["ModelType"].lower() == "regression":
        wf = Workflow(insight_data, config, args, RegressionComponents)
    elif config["Model"]["ModelType"].lower() in ("classification", "multiclass"):
        wf = Workflow(insight_data, config, args, ClassificationComponents)
    else:
        raise ValueError(
            "Model type {} cannot be matched to a score card generation workflow".format(
                config["Model"]["ModelType"]
            )
        )

    wf.generate_pdf()

    if not args.local:
        add_properties_to_gather_run_score_card(
            dashboard_info, {"ScoreCardTitle": config["Model"]["ModelName"]}
        )
        mlflow.log_artifacts(args.pdf_output_path, "scorecard")


class Workflow:
    def __init__(self, insight, config, arguments, pdf_generator):
        self.pdf_generator = pdf_generator
        self.args = arguments
        self.modeltype = config["Model"]["ModelType"]
        self.data = insight
        self.raiinsight = insight.raiinsight
        self.pdfdata = PdfDataGen(self.data, config)
        self.config = config
        self.cflags = {}
        self.set_component_flags()

    def set_component_flags(self):
        self.cflags = {
            "model_overview": True,
            "model_performance": True,
            "data_explorer": "DataExplorer" in self.config,
            "cohorts": "Cohorts" in self.config or len(self.raiinsight.list()["error_analysis"]["reports"]) > 0,
            "feature_importance": "FeatureImportance" in self.config and
                                  self.raiinsight.list()["explainer"]["is_computed"],
            "fairness": "Fairness" in self.config and len(self.config["Fairness"]["sensitive_features"]) > 0,
            "causal": "Causal" in self.config and len(self.raiinsight.list()["causal"]["causal_effects"]) > 0,
        }

    def generate_pdf(self):
        flag_to_page = {
            "model_overview": self.get_model_overview_page,
            "model_performance": self.get_model_performance_page,
            "data_explorer": self.get_data_explorer_page,
            "cohorts": self.get_cohorts_page,
            "feature_importance": self.get_feature_importance_page,
            "causal": self.get_causal_page,
            "fairness": self.get_fairlearn_page,
        }

        enabled_flags = [flags for flags, enabled in self.cflags.items() if enabled]

        print("Enabled pages are {}".format(enabled_flags))
        body = "".join([str(flag_to_page[flag]()) for flag in enabled_flags])

        to_pdf(
            get_full_html(body),
            os.path.join(self.args.pdf_output_path, "scorecard.pdf"),
            self.args.wkhtml2pdfpath,
        )

    def get_model_overview_page(self):
        data = self.pdfdata.get_model_overview_data()
        return self.pdf_generator.get_model_overview_page(data)

    def get_model_performance_page(self):
        data = self.pdfdata.get_model_performance_data()
        return self.pdf_generator.get_model_performance_page(data)

    def get_data_explorer_page(self):
        data = self.pdfdata.get_data_explorer_data()
        return self.pdf_generator.get_data_explorer_page(data)

    def get_cohorts_page(self):
        data = self.pdfdata.get_cohorts_data()
        return self.pdf_generator.get_cohorts_page(data, self.config["Metrics"])

    def get_feature_importance_page(self):
        data = self.pdfdata.get_feature_importance_data()
        return self.pdf_generator.get_feature_importance_page(data)

    def get_causal_page(self):
        data = self.pdfdata.get_causal_data()
        return self.pdf_generator.get_causal_page(data)

    def get_fairlearn_page(self):
        data = self.pdfdata.get_fairlearn_data()
        return self.pdf_generator.get_fairlearn_page(data)


if __name__ == "__main__":
    # add space in logs
    print("*" * 60)
    print("\n\n")

    # parse args
    args = get_parser().parse_args()
    print("Arguments parsed successfully")
    print(args)
    _module_name = args.component_name
    _module_version = args.component_version
    _get_logger()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
