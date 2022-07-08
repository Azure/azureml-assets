import argparse
import json
import logging
import os
import re

from azureml.core import Run
from responsibleai import __version__ as responsibleai_version

from _score_card._rai_insight_data import RaiInsightData, PdfDataGen
from _score_card.common_components import to_pdf, get_full_html
import _score_card.regression_components as RegressionComponents
import _score_card.classification_components as ClassificationComponents

from constants import DashboardInfo, PropertyKeyValues, RAIToolType
from rai_component_utilities import load_dashboard_info_file

threshold_reg = re.compile(r"([<>=]{1,2})([0-9.]+)")

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def parse_args():
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

    return parser.parse_args()


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


def add_properties_to_gather_run(dashboard_info, rai_info):
    included_tools = {
        RAIToolType.CAUSAL: False,
        RAIToolType.COUNTERFACTUAL: False,
        RAIToolType.ERROR_ANALYSIS: False,
        RAIToolType.EXPLANATION: False,
        RAIToolType.SCORECARD: True,
    }

    _logger.info("Adding properties to the gather run")
    run = Run.get_context()

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

    _logger.info("Appending tool present information")
    for k, v in included_tools.items():
        key = PropertyKeyValues.RAI_INSIGHTS_TOOL_KEY_FORMAT.format(k)
        run_properties[key] = str(v)

    _logger.info("Making service call")
    run.add_properties(run_properties)
    _logger.info("Properties added to score card run")


def validate_and_correct_config(config):
    return config


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

    config = validate_and_correct_config(config)

    for k, v in config["Metrics"].items():
        if "threshold" in v.keys():
            tt, ta = parse_threshold(v["threshold"])
            if tt and ta:
                config["Metrics"][k]["threshold"] = (tt, ta)
            else:
                config["Metrics"][k].pop("threshold")

    if not args.local:
        run = Run.get_context()
        config["runinfo"] = {
            "submittedBy": run.get_details()["submittedBy"],
            "startTimeUtc": run.get_details()["startTimeUtc"],
        }

    model_type = config["Model"]["ModelType"]
    if model_type == "Regression":
        wf = Workflow(insight_data, config, args, RegressionComponents)
    elif model_type == "Classification":
        wf = Workflow(insight_data, config, args, ClassificationComponents)
    else:
        # move logic to config validation
        error_msg = "Invalid model type supplied: {}".format(model_type)
        _logger.error(
            error_msg
        )
        raise ValueError(error_msg)

    wf.generate_pdf()

    if not args.local:
        add_properties_to_gather_run(
            dashboard_info, {"ScoreCardTitle": config["Model"]["ModelName"]}
        )


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
            "cohorts": "Cohorts" in self.config
            or len(self.raiinsight.list()["error_analysis"]["reports"]) > 0,
            "feature_importance": "FeatureImportance" in self.config
            and self.raiinsight.list()["explainer"]["is_computed"],
            "fairness": "Fairness" in self.config,
            "causal": len(self.raiinsight.list()["causal"]["causal_effects"]) > 0,
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

    # run main function
    main(parse_args())

    # add space in logs
    print("*" * 60)
    print("\n\n")
