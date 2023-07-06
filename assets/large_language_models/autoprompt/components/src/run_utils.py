# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Run Logging Utilities."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import traceback
import json

from azureml.core.run import Run, _OfflineRun
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from tasks.base import TaskResults
from constants import ALL_METRICS
from azureml.rag.utils.logging import get_logger
from logging_utilities import log_info, log_warning

logger = get_logger("run_utils")

current_run = Run.get_context()

PIPELINE_RUN = "azureml.PipelineRun"
STEP_RUN = "azureml.StepRun"


class DummyWorkspace:
    """Dummy Workspace class for offline logging."""

    def __init__(self):
        """__init__."""
        self.name = "local-ws"
        self.subscription_id = ""
        self.location = "local"


class DummyExperiment:
    """Dummy Experiment class for offline logging."""

    def __init__(self):
        """__init__."""
        self.name = "offline_default_experiment"
        self.id = "1"
        self.workspace = DummyWorkspace()


class TestRun:
    """Main class containing Current Run's details."""

    def __init__(self):
        """__init__."""
        self._run = Run.get_context()
        if isinstance(self._run, _OfflineRun):
            self._experiment = DummyExperiment()
            self._workspace = self._experiment.workspace
        else:
            self._experiment = self._run.experiment
            self._workspace = self._experiment.workspace

    @property
    def run(self):
        """Azureml Run.

        Returns:
            _type_: _description_
        """
        return self._run

    @property
    def experiment(self):
        """Azureml Experiment.

        Returns:
            _type_: _description_
        """
        return self._experiment

    @property
    def workspace(self):
        """Azureml Workspace.

        Returns:
            _type_: _description_
        """
        return self._workspace

    @property
    def compute(self):
        """Azureml compute instance.

        Returns:
            _type_: _description_
        """
        if not isinstance(self._run, _OfflineRun):
            target_name = self._run.get_details()["target"]
            if self.workspace.compute_targets.get(target_name):
                return self.workspace.compute_targets[target_name].vm_size
            else:
                return "serverless"
        return "local"

    @property
    def region(self):
        """Azure Region.

        Returns:
            _type_: _description_
        """
        return self._workspace.location

    @property
    def subscription(self):
        """Azureml Subscription.

        Returns:
            _type_: _description_
        """
        return self._workspace.subscription_id

    @property
    def root_run(self):
        """Get Roon run of the pipeline.

        Returns:
            _type_: _description_
        """
        cur_run = self._run
        if isinstance(cur_run, _OfflineRun) or cur_run.parent is None:
            return self._run
        while cur_run.parent is not None:
            cur_run = cur_run.parent
        return cur_run


# Creating a copy of custom dimensions to avoid cyclic dependency.
def get_custom_dimensions():
    """Get Custom Dimensions for Activity logging."""
    run = TestRun()
    custom_dimensions = {
        "app_name": "autoprompt",
        "run_id": run.run.id,
        "parent_run_id": run.root_run.id,
        "experiment_id": run.experiment.name,
        "workspace": run.workspace.name,
        "subscription": run.subscription,
        "target": run.compute,
        "region": run.region
    }
    return custom_dimensions


CUSTOM_DIMENSIONS = get_custom_dimensions()


def get_pipeline_job_parent(current_run):
    """
    Check if we should log to parent pipeline run.

    :return: Parent run if we should log else None.
    :rtype: azureml.core.run
    """
    # print('current_run: ', current_run)
    level = 1
    parent_run = current_run.parent
    # print(f'level {level} parent_run: ', parent_run)
    child_run = None
    while parent_run is not None and (parent_run.type == PIPELINE_RUN or parent_run.type == STEP_RUN):
        child_run = parent_run
        parent_run = parent_run.parent
        level += 1
        # print(f'level {level} parent_run: ', parent_run)
    return child_run


def log_tsne(current_run, data):
    """Log TSNE."""
    sample_size = 1000
    if len(data) < 1000:
        sample_size = data.shape[0]
    df = data.sample(n=sample_size)
    df = df.astype(str)

    def get_title(context):
        title = ''
        split_context = context.split("\nContext: ")
        if len(split_context) > 1:
            first_result = split_context[1]
            second_result = first_result.split("\ntitleSuffix:")[0]

            if second_result[-1] == "?":
                title = second_result[7:-1]
            else:
                title = second_result[7:]

            if title[0] in ['"', "'"]:
                title = title[1:]
            if title[-1] in ['"', "'"]:
                title = title[:-2]
        return title

    # populating df columns
    df['title'] = df['input'].apply(lambda x: get_title(x))
    if df[df['title'] == ''].shape[0] == df.shape[0]:
        log_info(logger, 'No title found for creating T-SNE chart', CUSTOM_DIMENSIONS)
        return
    title_count = df['title'].value_counts(sort=True, ascending=False)
    top_5_titles = title_count.index.tolist()[0:5]
    df['Topic'] = df['title'].apply(lambda x: x if x in top_5_titles else "Other")
    df['context_question'] = df['input']

    # getting embeddings
    texts = df['context_question'].values
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = model.encode(texts)
    # we might need to try with different values of perplexity based on dataset.
    # source : https://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html \
    #   #sphx-glr-auto-examples-manifold-plot-t-sne-perplexity-py
    # how to use perplexity effectively : https://distill.pub/2016/misread-tsne/
    perplexity_values = [10, 30, 50]

    for perplexity in perplexity_values:
        # perplexity should always be less than the number of data samples
        if len(data) > perplexity:
            # running t-SNE algorithm
            tsne = TSNE(n_components=2, random_state=1000, perplexity=perplexity)
            embeddings_2d = tsne.fit_transform(embeddings)

            # plotting results
            df['input-tsne-x'] = embeddings_2d[:, 0]
            df['input-tsne-y'] = embeddings_2d[:, 1]
            df['GPT-Similarity'] = df['gpt_similarity']

            plt.figure(dpi=200)
            ax = sns.scatterplot(
                x="input-tsne-x", y="input-tsne-y",
                hue="Topic",
                style='GPT-Similarity',
                palette=sns.color_palette("hls", 10),
                data=df,
                alpha=0.9
            )
            ax.set_title('Input sample clusters (t-SNE) and GPT-Similarity score distribution per cluster')

            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            figure_name = f"T-SNE_autoprompt_perplexity_{perplexity}.png"
            figure_path = f"./{figure_name}"
            plt.savefig(figure_path, bbox_inches="tight")
            current_run.log_image(figure_name, path=figure_path)


def log_results(results: TaskResults, y_true_test, top_k, predictions_file):
    """Log autoprompt metrics to current run.

    Args:
        results (TaskResults)
        y_true_test (pd.Series)
        top_k (int)
        predictions_file (str)
    """
    dev_results_df = pd.DataFrame(results.prompt_results)
    dev_results_df.to_csv("best_prompt_predictions.csv", index=False)

    all_results_df = pd.DataFrame(results.dev_results)
    all_results_df.to_csv("all_dev_results.csv", index=False)

    current_run.log("f1_macro", results.validation_f1)
    current_run.log("exact_match", results.validation_acc)
    for name in ALL_METRICS:
        value = getattr(results, name, None)
        if not value:
            log_warning(logger, f"{name} not found in results. Skipping.", CUSTOM_DIMENSIONS)
            continue
        try:
            if name.startswith("bert"):
                # print(name, value)
                current_run.log(name, np.mean(value))
            if name.startswith("gpt_"):
                if not isinstance(value, list) and not isinstance(value, np.ndarray):
                    log_warning(logger, f"{name} is not an iterable. \nValue: {value}", CUSTOM_DIMENSIONS)
                    continue
                metric_name = "mean_"+name
                int_score = [int(i) for i in value]
                current_run.log(metric_name, np.mean(int_score))
                counts = [0]*5
                for i in int_score:
                    counts[i-1] += 1
                cur_score = {
                    "_rating": [i for i in range(1, 6)],
                    "count": counts
                }
                current_run.log_table(name, cur_score)
        except Exception as e:
            if name.startswith("gpt_"):
                if (isinstance(value, list) or isinstance(value, np.ndarray)) and len(value) > 0:
                    exception_cls_name = value[0]
                    log_warning(
                        logger,
                        "Ignoring metric: " + name + "\n Computation Failed due to: " + exception_cls_name,
                        CUSTOM_DIMENSIONS)
            else:
                log_warning(logger, "Ignoring metric: " + name + " due to error: " + repr(e), CUSTOM_DIMENSIONS)
                traceback.print_exc()

    x, y = np.histogram(results.bert_f1, bins=10)
    current_run.log_table("Bert F1 Score", value={"_score": list(y)[1:], "count": list(x)})

    x, y = np.histogram(results.bert_precision, bins=10)
    current_run.log_table("Bert Precision", value={"_score": list(y)[1:], "count": list(x)})

    x, y = np.histogram(results.bert_recall, bins=10)
    current_run.log_table("Bert Recall", value={"_score": list(y)[1:], "count": list(x)})

    val_pred_df = pd.DataFrame({
        "ground_truths": y_true_test,
        "predictions": results.validation_predictions,
        "input": results.validation_texts
    })

    try:
        for name in ALL_METRICS:
            value = getattr(results, name, [np.nan]*len(y_true_test))
            val_pred_df[name] = value
        val_pred_df.to_csv("validation_predictions.csv", index=False)
        current_run.upload_file(
            "validation_predictions.csv",
            "validation_predictions.csv",
            datastore_name="workspaceblobstore")

        log_tsne(current_run, val_pred_df)
    except Exception as e:
        log_warning(logger, "Logging TSNE Plot failed with error:"+repr(e), CUSTOM_DIMENSIONS)
        traceback.print_exc()

    current_run.upload_file(
        "best_prompt_predictions.csv",
        "best_prompt_predictions.csv",
        datastore_name="workspaceblobstore")
    current_run.upload_file("all_dev_results.csv", "all_dev_results.csv", datastore_name="workspaceblobstore")

    top_k = min(top_k, dev_results_df.shape[0])
    with open(predictions_file, "w") as f:
        best_prompt_dict = {}
        for metric in ALL_METRICS:
            if metric == 'exact_match' or metric == 'gpt_consistency':
                continue
            sorted_table = dev_results_df.sort_values(metric, ascending=False)
            best_prompt_dict[f"best_prompt_{metric}"] = sorted_table["prompt"].tolist()[:top_k]
        json.dump(best_prompt_dict, f)
