# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""reward functions."""
import logging
import re
from typing import List

logger = logging.getLogger(__name__)


def format_reward(completions, **kwargs):
    """For determining whether the predicted answer is in the correct format.

    It checks if the reasoning process is enclosed within <think> and </think> tags,
    while the final answer is enclosed within <answer> and </answer> tags.

    Args:
        completions (list): List of model predictions.
    Returns:
        list: List of rewards (1.0 for correct format, 0.0 for incorrect format).
    """
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [
        re.match(pattern, content, re.DOTALL | re.MULTILINE)
        for content in completion_contents
    ]
    return [1.0 if match else 0.0 for match in matches]


def _medmcqa_match_fn(pred, ref):
    """Determine whether a predicted answer matches the reference choice for a single MCQA record.

    Even if the requested output format is not strictly followed, extract the answer based on static
    patterns and match the extracted answer (A, B, C, D) to the reference choice.

    Steps:
        Split the model’s full prediction text into individual lines.
        Inspect the last line:
            If it contains “Answer”, strip out everything before the actual answer and clean up whitespace and punctuation.
        If the last line does not include “Answer”, search the prediction text for these markers:
            “Final Answer”
            “<answer>”
            “<|begin_of_solution|>”
        For each marker, find its last occurrence, grab up to 250 characters immediately after it to avoid overly long context, and clean that snippet.
        If no markers appear, clean up the last line.
        Once you have a candidate answer string:
            If it matches the pattern “[A-D].”, discard the dot and keep only the letter.
            If the result is not one of “A”, “B”, “C”, or “D”, return False.
        Compare the cleaned-up answer letter to the reference label and return True if they match, otherwise return False.

    Args:
        pred (str): The raw model prediction text.
        ref (str): The reference answer choice (one of "A", "B", "C", "D").

    Returns:
        bool: True if the extracted choice equals ref; False otherwise.
    """
    ext_pattern1 = r"Final Answer"
    ext_pattern2 = r"<answer>"
    ext_pattern3 = r"<|begin_of_solution|>"

    choices_patobj = re.compile(r"[A-D]\.")
    logger.debug("**")

    def soln_start_match_fn(pat):
        return [match.span()[1] for match in re.finditer(pat, pred, re.DOTALL)]

    lines = pred.split("\n")

    if len(lines) > 0:
        # Finding the answer snippet when the LLM response is in required format
        check_next_cond = True
        if lines[-1].find("Answer") >= 0:
            hyp_unfilt = (
                lines[-1]
                .replace("*", "")
                .replace("Final Answer:", "")
                .replace("Answer:", "")
                .replace("<answer>", "")
                .replace("</answer>", "")
                .strip()
            )
            check_next_cond = False
        # Finding the answer based on the "Final Answer" string in the LLM response
        if check_next_cond:
            check_next_cond = True
            cond_match = soln_start_match_fn(ext_pattern1)

            if len(cond_match) > 0:
                suffix_text = pred[cond_match[-1]:]
                hyp_unfilt = suffix_text[: min(250, len(suffix_text))]
                check_next_cond = False
        # Finding the answer based on the "<answer>" tags in the LLM response
        if check_next_cond:
            check_next_cond = True
            cond_match = soln_start_match_fn(ext_pattern2)
            if len(cond_match) > 0:
                suffix_text = pred[cond_match[-1]:]
                hyp_unfilt = suffix_text[: min(250, len(suffix_text))]
                check_next_cond = False
        # Finding the answer based on the "<|begin_of_solution|>" tags in the LLM response
        if check_next_cond:
            check_next_cond = True
            cond_match = soln_start_match_fn(ext_pattern3)
            if len(cond_match) > 0:
                suffix_text = pred[cond_match[-1]:]
                hyp_unfilt = suffix_text[: min(250, len(suffix_text))]
                check_next_cond = False
        # Last ditch attempt to parse the response into an answer
        if check_next_cond:
            hyp_unfilt = (
                lines[-1]
                .replace("*", "")
                .replace("Final Answer:", "")
                .replace("Answer:", "")
                .replace("<answer>", "")
                .replace("</answer>", "")
                .strip()
            )

        # Extract the single letter answer from the response
        if re.match(choices_patobj, hyp_unfilt) is not None:
            hyp_unfilt = hyp_unfilt[0]
        logger.debug(f"Hyp Unfilt: {hyp_unfilt}")
        if hyp_unfilt not in ["A", "B", "C", "D"]:
            logger.debug(
                f"ERROR: Invalid answer (outside of allowed choices): {hyp_unfilt}"
            )
            return False

        return hyp_unfilt == ref


def accuracy(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth.

    Args:
        completions (list): List of model predictions.
        solution (list): List of ground truth answers.
    Returns:
        list: List of rewards (1.0 for correct answer, 0.0 for incorrect answer).
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        # _medmcqa_match_fn is used to calculate the accuracy reward for a single completion
        rewards.append(float(_medmcqa_match_fn(content, sol)))
    return rewards


reward_registry = {
    "format": format_reward,
    "accuracy": accuracy,
}


def get_rewards_funcs(reward_names: List[str]):
    """
    Retrieve the reward functions for the given names.

    Args:
        reward_names (list): List of reward function names.
    Returns:
        list: List of reward functions.
    """
    return [reward_registry[name] for name in reward_names if name in reward_registry]
