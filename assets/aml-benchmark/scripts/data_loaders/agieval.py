# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Data Loading Script for AGIEval."""

import datasets
import ast
import pandas as pd

_CITATION = """\
@misc{zhong2023agieval,
      title={AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models},
      author={Wanjun Zhong and Ruixiang Cui and Yiduo Guo and Yaobo Liang and Shuai Lu \
        and Yanlin Wang and Amin Saied and Weizhu Chen and Nan Duan},
      year={2023},
      eprint={2304.06364},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_HOMEPAGE = "https://github.com/ruixiangcui/AGIEval/tree/main"

_LICENSE = (
    "https://github.com/ruixiangcui/AGIEval/blob/main/LICENSE"
)

_HEAD = 'https://raw.githubusercontent.com/ruixiangcui/AGIEval/main/data/v1/'
_FEWSHOT_URL = 'https://raw.githubusercontent.com/ruixiangcui/AGIEval/main/data/few_shot_prompts.csv'
_DESCRIPTION = "AGIEval is a human-centric benchmark specifically designed to evaluate the general abilities \
    of foundation models in tasks pertinent to human cognition and problem-solving. This benchmark is derived \
    from 20 official, public, and high-standard admission and qualification exams intended for \
    general human test-takers, such as general college admission tests"

_CONFIGS = [
    'aqua-rat',
    'gaokao-biology',
    'gaokao-chemistry',
    'gaokao-chinese',
    'gaokao-english',
    'gaokao-geography',
    'gaokao-history',
    'gaokao-mathqa',
    'lsat-ar',
    'lsat-lr',
    'lsat-rc',
    'sat-en',
    'sat-math'
]

_INTRO = {
    'english': 'Here are the answers for the problems in the exam.',
    'chinese': '\u4ee5\u4e0b\u662f\u8003\u8bd5\u4e2d\u5404\u4e2a\u95ee\u9898\u7684\u7b54\u6848\u3002',
}

_PROBLEM_HEADING = {
    'english': 'Problem',
    'chinese': '\u95ee\u9898'
}

_CHOOSE_OPTIONS = {
    'english': 'Choose from the following options:',
    'chinese': '\u4ece\u4ee5\u4e0b\u9009\u9879\u4e2d\u9009\u62e9'
}

_ANSWER_INTRO = {
    'english': 'The answer is therefore',
    'chinese': '\u7b54\u6848\u662f'
}


def _get_language(config):
    if config != 'gaokao-english' and ('gaokao' in config or config == 'logiqa-zh'):
        return 'chinese'
    else:
        return 'english'


def _construct_example_dict(src_dict):
    # Extract a single example from the source dictionary
    out_dict = {}
    out_dict['passage'] = ''
    out_dict['solution'] = ''
    if 'passage' in src_dict and isinstance(src_dict['passage'], str) and len(src_dict['passage']) > 0:
        out_dict['passage'] = src_dict['passage']
    other_dict = src_dict.get('other', {})
    if isinstance(other_dict, dict):
        solution = other_dict.get('solution', '')
        if isinstance(solution, str):
            out_dict['solution'] = solution
    out_dict['question'] = src_dict['question']
    out_dict['options'] = src_dict['options']
    out_dict['label'] = src_dict['label']

    return out_dict


class AgiEval(datasets.GeneratorBasedBuilder):
    """Builder for AGIEval dataset."""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=sub,
            version=datasets.Version("1.0.0"),
            description=_DESCRIPTION,
        )
        for sub in _CONFIGS
    ]
    DEFAULT_CONFIG_NAME = "aqua-rat"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls = {
            "test": _HEAD + f'{self.config.name}.jsonl',
            "few_shot": _FEWSHOT_URL,
        }
        data_dir = dl_manager.download_and_extract(urls)
        splits = [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"eval_path": data_dir["test"], "fewshot_path": data_dir["few_shot"]}
            )
        ]

        return splits

    def _generate_examples(self, eval_path, fewshot_path):
        # Load few shot data
        df_fs = pd.read_csv(fewshot_path)
        shots = df_fs[df_fs.index % 2 == 0].reset_index(drop=True)

        # Add optional, language dependent intros/headings
        lang = _get_language(self.config.name)
        out_dict = {}
        out_dict['prompt_intro'] = _INTRO[lang]
        out_dict['problem_heading'] = _PROBLEM_HEADING[lang]
        out_dict['options_intro'] = _CHOOSE_OPTIONS[lang]
        out_dict['answer_intro'] = _ANSWER_INTRO[lang]

        # Add list of example shots
        shot_list = []
        for i in range(shots.shape[0]):
            if pd.isna(shots[self.config.name][i]):
                continue
            fs_dict = ast.literal_eval(shots[self.config.name][i])
            shot_list.append(_construct_example_dict(fs_dict))
        out_dict['shots'] = shot_list

        # Load eval data and yield examples
        df = pd.read_json(eval_path, lines=True)
        for key, row in df.iterrows():
            out_dict.update(_construct_example_dict(row))
            yield key, out_dict.copy()
