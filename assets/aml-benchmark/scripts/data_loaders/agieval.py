import datasets
import json
import ast
import pandas as pd

_CITATION = """\
@ARTICLE{10174688,
  author={Liu, Hanmeng and Liu, Jian and Cui, Leyang and Teng, Zhiyang and Duan, Nan and Zhou, Ming and Zhang, Yue},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  title={LogiQA 2.0 — An Improved Dataset for Logical Reasoning in Natural Language Understanding},
  year={2023},
  volume={},
  number={},
  pages={1-16},
  doi={10.1109/TASLP.2023.3293046}}
"""

_HOMEPAGE = "https://github.com/csitfun/LogiQA2.0/tree/main"

_LICENSE = (
    "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License"
)

_HEAD = 'https://raw.githubusercontent.com/ruixiangcui/AGIEval/main/data/v1/'
_FEWSHOT_URL = 'https://raw.githubusercontent.com/ruixiangcui/AGIEval/main/data/few_shot_prompts.csv'

_DESCRIPTION = "AGIEval is a human-centric benchmark specifically designed to evaluate the general abilities of foundation models in tasks pertinent to human cognition and problem-solving. This benchmark is derived from 20 official, public, and high-standard admission and qualification exams intended for general human test-takers, such as general college admission tests"

_CONFIGS = [
    'aqua-rat',
    'gaokao-biology',
    'gaokao-chemistry',
    'gaokao-chinese',
    'gaokao-english',
    'gaokao-geography',
    'gaokao-history',
    'gaokao-mathcloze',
    'gaokao-mathqa',
    'logiqa-en',
    'logiqa-zh',
    'lsat-ar',
    'lsat-lr',
    'lsat-rc',
    'math',
    'sat-en-without-passage',
    'sat-en',
    'sat-math'
]

_INTRO = {
    'english': 'Here are the answers for the problems in the exam.',
    'chinese': '\u4ee5\u4e0b\u662f\u8003\u8bd5\u4e2d\u5404\u4e2a\u95ee\u9898\u7684\u7b54\u6848\u3002',
}

_PROBLEM_TEMPLATE = {
    'english': 'Problem {}.',
    'chinese': '\u95ee\u9898 {}.   '
}

_CHOOSE_OPTIONS = {

    'english': 'Choose from the following options:',
    'chinese': '\u4ece\u4ee5\u4e0b\u9009\u9879\u4e2d\u9009\u62e9'
}

_ANSWER_INTRO = {
    'english': 'The answer is therefore',
    'chinese': '\u7b54\u6848\u662f'
}

_SHOT_SEPARATOR = '\n<END>\n'

def get_language(config):
    if 'gaokao' in config or config == 'logiqa-zh':
        return 'chinese'
    else:
        return 'english'
    

def format_question(src_dict, lang, problem_number, add_label=False):
    s = _PROBLEM_TEMPLATE[lang].format(problem_number) + '   '
    s += src_dict['question'] + '\n'
    s += _CHOOSE_OPTIONS[lang] + '    '
    s += ' '.join(src_dict['options']) + '\n'
    if add_label:
        s += _ANSWER_INTRO[lang] + f' {src_dict["label"]}' + _SHOT_SEPARATOR
    
    return s


class AgiEval(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

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

        features = datasets.Features(
            {
                "passage": datasets.Value("string"),
                "question": datasets.Value("string"),
                "options": datasets.features.Sequence(datasets.Value("string")),
                "label": datasets.ClassLabel(num_classes=5, names=["A", "B", "C", "D", "E"]),
                "solution": datasets.Value("string"),
            }
        )

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
        # Load the few shot data for the config
        df_fs = pd.read_csv(fewshot_path, keep_default_na=False)
        samples = df_fs[df_fs.index % 2 == 0].reset_index(drop=True)
        explanations = df_fs[df_fs.index % 2 != 0].reset_index(drop=True)

        # Extract fewshot samples
        n_shots = samples.shape[0]
        lang = get_language(self.config.name)
        fs_str = _INTRO[lang] + '\n'
        for key in range(samples.shape[0]):
            fs_dict = ast.literal_eval(samples[self.config.name][key])
            fs_str += format_question(fs_dict, lang, key + 1, add_label=True)

        # Extract the eval data 
        df = pd.read_json(eval_path, lines=True)
        for key, row in df.iterrows():
            prompt_str = format_question(row, lang, n_shots + 1, add_label=False)
            yield key, {
                "fewshot_prompt": fs_str + prompt_str,
                "label": row["label"],
            }
