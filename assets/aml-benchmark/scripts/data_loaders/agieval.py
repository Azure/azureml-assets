import datasets
import json
import ast
import pandas as pd

_CITATION = """\
@misc{zhong2023agieval,
      title={AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models}, 
      author={Wanjun Zhong and Ruixiang Cui and Yiduo Guo and Yaobo Liang and Shuai Lu and Yanlin Wang and Amin Saied and Weizhu Chen and Nan Duan},
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

def _get_language(config):
    if 'gaokao' in config or config == 'logiqa-zh':
        return 'chinese'
    else:
        return 'english'
    

def _format_question(src_dict, lang, problem_number, add_label=False):
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
        # Load few shot data
        df_fs = pd.read_csv(fewshot_path, keep_default_na=False)
        shots = df_fs[df_fs.index % 2 == 0].reset_index(drop=True)
        explanations = df_fs[df_fs.index % 2 != 0].reset_index(drop=True)

        # Format fewshot prompt
        n_shots = shots.shape[0]
        lang = _get_language(self.config.name)
        fs_str = _INTRO[lang] + '\n'
        for ishot in range(shots.shape[0]):
            if pd.isna(shots[self.config.name][ishot]):
                continue
            fs_dict = ast.literal_eval(shots[self.config.name][ishot])
            fs_str += _format_question(fs_dict, lang, ishot + 1, add_label=True)

        # Format eval questions for the prompt 
        df = pd.read_json(eval_path, lines=True)
        for key, row in df.iterrows():
            prompt_str = _format_question(row, lang, n_shots + 1, add_label=False)
            yield key, {
                "fewshot_prompt": fs_str + prompt_str,
                "label": row["label"],
            }
