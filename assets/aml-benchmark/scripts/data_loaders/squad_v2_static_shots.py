# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Data Loading Script for Truthfulqa static shots."""

import datasets


_CITATION = """\
@misc{wei2022chainofthought,
    title={Squad_v2: Measuring How Models Mimic Human Falsehoods},
    author={Stephanie Lin and Jacob Hilton and Owain Evans},
    year={2022},
    eprint={2109.07958v2},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
This dataset contains 2 static shots to give examples of how to respond to questions with no answer in
the context data for squad_v2.These shots was created by Microsoft.
"""

_SQUAD_v2_STATIC_SHOT_DATA = [
    {
        "question":
            ("What is human life expectancy in the United States?"),
        "context":
            ("Happiness is the key to a long life."),
        "answers":({"text": [],"answer_start": []}),
    },
    {
        "question":
            ("What is the colour of the box?"),
        "context":
            ("A black box is lying on a table."),
        "answers":({"text": ["The colour of the box is black."], "answer_start": []}),
    }
]


class GSM8kStaticShots(datasets.GeneratorBasedBuilder):
    """Squad_v2 static shot dataset."""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="fewshot",
            version=datasets.Version("1.0.0"),
            description="Squad_v2 static shots for fewshot learning. Created by Microsoft."
        )
    ]

    def _info(self):
        features = datasets.Features(
            {
                "question": datasets.Value("string"),
                "context": datasets.Value("string"),
                "answers": datasets.Features(
                    {
                        "text": datasets.features.Sequence(datasets.Value("string")),
                        "answer_start": datasets.features.Sequence(datasets.Value("string"))
                    }
                ),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            # citation=_CITATION
        )

    def _split_generators(self, dl_manager):
        """Return SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name="dev",
                gen_kwargs={
                    "shot_data": _SQUAD_v2_STATIC_SHOT_DATA,
                },
            )
        ]

    def _generate_examples(self, shot_data):
        """Yield examples as a tuple."""
        for i, one_shot_dict in enumerate(shot_data):
            yield i, one_shot_dict
