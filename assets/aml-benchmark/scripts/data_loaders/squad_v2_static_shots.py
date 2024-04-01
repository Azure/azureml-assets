# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Data Loading Script for Truthfulqa static shots."""

import datasets


_CITATION = """\
@article{2016arXiv160605250R,
       author = {{Rajpurkar}, Pranav and {Zhang}, Jian and {Lopyrev},
                 Konstantin and {Liang}, Percy},
        title = "{SQuAD: 100,000+ Questions for Machine Comprehension of Text}",
      journal = {arXiv e-prints},
         year = 2016,
          eid = {arXiv:1606.05250},
        pages = {arXiv:1606.05250},
archivePrefix = {arXiv},
       eprint = {1606.05250},
}
"""

_DESCRIPTION = """\
This dataset contains 2 static shots to give examples of how to respond to questions which does not have the answer in
the context data.These shots are not part of squad_v2 dataset but is created by Microsoft Corporation.
"""

_SQUAD_v2_STATIC_SHOT_DATA = [
    {
        "context":
            ("The population of India is more than a billion."),
        "question":
            ("What is the population of Italy?"),
        "answers": ({"text": [], "answer_start": []}),
    },
    {
        "context":
            ("A black box is lying on a table."),
        "question":
            ("What is the colour of the box?"),
        "answers": ({"text": ["The colour of the box is black."], "answer_start": []}),
    }
]


class Squadv2StaticShots(datasets.GeneratorBasedBuilder):
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
