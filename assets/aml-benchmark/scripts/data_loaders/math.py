# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Data Loading Script for MATH dataset."""

import json

import datasets


_CITATION = """\
@article{hendrycksmath2021,
  title={Measuring Mathematical Problem Solving With the MATH Dataset},
  author={Dan Hendrycks and Collin Burns and Saurav Kadavath and Akul Arora and Steven Basart and Eric Tang \
and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
"""

_DESCRIPTION = """\
The Mathematics Aptitude Test of Heuristics (MATH) dataset consists of problems \
from mathematics competitions, covering 7 subjects including algebra, geometry, \
precalculus, and more. Each problem in MATH has a full step-by-step solution, \
which can be used to teach models to generate answer derivations and explanations.\
"""

_HOMEPAGE = "https://github.com/hendrycks/math/"
_LICENSE = "MIT"
_URL = "https://people.eecs.berkeley.edu/~hendrycks/MATH.tar"

_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


class MATH(datasets.GeneratorBasedBuilder):
    """MATH dataset."""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=sub,
            version=datasets.Version("1.0.0"),
            description=f"MATH - Subject: {sub}",
        )
        for sub in _SUBJECTS
    ]

    def _info(self):
        features = datasets.Features(
            {
                "problem": datasets.Value("string"),
                "level": datasets.Value("string"),
                "type": datasets.Value("string"),
                "solution": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Return SplitGenerators."""
        archive = dl_manager.download(_URL)
        iter_archive = dl_manager.iter_archive(archive)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "iter_archive": iter_archive,
                    "split": "train"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "iter_archive": iter_archive,
                    "split": "test"
                },
            ),
        ]

    def _generate_examples(self, iter_archive, split):
        """Yield examples as a tuple."""
        for id_file, (path, file) in enumerate(iter_archive):
            if f"{split}/{self.config.name}" in path:
                content = file.read().decode('utf-8')
                data = json.loads(content)
                yield id_file, data
