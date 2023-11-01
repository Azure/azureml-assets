# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Data Loading Script for GSM-8K static shots."""

import datasets


_CITATION = """\
@misc{wei2022chainofthought,
    title={Chain-of-Thought Prompting Elicits Reasoning in Large Language Models},
    author={Jason Wei and Xuezhi Wang and Dale Schuurmans and Maarten Bosma and Brian Ichter \
        and Fei Xia and Ed Chi and Quoc Le and Denny Zhou},
    year={2022},
    eprint={2201.11903},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
This dataset contains eight examples of math word problems, with answers, for full chain-of-thought prompting.
The original data is from Wei et al., Appendix G.
"""

_GSM_STATIC_SHOT_DATA = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. \
            After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "We start with 15 trees. Later we have 21 trees. The difference must be the number of trees \
            they planted. So, they must have planted 21 - 15 = 6 trees. #### 6"
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the \
            parking lot?",
        "answer": "There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. #### 5"
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces \
            do they have left in total?",
        "answer": "Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 \
            chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. #### 39"
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. \
            How many lollipops did Jason give to Denny?",
        "answer": "Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. \
            The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. #### 8"
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. \
            How many toys does he have now?",
        "answer": "He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from \
            dad, so in total he has 7 + 2 = 9 toys. #### 9"
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, \
            from monday to thursday. How many computers are now in the server room?",
        "answer": "There are 4 days from monday to thursday. 5 computers were added each day. \
            That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, \
            so now there are 9 + 20 = 29 computers. #### 29"
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. \
            How many golf balls did he have at the end of wednesday?",
        "answer": "Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. \
            On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. #### 33"
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "answer": "She bought five bagels for $3 each. This means she spent $5 * $3 = $15 on the bagels. \
            She had $23 in beginning, so now she has $23 - $15 = $8. #### 8"
    }
]


class GSM8kStaticShots(datasets.GeneratorBasedBuilder):
    """GSM8K static shot dataset."""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="fewshot",
            version=datasets.Version("1.0.0"),
            description="GSM8K - Static shot data from Wei et al"
        )
    ]

    def _info(self):
        features = datasets.Features(
            {
                "question": datasets.Value("string"),
                "answer": datasets.Value("string")
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            citation=_CITATION
        )

    def _split_generators(self, dl_manager):
        """Return SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name="dev",
                gen_kwargs={
                    "shot_data": _GSM_STATIC_SHOT_DATA
                },
            )
        ]

    def _generate_examples(self, shot_data):
        """Yield examples as a tuple."""
        for i, one_shot_dict in enumerate(shot_data):
            yield i, one_shot_dict
