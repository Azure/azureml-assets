# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Groundedness Evaluator with real flow execution."""

import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner, ExpectedResult
from ...builtin.groundedness.evaluator._groundedness import GroundednessEvaluator


@pytest.mark.quality
class TestGroundednessEvaluatorQuality(BaseQualityEvaluatorRunner):
    """
    Quality tests for Groundedness Evaluator.

    Tests actual LLM evaluation with real flow execution (no mocking).
    """

    evaluator_type = GroundednessEvaluator

    file_search_tool_definitions = [
        {
            "name": "file_search",
            "type": "file_search",
            "description": (
                "Search for data across uploaded files. A single call can return multiple "
                "results/files in the 'results' field."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ranking_options": {
                        "type": "object",
                        "properties": {
                            "ranker": {
                                "type": "string",
                                "description": "Ranking algorithm to use."
                            },
                            "score_threshold": {
                                "type": "number",
                                "description": "Threshold for search results."
                            },
                        },
                        "description": "Ranking options for search results.",
                    }
                },
            },
        }
    ]

    swiss_news_content_1 = (  # noqa: E501
        "It added that their offer would take \u201cUS concerns into account"
        "\u201d.\n\nBern was blindsided last week when Trump slapped a 39 per cent tariff rate on the country, "
        "citing\n\nthe US\u2019s \u201chuge\u201d deficit with Switzerland.\n\nThe new rate is higher than the"
        " 31 per cent announced by the US president on \u201cliberation day\u201d in\n\nApril, and is set to come"
        " into force on August 7.\n\nKeller-Sutter, who has rejected press claims that her call with Trump last "
        "Thursday led to the\n\nhigher rate, had hoped for a levy of closer to 10 per cent as part of lengthy"
        " negotiations with\n\nWashington.\n\nOn Monday, the government held an extraordinary cabinet meeting in"
        " which it discussed greater\n\ninvolvement of Parmelin\u2019s department in negotiations with the White"
        " House, according to two\n\npeople familiar with talks.\n\n\n\n\u201cIt makes sense for this department"
        " to be involved,\u201d one of the people said. \n\nParmelin, who is also the country\u2019s vice-president,"
        " is part of the rightwing Swiss People\u2019s party.\n\nKeller-Sutter is from the centrist Liberal party."
        "\n\nParmelin has been vocal since the US imposed the 39 per cent tariff, signalling publicly over the"
        "\n\nweekend a willingness to revise Switzerland\u2019s offer to the US.\n\nHe proposed increasing Swiss"
        " investment in America, including in US LNG imports, as part of a\n\npossible trade concession package."
        " \n\nThe delegation in Washington will also include Helene Budliger Artieda, state secretary"
        " for\n\neconomic affairs, and Daniela Stoffel, state secretary for international finance."
        "\n\nKeller\u2011Sutter has voiced regret over her government\u2019s failure to reach a settlement"
        " with the US last\n\nweek.\n\nThe country held multiple rounds of talks with Trump\u2019s trade"
        " representative Jamieson Greer and\n\nTreasury secretary Scott Bessent, and Bern had believed that"
        " a provisional deal of closer to 10 per\n\ncent had been reached.\n\nTrump last week signalled that"
        " the US\u2019s $39bn trade deficit with Switzerland was the key reason\n\nfor the high levy."
        " However, many business groups have pointed out that this turned to a surplus in\n\n"
        "April and May of this year.\n\nData compiled by the Swiss American Chamber of Commerce"
        " show Switzerland ran a goods trade\n\nsurplus of $4.2bn with the US in May, compared with"
        " a goods deficit of $22bn in January.\n\n\n\nCopyright The Financial Times Limited 2025."
        " All rights reserved.\n\nIt said this was primarily due to pharmaceutical companies stockpiling "
        "ahead of Trump\u2019s first\n\nround of tariffs in April.\n\nTrade lobby groups have branded the tariff "
        "arbitrary, and warned of threats to tens of thousands of\n\njobs in watchmaking, machinery, chemicals, "
        "food and pharma sectors \u2014 industries heavily\n\norientated towards the US market.\n\nThe Swiss "
        "government said on Monday that the US deficit with Switzerland up to March 2025 was\n\nnot the result "
        "of any \u201cunfair trade practices\u201d. It added that it had unilaterally scrapped all tariffs on\n\n"
        "industrial goods as of January 2024, meaning that more than 99 per cent of US goods enter\n\n"
        "Switzerland tariff-free.\n\nhttps://help.ft.com/help/legal-privacy/copyright/copyright-policy/"
    )

    swiss_news_content_2 = (  # noqa: E501
        "Swiss president to present \u2018more attractive\u2019 tariff offer to White House\n\n\n"
        "Start your weekend thinking\nEnjoy FT Weekend, beautifully laid out in print and on your devices "
        "with FT Digital\nEdition.\n\nLearn\n\nmore\n\nTrump tariffs\n\nSwiss president to present \u2018more "
        "attractive\u2019 tariff offer to White\n\nHouse\n\nKarin Keller-Sutter and economy minister Guy "
        "Parmelin\u2019s Washington trip\n\nfollows Trump decision to hit country with steep tariffs\n\n"
        "https://subs.ft.com/weekendexp?segmentId=41f4eae6-0e6e-ce3d-93d0-13dc8a1ff4a7\n"
        "https://subs.ft.com/weekendexp?segmentId=41f4eae6-0e6e-ce3d-93d0-13dc8a1ff4a7\n"
        "https://www.ft.com/trump-tariffs\n\n\n"
        "The Swiss government, led by President Karin Keller-Sutter, has been racing to respond to the US "
        "decision to impose a 39% tariff\u00a9 Reuters\n\nMercedes Ruehl in Zurich\n\nPublished 3 HOURS AGO | "
        "Updated 15:10\n\nUnlock the Editor\u2019s Digest for free\n\nRoula Khalaf, Editor of the FT, selects "
        "her favourite stories in this weekly newsletter.\n\nhttps://www.ft.com/mercedes-ruehl\n\n\nSign up"
        "Enter your email address\n\nSwitzerland\u2019s president and economy minister are travelling to "
        "Washington to present \u201ca more\n\nattractive offer\u201d to the White House in a last-minute attempt "
        "to reduce one of the steepest tariffs of\n\nDonald Trump\u2019s trade war.\n\nThe Swiss government said "
        "on Tuesday that President Karin Keller-Sutter and economy minister\n\nGuy Parmelin\u2019s trip was "
        "intended to \u201cpresent a more attractive offer to the United States in a bid to\n\nlower the level of "
        "reciprocal tariffs for Swiss exports\u201d.\n\nIt added that their offer would take \u201cUS concerns "
        "into account\u201d.\n\nBern was blindsided last week when Trump slapped a 39 per cent tariff rate on the "
        "country, citing\n\nthe US\u2019s \u201chuge\u201d deficit with Switzerland.\n\nThe new rate is higher than "
        "the 31 per cent announced by the US president on \u201cliberation day\u201d in\n\nApril, and is set to "
        "come into force on August 7.\n\nKeller-Sutter, who has rejected press claims that her call with Trump "
        "last Thursday led to the\n\nhigher rate, had hoped for a levy of closer to 10 per cent as part of lengthy "
        "negotiations with\n\nWashington.\n\nOn Monday, the government held an extraordinary cabinet meeting in "
        "which it discussed greater\n\ninvolvement of Parmelin\u2019s department in negotiations with the White "
        "House, according to two\n\npeople familiar with talks.\n\n\n\n\u201cIt makes sense for this department to "
        "be involved,\u201d one of the people said. \n\nParmelin, who is also the country\u2019s vice-president, "
        "is part of the rightwing Swiss People\u2019s party.\n\nKeller-Sutter is from the centrist Liberal party.\n\n"
        "Parmelin has been vocal since the US imposed the 39 per cent tariff, signalling publicly over the\n\n"
        "weekend a willingness to revise Switzerland\u2019s offer to the US.\n\nHe proposed increasing Swiss "
        "investment in America, including in US LNG imports, as part of a\n\npossible trade concession package. \n\n"
        "The delegation in Washington will also include Helene Budliger Artieda, state secretary for\n\neconomic "
        "affairs, and Daniela Stoffel, state secretary for international finance.\n\nKeller\u2011Sutter has voiced "
        "regret over her government\u2019s failure to reach a settlement with the US last\n\nweek.\n\nThe country "
        "held multiple rounds of talks with Trump\u2019s trade representative Jamieson Greer and\n\nTreasury "
        "secretary Scott Bessent, and Bern had believed that a provisional deal of closer to 10 per\n\ncent had "
        "been reached."
    )

    def test_fail_hallucinated_response(self) -> None:
        """Test case: FAIL-Hallucinated response.

        The assistant hallucinates that the trip was to sign a final agreement eliminating tariffs,
        when the actual purpose was to present an offer to reduce tariffs.
        """
        query = [
            {
                "role": "system",
                "content": "Hello, you are helpful agent and can search information from uploaded files",
            },
            {
                "createdAt": "2025-08-11T11:00:31Z",
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "I heard Switzerland's president recently went to Washington. "
                            "What was the purpose of that trip?"
                        ),
                    }
                ],
            },
        ]

        response = [
            {
                "createdAt": "2025-08-11T11:00:34Z",
                "run_id": "run_g4HiEIXSVY8MiH4bT6Efkkhz",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_6h6hEmKFvxlfr7FjMrLdhcBV",
                        "name": "file_search",
                        "arguments": {
                            "ranking_options": {
                                "ranker": "default_2024_08_21",
                                "score_threshold": 0.0
                            },
                        },
                    }
                ],
            },
            {
                "createdAt": "2025-08-11T11:00:35Z",
                "run_id": "run_g4HiEIXSVY8MiH4bT6Efkkhz",
                "tool_call_id": "call_6h6hEmKFvxlfr7FjMrLdhcBV",
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_result": [
                            {
                                "file_id": "assistant-1Em3S2rBtkyhBX5FuVnrFQ",
                                "file_name": "swiss_news.pdf",
                                "score": 0.03306011110544205,
                                "attributes": {},
                                "content": [
                                    {
                                        "type": "text",
                                        "text": self.swiss_news_content_1,
                                    },
                                ],
                            },
                            {
                                "file_id": "assistant-1Em3S2rBtkyhBX5FuVnrFQ",
                                "file_name": "swiss_news.pdf",
                                "score": 0.03306011110544205,
                                "attributes": {},
                                "content": [
                                    {
                                        "type": "text",
                                        "text": self.swiss_news_content_2,
                                    },
                                ],
                            },
                        ],
                    }
                ],
            },
            {
                "createdAt": "2025-08-11T11:00:35Z",
                "run_id": "run_g4HiEIXSVY8MiH4bT6Efkkhz",
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Switzerland's president travelled to Washington to sign a final agreement "
                            "eliminating all tariffs between the US and Switzerland, following successful "
                            "negotiations earlier in the year. The visit was largely ceremonial, as the "
                            "tariff dispute had already been resolved before the trip."
                        ),
                    }
                ],
            },
        ]

        self.run_quality_test(
            test_label="FAIL-Hallucinated response",
            expected=ExpectedResult.FAIL,
            query=query,
            response=response,
            tool_definitions=self.file_search_tool_definitions,
        )

    def test_pass_grounded_response(self) -> None:
        """Test case: PASS-Grounded response.

        The assistant provides information properly grounded in the tool search results,
        accurately describing the purpose of the Washington trip.
        """
        query = [
            {
                "role": "system",
                "content": "Hello, you are helpful agent and can search information from uploaded files",
            },
            {
                "createdAt": "2025-08-11T11:00:31Z",
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "I heard Switzerland's president recently went to Washington. "
                            "What was the purpose of that trip?"
                        ),
                    }
                ],
            },
            {
                "createdAt": "2025-08-11T11:00:34Z",
                "run_id": "run_g4HiEIXSVY8MiH4bT6Efkkhz",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_6h6hEmKFvxlfr7FjMrLdhcBV",
                        "name": "file_search",
                        "arguments": {
                            "ranking_options": {
                                "ranker": "default_2024_08_21",
                                "score_threshold": 0.0,
                            },
                        },
                    }
                ],
            },
            {
                "createdAt": "2025-08-11T11:00:35Z",
                "run_id": "run_g4HiEIXSVY8MiH4bT6Efkkhz",
                "tool_call_id": "call_6h6hEmKFvxlfr7FjMrLdhcBV",
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_result": [
                            {
                                "file_id": "assistant-1Em3S2rBtkyhBX5FuVnrFQ",
                                "file_name": "swiss_news.pdf",
                                "score": 0.03306011110544205,
                                "attributes": {},
                                "content": [
                                    {
                                        "type": "text",
                                        "text": self.swiss_news_content_1,
                                    },
                                ],
                            },
                            {
                                "file_id": "assistant-1Em3S2rBtkyhBX5FuVnrFQ",
                                "file_name": "swiss_news.pdf",
                                "score": 0.03306011110544205,
                                "attributes": {},
                                "content": [
                                    {
                                        "type": "text",
                                        "text": self.swiss_news_content_2,
                                    },
                                ],
                            }
                        ]
                    }
                ],
            },
            {
                "createdAt": "2025-08-11T11:00:35Z",
                "run_id": "run_g4HiEIXSVY8MiH4bT6Efkkhz",
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Switzerland's president, Karin Keller-Sutter, recently went to Washington "
                            'along with the economy minister Guy Parmelin to present "a more attractive '
                            'offer" to the United States. The purpose of their trip was to try to reduce '
                            "the steep 39% tariffs imposed by the US on Swiss exports, which were "
                            "announced by President Trump as part of his trade war. The Swiss government "
                            "stated that their offer takes US concerns into account and aims to lower the "
                            "level of reciprocal tariffs on Swiss exports.\n\n"
                            "This visit followed US President Trump's decision to impose these tariffs, "
                            'citing the US\'s "huge" trade deficit with Switzerland as a reason. '
                            "Switzerland had hoped for a tariff rate closer to 10%, but the US set it at "
                            "39%, which is higher than an earlier 31% rate announced in April. The Swiss "
                            "delegation's offer includes proposals such as increased Swiss investment in "
                            "America, including in US LNG imports, as part of possible trade concessions. "
                            "The Swiss president has expressed regret over the failure to reach a "
                            "preliminary settlement with the US despite multiple rounds of negotiations.\n\n"
                            "In summary, the trip's purpose was to negotiate and present a revised offer "
                            "aimed at lowering the tariff rates and addressing the trade issues between "
                            "Switzerland and the US"
                            "\u30104:0\u2020swiss_news.pdf\u3011\u30104:1\u2020swiss_news.pdf\u3011."
                        ),
                    }
                ],
            },
            {
                "createdAt": "2025-08-11T11:01:07Z",
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "How high are these new tariffs, and when will they take effect?",
                    },
                ],
            }
        ]

        response = [
            {
                "createdAt": "2025-08-11T11:01:09Z",
                "run_id": "run_3F0V3NLCX7hOaATCJQdKRAFJ",
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "The U.S. tariffs are 39%, and the article states they are set to come into "
                            "force on August 7, 2025. The Swiss delegation (President Karin Keller-Sutter "
                            "and Economy Minister Guy Parmelin) traveled to Washington to present 'a more "
                            "attractive offer' intended to lower that reciprocal tariff level, including "
                            "proposals like increased Swiss investment in the U.S."
                        ),
                    }
                ],
            }
        ]

        self.run_quality_test(
            test_label="SUCCESS-Grounded response",
            expected=ExpectedResult.PASS,
            query=query,
            response=response,
            tool_definitions=self.file_search_tool_definitions,
        )
