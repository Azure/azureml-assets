# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Response Completeness Evaluator with real flow execution."""

import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner, ExpectedResult
from ...builtin.response_completeness.evaluator._response_completeness import ResponseCompletenessEvaluator


@pytest.mark.quality
class TestResponseCompletenessEvaluatorQuality(BaseQualityEvaluatorRunner):
    """
    Quality tests for Response Completeness Evaluator.

    Tests actual LLM evaluation with real flow execution (no mocking).

    Completeness measures how accurately and thoroughly a response represents the information
    provided in the ground truth. Scores range from 1 to 5:
    - 1: Fully incomplete - Contains none of the necessary information
    - 2: Barely complete - Contains only a small portion of required information
    - 3: Moderately complete - Covers about half of the required content
    - 4: Mostly complete - Includes most necessary details with minimal omissions
    - 5: Fully complete - Contains all key information without any omissions
    """

    evaluator_type = ResponseCompletenessEvaluator

    # ==================== PASS CASES (Score >= 3) ====================

    def test_pass_fully_complete_exact_match(self) -> None:
        """Test case: PASS - Fully complete response with exact match (expected score: 5).

        Response contains all information from ground truth without any omissions.
        """
        self.run_quality_test(
            test_label="PASS-fully-complete-exact-match",
            expected=ExpectedResult.PASS,
            ground_truth=(
                "Flu shot can prevent flu-related illnesses. Stay healthy by proper hydration "
                "and moderate exercise. Even a few hours of exercise per week can have long-term "
                "benefits for physical and mental health. This is because physical and mental "
                "health benefits have intricate relationships through behavioral changes. "
                "Scientists are starting to discover them through rigorous studies."
            ),
            response=(
                "Flu shot can prevent flu-related illnesses. Stay healthy by proper hydration "
                "and moderate exercise. Even a few hours of exercise per week can have long-term "
                "benefits for physical and mental health. This is because physical and mental "
                "health benefits have intricate relationships through behavioral changes. "
                "Scientists are starting to discover them through rigorous studies."
            ),
        )

    def test_pass_fully_complete_paraphrased(self) -> None:
        """Test case: PASS - Fully complete with paraphrasing (expected score: 5).

        Response covers all ground truth information using different wording.
        """
        self.run_quality_test(
            test_label="PASS-fully-complete-paraphrased",
            expected=ExpectedResult.PASS,
            ground_truth=(
                "Python is a high-level programming language. It was created by Guido van Rossum "
                "in 1991. Python emphasizes code readability with its use of significant whitespace."
            ),
            response=(
                "Python, a high-level programming language developed by Guido van Rossum in 1991, "
                "focuses on code readability through the use of meaningful indentation and whitespace."
            ),
        )

    def test_pass_fully_complete_different_order(self) -> None:
        """Test case: PASS - Fully complete with different ordering (expected score: 5).

        Response contains all ground truth information but presents it in a different sequence.
        """
        self.run_quality_test(
            test_label="PASS-fully-complete-different-order",
            expected=ExpectedResult.PASS,
            ground_truth=(
                "The water cycle consists of evaporation, condensation, and precipitation. "
                "Water evaporates from oceans and lakes. Vapor condenses into clouds. "
                "Rain falls back to Earth."
            ),
            response=(
                "Rain falls back to Earth in the precipitation stage. Before that, vapor condenses "
                "into clouds. The process begins when water evaporates from oceans and lakes. "
                "These three stages - evaporation, condensation, and precipitation - make up "
                "the water cycle."
            ),
        )

    def test_pass_mostly_complete_minor_omission(self) -> None:
        """Test case: PASS - Mostly complete with minor omission (expected score: 4).

        Response includes most ground truth details but misses one minor element.
        """
        self.run_quality_test(
            test_label="PASS-mostly-complete-minor-omission",
            expected=ExpectedResult.PASS,
            ground_truth=(
                "The Renaissance was a period of cultural rebirth in Europe from the 14th to 17th "
                "century. It began in Italy and spread across Europe. Key features included renewed "
                "interest in classical art, literature, and science. Famous figures include Leonardo "
                "da Vinci, Michelangelo, and Galileo."
            ),
            response=(
                "The Renaissance was a cultural rebirth period from the 14th to 17th century that "
                "started in Italy and spread throughout Europe. It featured renewed interest in "
                "classical art, literature, and science. Notable figures include Leonardo da Vinci "
                "and Michelangelo."
            ),
        )

    @pytest.mark.flaky(reruns=3)
    def test_pass_moderately_complete_at_threshold(self) -> None:
        """Test case: PASS - Moderately complete at threshold (expected score: 3).

        Response covers about half of the ground truth content.
        """
        self.run_quality_test(
            test_label="PASS-moderately-complete-at-threshold",
            expected=ExpectedResult.PASS,
            ground_truth=(
                "Climate change is causing global temperatures to rise. This leads to melting ice "
                "caps and rising sea levels. Extreme weather events are becoming more frequent. "
                "Carbon emissions from fossil fuels are the primary cause. Renewable energy sources "
                "like solar and wind can help mitigate the impact."
            ),
            response=(
                "Climate change is causing global temperatures to rise, resulting in melting ice "
                "caps and rising sea levels. Extreme weather events are also becoming more frequent."
            ),
        )

    def test_pass_complete_with_extra_correct_info(self) -> None:
        """Test case: PASS - Complete with additional correct information (expected score: 5).

        Response contains all ground truth information plus extra relevant details.
        """
        self.run_quality_test(
            test_label="PASS-complete-with-extra-correct-info",
            expected=ExpectedResult.PASS,
            ground_truth=(
                "Photosynthesis is the process by which plants convert sunlight into energy. "
                "It occurs in chloroplasts and produces oxygen as a byproduct."
            ),
            response=(
                "Photosynthesis is the process by which plants convert sunlight into energy. "
                "It occurs in chloroplasts, which contain chlorophyll, the green pigment. "
                "The process produces oxygen as a byproduct, which is released into the atmosphere. "
                "This oxygen is essential for most life on Earth."
            ),
        )

    def test_pass_technical_domain_complete(self) -> None:
        """Test case: PASS - Technical domain with complete coverage (expected score: 5).

        Response accurately covers all technical details from ground truth.
        """
        self.run_quality_test(
            test_label="PASS-technical-domain-complete",
            expected=ExpectedResult.PASS,
            ground_truth=(
                "RESTful APIs use HTTP methods: GET retrieves data, POST creates new resources, "
                "PUT updates existing resources, and DELETE removes resources. Status codes indicate "
                "operation results: 200 for success, 404 for not found, 500 for server errors."
            ),
            response=(
                "RESTful APIs utilize HTTP methods where GET retrieves data, POST creates new "
                "resources, PUT updates existing ones, and DELETE removes resources. The system "
                "uses status codes to communicate results: 200 indicates success, 404 means "
                "not found, and 500 signals server errors."
            ),
        )

    def test_pass_multiple_independent_claims_complete(self) -> None:
        """Test case: PASS - Multiple independent claims all covered (expected score: 5).

        Ground truth has multiple unrelated claims, all covered in response.
        """
        self.run_quality_test(
            test_label="PASS-multiple-independent-claims-complete",
            expected=ExpectedResult.PASS,
            ground_truth=(
                "The Earth revolves around the Sun once per year. Water boils at 100 degrees "
                "Celsius at sea level. The human heart has four chambers. Japan is an island "
                "nation in East Asia."
            ),
            response=(
                "The Earth completes one revolution around the Sun annually. At sea level, water "
                "reaches its boiling point at 100 degrees Celsius. The human heart consists of "
                "four chambers. Japan is located in East Asia and is an island nation."
            ),
        )

    def test_pass_short_ground_truth_complete(self) -> None:
        """Test case: PASS - Short ground truth fully covered (expected score: 5).

        Tests completeness with brief ground truth statement.
        """
        self.run_quality_test(
            test_label="PASS-short-ground-truth-complete",
            expected=ExpectedResult.PASS,
            ground_truth="The capital of France is Paris.",
            response="Paris is the capital of France.",
        )

    # ==================== FAIL CASES (Score < 3) ====================

    def test_fail_fully_incomplete(self) -> None:
        """Test case: FAIL - Fully incomplete response (expected score: 1).

        Response contains none of the necessary information from ground truth.
        """
        self.run_quality_test(
            test_label="FAIL-fully-incomplete",
            expected=ExpectedResult.FAIL,
            ground_truth=(
                "Flu shot can prevent flu-related illnesses. Stay healthy by proper hydration "
                "and moderate exercise. Even a few hours of exercise per week can have long-term "
                "benefits for physical and mental health."
            ),
            response=(
                "Flu shot cannot cure cancer. Stay healthy requires sleeping exactly 8 hours a day. "
                "A few hours of exercise per week will have little benefits for physical and mental health."
            ),
        )

    def test_fail_barely_complete_small_portion(self) -> None:
        """Test case: FAIL - Barely complete with small portion (expected score: 2).

        Response contains only a tiny fraction of the ground truth information.
        """
        self.run_quality_test(
            test_label="FAIL-barely-complete-small-portion",
            expected=ExpectedResult.FAIL,
            ground_truth=(
                "The American Revolution began in 1775 and ended in 1783. It was fought between "
                "Great Britain and the thirteen American colonies. Key events include the Boston "
                "Tea Party, the Declaration of Independence, and the Battle of Yorktown. The war "
                "resulted in American independence and the formation of the United States."
            ),
            response=(
                "The American Revolution was a war between Great Britain and some colonies. "
                "It happened a long time ago."
            ),
        )

    def test_fail_wrong_information(self) -> None:
        """Test case: FAIL - Response with incorrect information (expected score: 1).

        Response contradicts ground truth with factually wrong statements.
        """
        self.run_quality_test(
            test_label="FAIL-wrong-information",
            expected=ExpectedResult.FAIL,
            ground_truth=(
                "The Moon orbits the Earth. It takes approximately 27.3 days to complete one orbit. "
                "The Moon is Earth's only natural satellite."
            ),
            response=(
                "The Moon orbits the Sun directly. It takes 365 days to complete one orbit. "
                "The Moon is one of several satellites orbiting Earth."
            ),
        )

    def test_fail_unrelated_response(self) -> None:
        """Test case: FAIL - Completely unrelated response (expected score: 1).

        Response addresses a completely different topic than ground truth.
        """
        self.run_quality_test(
            test_label="FAIL-unrelated-response",
            expected=ExpectedResult.FAIL,
            ground_truth=(
                "DNA stands for deoxyribonucleic acid. It contains genetic instructions for the "
                "development and functioning of living organisms. DNA is structured as a double helix."
            ),
            response=(
                "Pizza is a popular Italian dish. It typically consists of dough, tomato sauce, "
                "and cheese. Many people enjoy pizza with various toppings."
            ),
        )

    @pytest.mark.flaky(reruns=3)
    def test_fail_partial_with_significant_gaps(self) -> None:
        """Test case: FAIL - Partial completeness with significant gaps (expected score: 2).

        Response covers less than half of the ground truth, missing major points.
        """
        self.run_quality_test(
            test_label="FAIL-partial-with-significant-gaps",
            expected=ExpectedResult.FAIL,
            ground_truth=(
                "Machine learning is a subset of artificial intelligence. It enables systems to "
                "learn from data without explicit programming. Common types include supervised "
                "learning, unsupervised learning, and reinforcement learning. Applications include "
                "image recognition, natural language processing, and recommendation systems."
            ),
            response=(
                "Machine learning is related to artificial intelligence. It has various applications "
                "in technology."
            ),
        )

    def test_fail_multiple_claims_mostly_missing(self) -> None:
        """Test case: FAIL - Multiple claims with most missing (expected score: 2).

        Ground truth has multiple independent claims, response covers only one.
        """
        self.run_quality_test(
            test_label="FAIL-multiple-claims-mostly-missing",
            expected=ExpectedResult.FAIL,
            ground_truth=(
                "The Great Wall of China is over 13,000 miles long. It was built over several "
                "centuries by different dynasties. The wall was constructed to protect against "
                "invasions. It is one of the Seven Wonders of the Medieval World. Today it is "
                "a major tourist attraction."
            ),
            response="The Great Wall of China is a very old structure.",
        )

    # ==================== EDGE CASES ====================

    def test_edge_case_response_paraphrases_with_synonyms(self) -> None:
        """Test case: Edge - Response uses extensive synonyms (expected score: 5).

        Tests whether evaluator recognizes completeness despite extensive paraphrasing.
        """
        self.run_quality_test(
            test_label="EDGE-response-paraphrases-with-synonyms",
            expected=ExpectedResult.PASS,
            ground_truth=(
                "Regular exercise improves cardiovascular health and reduces the risk of heart disease."
            ),
            response=(
                "Consistent physical activity enhances heart function and lowers the likelihood of "
                "cardiac conditions."
            ),
        )

    def test_edge_case_response_elaborates_but_complete(self) -> None:
        """Test case: Edge - Response elaborates but covers all ground truth (expected score: 5).

        Response provides additional context and examples while maintaining completeness.
        """
        self.run_quality_test(
            test_label="EDGE-response-elaborates-but-complete",
            expected=ExpectedResult.PASS,
            ground_truth="Antibiotics treat bacterial infections but not viral infections.",
            response=(
                "Antibiotics are medications specifically designed to treat bacterial infections "
                "by killing bacteria or preventing their growth. However, they are ineffective "
                "against viral infections such as the common cold or flu, because viruses have "
                "a completely different structure and replication mechanism than bacteria."
            ),
        )

    def test_edge_case_numerical_data_accuracy(self) -> None:
        """Test case: Edge - Numerical data must be accurate (expected score: depends on accuracy).

        Tests completeness with specific numerical values that must match.
        """
        self.run_quality_test(
            test_label="EDGE-numerical-data-accuracy",
            expected=ExpectedResult.FAIL,
            ground_truth=(
                "The speed of light in vacuum is 299,792,458 meters per second. This is often "
                "approximated as 300,000 kilometers per second."
            ),
            response=(
                "The speed of light in vacuum is approximately 300,000 meters per second, which "
                "is about 300 kilometers per second."
            ),
        )

    def test_edge_case_response_has_correct_subset(self) -> None:
        """Test case: Edge - Response is a correct subset (expected score: 3-4).

        Response covers multiple points accurately but misses some.
        """
        self.run_quality_test(
            test_label="EDGE-response-has-correct-subset",
            expected=ExpectedResult.PASS,
            ground_truth=(
                "Shakespeare wrote 37 plays, 154 sonnets, and several narrative poems. "
                "His works include tragedies, comedies, and histories. Famous plays include "
                "Hamlet, Romeo and Juliet, and Macbeth."
            ),
            response=(
                "Shakespeare wrote 37 plays and 154 sonnets. His works span multiple genres "
                "including tragedies, comedies, and histories."
            ),
        )

    @pytest.mark.flaky(reruns=3)
    def test_edge_case_implicit_vs_explicit_information(self) -> None:
        """Test case: Edge - Implicit information in response (expected score: 4-5).

        Ground truth is explicit, response conveys same information implicitly.
        """
        self.run_quality_test(
            test_label="EDGE-implicit-vs-explicit-information",
            expected=ExpectedResult.PASS,
            ground_truth="Birds are warm-blooded vertebrates with feathers and wings.",
            response=(
                "Birds belong to the class Aves, characterized by being endothermic vertebrates "
                "covered in feathers and possessing forelimbs modified into wings."
            ),
        )

    def test_edge_case_long_ground_truth_with_multiple_paragraphs(self) -> None:
        """Test case: Edge - Long multi-paragraph ground truth (expected score: varies).

        Tests completeness with extensive ground truth requiring comprehensive response.
        """
        self.run_quality_test(
            test_label="EDGE-long-ground-truth-multiple-paragraphs",
            expected=ExpectedResult.PASS,
            ground_truth=(
                "Quantum computing uses quantum bits or qubits instead of classical bits. "
                "Unlike classical bits which are either 0 or 1, qubits can exist in superposition, "
                "representing both states simultaneously. This allows quantum computers to process "
                "multiple possibilities at once.\n\n"
                "Quantum entanglement is another key principle, where qubits become correlated "
                "and the state of one instantly affects the other, regardless of distance. "
                "This enables quantum computers to solve certain problems exponentially faster "
                "than classical computers.\n\n"
                "Current challenges include maintaining quantum coherence and reducing quantum "
                "decoherence. Quantum computers require extremely cold temperatures and isolation "
                "from environmental interference."
            ),
            response=(
                "Quantum computing operates on quantum bits (qubits) rather than classical bits. "
                "Qubits can exist in superposition, simultaneously representing 0 and 1, enabling "
                "parallel processing of multiple possibilities. Another fundamental principle is "
                "quantum entanglement, where qubits become correlated such that one's state "
                "instantaneously influences another's regardless of separation. This allows quantum "
                "computers to solve specific problems exponentially faster than classical systems. "
                "However, major challenges exist in maintaining quantum coherence and minimizing "
                "decoherence. These systems require extremely low temperatures and must be isolated "
                "from environmental disturbances."
            ),
        )

    @pytest.mark.flaky(reruns=3)
    def test_edge_case_mixed_accuracy_some_correct_some_wrong(self) -> None:
        """Test case: Edge - Mixed accuracy (expected score: 2-3).

        Response gets some facts right but includes incorrect information for other facts.
        """
        self.run_quality_test(
            test_label="EDGE-mixed-accuracy-some-correct-some-wrong",
            expected=ExpectedResult.PASS_WITH_SCORE_3,
            ground_truth=(
                "The human body has 206 bones in adults. The smallest bone is the stapes in the ear. "
                "The largest bone is the femur in the thigh. Bones provide structure and protection."
            ),
            response=(
                "The human body has 206 bones in adults. The smallest bone is in the finger. "
                "The largest bone is the femur in the thigh. Bones help us move around."
            ),
        )

    def test_edge_case_list_format_completeness(self) -> None:
        """Test case: Edge - List format with missing items (expected score: 3-4).

        Ground truth is a list, response covers most but not all items.
        """
        self.run_quality_test(
            test_label="EDGE-list-format-completeness",
            expected=ExpectedResult.PASS,
            ground_truth=(
                "The five basic taste sensations are: sweet, sour, salty, bitter, and umami. "
                "Sweet detects sugars. Sour detects acids. Salty detects sodium. Bitter detects "
                "potentially harmful substances. Umami detects amino acids."
            ),
            response=(
                "The five basic taste sensations are sweet, sour, salty, bitter, and umami. "
                "Sweet detects sugars, sour detects acids, and salty detects sodium. "
                "Umami detects amino acids."
            ),
        )

    def test_edge_case_cause_and_effect_completeness(self) -> None:
        """Test case: Edge - Cause and effect relationship (expected score: varies).

        Tests completeness when ground truth describes causal relationships.
        """
        self.run_quality_test(
            test_label="EDGE-cause-and-effect-completeness",
            expected=ExpectedResult.PASS,
            ground_truth=(
                "Deforestation leads to loss of biodiversity because it destroys habitats. "
                "This also contributes to climate change by reducing carbon absorption. "
                "Additionally, it increases soil erosion."
            ),
            response=(
                "Deforestation causes biodiversity loss through habitat destruction. It contributes "
                "to climate change by decreasing the earth's capacity to absorb carbon. "
                "It also results in increased soil erosion."
            ),
        )

    def test_edge_case_one_word_responses(self) -> None:
        """Test case: Edge - Very brief ground truth and response (expected score: 5).

        Tests completeness with minimal text.
        """
        self.run_quality_test(
            test_label="EDGE-one-word-responses",
            expected=ExpectedResult.PASS,
            ground_truth="Paris.",
            response="Paris",
        )

    def test_edge_case_response_summarizes_concisely(self) -> None:
        """Test case: Edge - Concise summary vs detailed ground truth (expected score: 4-5).

        Tests whether concise but complete response scores well.
        """
        self.run_quality_test(
            test_label="EDGE-response-summarizes-concisely",
            expected=ExpectedResult.PASS,
            ground_truth=(
                "Vaccines work by introducing a weakened or inactive form of a pathogen into the body. "
                "This stimulates the immune system to produce antibodies without causing the disease. "
                "The immune system remembers the pathogen, so if exposed to it again, it can respond "
                "quickly and effectively. This provides immunity against future infections."
            ),
            response=(
                "Vaccines introduce weakened pathogens to stimulate antibody production without "
                "causing disease. The immune system develops memory, enabling rapid response to "
                "future exposures and providing immunity."
            ),
        )
