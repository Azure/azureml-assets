# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Coherence Evaluator with real flow execution."""

import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner, ExpectedResult
from ...builtin.coherence.evaluator._coherence import CoherenceEvaluator
from .common_test_data import ResponseTexts


@pytest.mark.quality
class TestCoherenceEvaluatorQuality(BaseQualityEvaluatorRunner):
    """
    Quality tests for Coherence Evaluator.

    Tests actual LLM evaluation with real flow execution (no mocking).
    
    Coherence measures the logical and orderly presentation of ideas in a response,
    allowing the reader to easily follow and understand the writer's train of thought.
    
    Rating scale (default threshold=3):
    - 1: Incoherent - disjointed words/phrases, no logical connection
    - 2: Poorly coherent - fragmented sentences, limited connection
    - 3: Partially coherent - some issues with logical flow/organization
    - 4: Coherent - logically organized, clear connections, smooth flow
    - 5: Highly coherent - exceptional organization, sophisticated flow
    """

    evaluator_type = CoherenceEvaluator

    # ==================== PASS CASES (Score >= 3) ====================

    def test_pass_highly_coherent_response(self) -> None:
        """Test case: PASS - Highly coherent response (expected score: 5).
        
        Response demonstrates exceptional organization with sophisticated flow,
        excellent use of transitional phrases, and clear connections between concepts.
        """
        self.run_quality_test(
            test_label="PASS-highly-coherent-response",
            expected=ExpectedResult.PASS,
            query="Analyze the economic impacts of climate change on coastal cities.",
            response=ResponseTexts.CLIMATE_CHANGE_ANALYSIS,
        )

    def test_pass_coherent_response_with_clear_structure(self) -> None:
        """Test case: PASS - Coherent response with clear structure (expected score: 4).
        
        Response is well-organized with clear connections between sentences,
        appropriate transitions, and smooth flow.
        """
        self.run_quality_test(
            test_label="PASS-coherent-clear-structure",
            expected=ExpectedResult.PASS,
            query="What is the water cycle and how does it work?",
            response=ResponseTexts.WATER_CYCLE_EXPLANATION,
        )

    @pytest.mark.flaky(reruns=3)
    def test_pass_partially_coherent_at_threshold(self) -> None:
        """Test case: PASS - Partially coherent response at threshold (expected score: 3).
        
        Response addresses the question with relevant information but has some
        issues with logical flow. Connections between sentences may be unclear.
        """
        self.run_quality_test(
            test_label="PASS-partially-coherent-at-threshold",
            expected=ExpectedResult.PASS,
            query="What causes earthquakes?",
            response=(
                "Earthquakes happen when tectonic plates move suddenly. The Earth's crust is made "
                "of these plates. Energy builds up over time at fault lines. When it releases, the "
                "ground shakes. This can cause damage to buildings. Scientists use seismographs to "
                "measure them. The magnitude tells us how strong it was."
            ),
        )

    def test_pass_short_but_coherent_response(self) -> None:
        """Test case: PASS - Short but coherent response (edge case).
        
        A brief response that still maintains logical flow and directly addresses
        the question in a coherent manner.
        """
        self.run_quality_test(
            test_label="PASS-short-but-coherent-response",
            expected=ExpectedResult.PASS,
            query="What is the capital of France?",
            response=(
                "The capital of France is Paris. It is located in the northern part of the "
                "country along the Seine River and serves as the nation's political, economic, "
                "and cultural center."
            ),
        )

    def test_pass_technical_coherent_response(self) -> None:
        """Test case: PASS - Technical but coherent response.
        
        Response uses technical language but maintains logical structure and
        clear connections between concepts.
        """
        self.run_quality_test(
            test_label="PASS-technical-coherent-response",
            expected=ExpectedResult.PASS,
            query="Explain how a compiler works.",
            response=(
                "A compiler is a program that translates source code written in a high-level "
                "programming language into machine code that a computer can execute. The compilation "
                "process typically involves several phases. First, the lexical analyzer breaks the "
                "source code into tokens. Next, the parser analyzes the grammatical structure and "
                "builds a syntax tree. The semantic analyzer then checks for type errors and other "
                "semantic issues. Following this, the intermediate code generator creates an "
                "abstract representation. Finally, the code optimizer improves efficiency before "
                "the code generator produces the target machine code. Each phase builds upon the "
                "previous one, transforming the code step by step."
            ),
        )

    # ==================== FAIL CASES (Score < 3) ====================

    def test_fail_incoherent_response(self) -> None:
        """Test case: FAIL - Completely incoherent response (expected score: 1).
        
        Response consists of disjointed words or phrases that do not form
        meaningful sentences with no logical connection to the question.
        """
        self.run_quality_test(
            test_label="FAIL-incoherent-response",
            expected=ExpectedResult.FAIL,
            query="What are the benefits of renewable energy?",
            response="Wind sun green jump apple silence over mountain blue quickly.",
        )

    def test_fail_poorly_coherent_fragmented(self) -> None:
        """Test case: FAIL - Poorly coherent fragmented response (expected score: 2).
        
        Response shows minimal coherence with fragmented sentences and
        limited logical structure.
        """
        self.run_quality_test(
            test_label="FAIL-poorly-coherent-fragmented",
            expected=ExpectedResult.FAIL,
            query="How does vaccination work?",
            response="Vaccines protect disease. Immune system fight. Health better. Antibodies made. Doctor gives shot.",
        )

    def test_fail_response_with_random_topic_jumps(self) -> None:
        """Test case: FAIL - Response with random topic jumps.
        
        Response starts addressing the question but randomly jumps to
        unrelated topics without any logical transitions.
        """
        self.run_quality_test(
            test_label="FAIL-random-topic-jumps",
            expected=ExpectedResult.FAIL,
            query="What is photosynthesis?",
            response=(
                "Photosynthesis is a process used by plants. My grandmother has a garden. "
                "The stock market was volatile yesterday. Plants use sunlight for energy. "
                "I need to buy groceries later. Chlorophyll is green. The movie last night was boring."
            ),
        )

    def test_fail_contradictory_statements(self) -> None:
        """Test case: FAIL - Response with contradictory statements.
        
        Response contains self-contradicting information that breaks
        logical coherence.
        """
        self.run_quality_test(
            test_label="FAIL-contradictory-statements",
            expected=ExpectedResult.FAIL,
            query="Is the Earth round or flat?",
            response=(
                "The Earth is definitely round, as proven by countless scientific observations. "
                "However, the Earth is actually flat, which is why we don't fall off. "
                "The spherical shape of Earth is confirmed by satellite images. "
                "But flat Earth theory is correct because we can't feel the curvature."
            ),
        )

    # ==================== EDGE CASES ====================

    def test_edge_case_list_format_coherent(self) -> None:
        """Test case: Edge case - List format response.
        
        Response uses a list format but maintains coherence through
        logical ordering and clear connections.
        """
        self.run_quality_test(
            test_label="EDGE-list-format-coherent",
            expected=ExpectedResult.PASS,
            query="What are the steps to make a cup of tea?",
            response=(
                "To make a cup of tea, follow these steps in order: First, fill a kettle with "
                "fresh water and bring it to a boil. While waiting, place a tea bag in your cup. "
                "Once the water has boiled, pour it over the tea bag and let it steep for 3-5 "
                "minutes depending on your preferred strength. After steeping, remove the tea bag "
                "and add milk or sugar if desired. Finally, stir well and enjoy your tea while "
                "it's still hot."
            ),
        )

    def test_edge_case_starts_coherent_loses_coherence(self) -> None:
        """Test case: Edge case - Starts coherent but loses coherence.
        
        Response begins with good coherence but progressively becomes
        more disjointed and loses logical flow.
        """
        self.run_quality_test(
            test_label="EDGE-starts-coherent-loses-coherence",
            expected=ExpectedResult.FAIL,
            query="Explain the importance of sleep for health.",
            response=(
                "Sleep is essential for maintaining good health and well-being. During sleep, "
                "your body repairs tissues and consolidates memories. Purple elephants dancing. "
                "The immune system strengthens during rest. Calculator pizza umbrella forgotten. "
                "Seven hours minimum recommended sideways. Brain needs sleep function properly "
                "jumping clouds."
            ),
        )

    def test_edge_case_repetitive_but_coherent(self) -> None:
        """Test case: Edge case - Repetitive but coherent response.
        
        Response repeats ideas but maintains logical coherence throughout.
        Tests whether repetition affects coherence scoring.
        """
        # TODO: Decide if this should pass or fail based on repetition handling
        self.run_quality_test(
            test_label="EDGE-repetitive-but-coherent",
            expected=ExpectedResult.PASS,
            query="Why is exercise important?",
            response=(
                "Exercise is important for maintaining good health. Physical activity helps "
                "keep your body healthy. When you exercise regularly, you improve your health. "
                "Being active through exercise supports your overall well-being. Regular physical "
                "activity is beneficial for health. In summary, exercise plays a crucial role in "
                "staying healthy."
            ),
        )

    def test_edge_case_complex_sentence_structure(self) -> None:
        """Test case: Edge case - Complex nested sentence structure.
        
        Response uses complex, nested sentences that are still coherent
        but require careful reading to follow.
        """
        self.run_quality_test(
            test_label="EDGE-complex-sentence-structure",
            expected=ExpectedResult.PASS,
            query="How do international trade agreements affect local economies?",
            response=(
                "International trade agreements, which are negotiated between countries to "
                "reduce barriers such as tariffs and quotas, can significantly impact local "
                "economies in ways that, while sometimes beneficial for consumers who gain "
                "access to cheaper imported goods, may also pose challenges for local industries "
                "that, unable to compete with lower-cost foreign producers, might face declining "
                "sales, leading to job losses in sectors that were previously protected, though "
                "proponents argue that the overall economic efficiency gains and new export "
                "opportunities often outweigh these localized negative effects in the long term."
            ),
        )

    def test_edge_case_single_sentence_response(self) -> None:
        """Test case: Edge case - Single sentence response.
        
        A minimal single-sentence response that is coherent but very brief.
        """
        self.run_quality_test(
            test_label="EDGE-single-sentence-response",
            expected=ExpectedResult.PASS,
            query="What color is the sky?",
            response="The sky appears blue during the day due to the scattering of sunlight by the atmosphere.",
        )

    def test_edge_case_one_word_correct_response(self) -> None:
        """Test case: Edge case - One word correct response.
        
        A single word response that directly and coherently answers the question,
        testing the minimum possible coherent response.
        """
        self.run_quality_test(
            test_label="EDGE-one-word-correct-response",
            expected=ExpectedResult.PASS,
            query="What is the capital of France?",
            response="Paris.",
        )

    def test_edge_case_two_word_correct_response(self) -> None:
        """Test case: Edge case - Two word correct response.
        
        A two-word response that coherently addresses the question,
        testing minimal but valid coherent expression.
        """
        # TODO: Test fails - evaluator scores 2 (Poorly coherent) for minimal responses.
        # Reason: "The response lacks clarity and coherence due to its incomplete structure."
        # Decision needed: Should minimal but correct responses pass coherence, or is this expected behavior?
        self.run_quality_test(
            test_label="EDGE-two-word-correct-response",
            expected=ExpectedResult.PASS,
            query="Is Paris the capital of France?",
            response="Yes it is.",
        )