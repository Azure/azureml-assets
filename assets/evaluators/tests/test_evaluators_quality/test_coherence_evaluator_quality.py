# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Coherence Evaluator with real flow execution."""

import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner
from ...builtin.coherence.evaluator._coherence import CoherenceEvaluator


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

    def test_pass_highly_coherent_response(self):
        """Test case: PASS - Highly coherent response (expected score: 5).
        
        Response demonstrates exceptional organization with sophisticated flow,
        excellent use of transitional phrases, and clear connections between concepts.
        """
        query = "Analyze the economic impacts of climate change on coastal cities."
        
        response = (
            "Climate change significantly affects the economies of coastal cities through "
            "rising sea levels, increased flooding, and more intense storms. These environmental "
            "changes can damage infrastructure, disrupt businesses, and lead to costly repairs. "
            "For instance, frequent flooding can hinder transportation and commerce, while the "
            "threat of severe weather may deter investment and tourism. Consequently, cities may "
            "face increased expenses for disaster preparedness and mitigation efforts, straining "
            "municipal budgets and impacting economic growth. Furthermore, property values in "
            "vulnerable areas may decline, reducing tax revenues and affecting homeowners' wealth. "
            "In response, many coastal cities are investing in resilient infrastructure and "
            "exploring innovative solutions to protect their economies for the future."
        )
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "PASS-Highly coherent response")
        
        self.assert_pass(result_data)

    def test_pass_coherent_response_with_clear_structure(self):
        """Test case: PASS - Coherent response with clear structure (expected score: 4).
        
        Response is well-organized with clear connections between sentences,
        appropriate transitions, and smooth flow.
        """
        query = "What is the water cycle and how does it work?"
        
        response = (
            "The water cycle is the continuous movement of water on Earth through several "
            "key processes. First, water evaporates from oceans, lakes, and rivers when heated "
            "by the sun. This water vapor rises into the atmosphere where it cools and condenses "
            "to form clouds. When the water droplets in clouds become heavy enough, they fall "
            "back to Earth as precipitation, such as rain or snow. This water then collects in "
            "bodies of water or seeps into the ground, eventually making its way back to the "
            "oceans to begin the cycle again. This continuous process is essential for "
            "distributing water resources across the planet."
        )
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "PASS-Coherent response with clear structure")
        
        self.assert_pass(result_data)

    @pytest.mark.flaky(reruns=3)
    def test_pass_partially_coherent_at_threshold(self):
        """Test case: PASS - Partially coherent response at threshold (expected score: 3).
        
        Response addresses the question with relevant information but has some
        issues with logical flow. Connections between sentences may be unclear.
        """
        query = "What causes earthquakes?"
        
        response = (
            "Earthquakes happen when tectonic plates move suddenly. The Earth's crust is made "
            "of these plates. Energy builds up over time at fault lines. When it releases, the "
            "ground shakes. This can cause damage to buildings. Scientists use seismographs to "
            "measure them. The magnitude tells us how strong it was."
        )
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "PASS-Partially coherent at threshold")
        
        self.assert_pass(result_data)

    def test_pass_short_but_coherent_response(self):
        """Test case: PASS - Short but coherent response (edge case).
        
        A brief response that still maintains logical flow and directly addresses
        the question in a coherent manner.
        """
        query = "What is the capital of France?"
        
        response = (
            "The capital of France is Paris. It is located in the northern part of the "
            "country along the Seine River and serves as the nation's political, economic, "
            "and cultural center."
        )
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "PASS-Short but coherent response")
        
        self.assert_pass(result_data)

    def test_pass_technical_coherent_response(self):
        """Test case: PASS - Technical but coherent response.
        
        Response uses technical language but maintains logical structure and
        clear connections between concepts.
        """
        query = "Explain how a compiler works."
        
        response = (
            "A compiler is a program that translates source code written in a high-level "
            "programming language into machine code that a computer can execute. The compilation "
            "process typically involves several phases. First, the lexical analyzer breaks the "
            "source code into tokens. Next, the parser analyzes the grammatical structure and "
            "builds a syntax tree. The semantic analyzer then checks for type errors and other "
            "semantic issues. Following this, the intermediate code generator creates an "
            "abstract representation. Finally, the code optimizer improves efficiency before "
            "the code generator produces the target machine code. Each phase builds upon the "
            "previous one, transforming the code step by step."
        )
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "PASS-Technical coherent response")
        
        self.assert_pass(result_data)

    # ==================== FAIL CASES (Score < 3) ====================

    def test_fail_incoherent_response(self):
        """Test case: FAIL - Completely incoherent response (expected score: 1).
        
        Response consists of disjointed words or phrases that do not form
        meaningful sentences with no logical connection to the question.
        """
        query = "What are the benefits of renewable energy?"
        
        response = "Wind sun green jump apple silence over mountain blue quickly."
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "FAIL-Incoherent response")
        
        self.assert_fail(result_data)

    def test_fail_poorly_coherent_fragmented(self):
        """Test case: FAIL - Poorly coherent fragmented response (expected score: 2).
        
        Response shows minimal coherence with fragmented sentences and
        limited logical structure.
        """
        query = "How does vaccination work?"
        
        response = "Vaccines protect disease. Immune system fight. Health better. Antibodies made. Doctor gives shot."
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "FAIL-Poorly coherent fragmented")
        
        self.assert_fail(result_data)

    def test_fail_response_with_random_topic_jumps(self):
        """Test case: FAIL - Response with random topic jumps.
        
        Response starts addressing the question but randomly jumps to
        unrelated topics without any logical transitions.
        """
        query = "What is photosynthesis?"
        
        response = (
            "Photosynthesis is a process used by plants. My grandmother has a garden. "
            "The stock market was volatile yesterday. Plants use sunlight for energy. "
            "I need to buy groceries later. Chlorophyll is green. The movie last night was boring."
        )
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "FAIL-Response with random topic jumps")
        
        self.assert_fail(result_data)

    def test_fail_contradictory_statements(self):
        """Test case: FAIL - Response with contradictory statements.
        
        Response contains self-contradicting information that breaks
        logical coherence.
        """
        query = "Is the Earth round or flat?"
        
        response = (
            "The Earth is definitely round, as proven by countless scientific observations. "
            "However, the Earth is actually flat, which is why we don't fall off. "
            "The spherical shape of Earth is confirmed by satellite images. "
            "But flat Earth theory is correct because we can't feel the curvature."
        )
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "FAIL-Contradictory statements")
        
        self.assert_fail(result_data)

    # ==================== EDGE CASES ====================

    def test_edge_case_list_format_coherent(self):
        """Test case: Edge case - List format response.
        
        Response uses a list format but maintains coherence through
        logical ordering and clear connections.
        """
        query = "What are the steps to make a cup of tea?"
        
        response = (
            "To make a cup of tea, follow these steps in order: First, fill a kettle with "
            "fresh water and bring it to a boil. While waiting, place a tea bag in your cup. "
            "Once the water has boiled, pour it over the tea bag and let it steep for 3-5 "
            "minutes depending on your preferred strength. After steeping, remove the tea bag "
            "and add milk or sugar if desired. Finally, stir well and enjoy your tea while "
            "it's still hot."
        )
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "Edge case-List format coherent")
        
        self.assert_pass(result_data)

    def test_edge_case_starts_coherent_loses_coherence(self):
        """Test case: Edge case - Starts coherent but loses coherence.
        
        Response begins with good coherence but progressively becomes
        more disjointed and loses logical flow.
        """
        query = "Explain the importance of sleep for health."
        
        response = (
            "Sleep is essential for maintaining good health and well-being. During sleep, "
            "your body repairs tissues and consolidates memories. Purple elephants dancing. "
            "The immune system strengthens during rest. Calculator pizza umbrella forgotten. "
            "Seven hours minimum recommended sideways. Brain needs sleep function properly "
            "jumping clouds."
        )
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "Edge case-Starts coherent loses coherence")
        
        self.assert_fail(result_data)

    def test_edge_case_repetitive_but_coherent(self):
        """Test case: Edge case - Repetitive but coherent response.
        
        Response repeats ideas but maintains logical coherence throughout.
        Tests whether repetition affects coherence scoring.
        """
        query = "Why is exercise important?"
        
        response = (
            "Exercise is important for maintaining good health. Physical activity helps "
            "keep your body healthy. When you exercise regularly, you improve your health. "
            "Being active through exercise supports your overall well-being. Regular physical "
            "activity is beneficial for health. In summary, exercise plays a crucial role in "
            "staying healthy."
        )
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "Edge case-Repetitive but coherent")
        
        # TODO: Decide if this should pass or fail based on repetition handling
        self.assert_pass(result_data)

    def test_edge_case_complex_sentence_structure(self):
        """Test case: Edge case - Complex nested sentence structure.
        
        Response uses complex, nested sentences that are still coherent
        but require careful reading to follow.
        """
        query = "How do international trade agreements affect local economies?"
        
        response = (
            "International trade agreements, which are negotiated between countries to "
            "reduce barriers such as tariffs and quotas, can significantly impact local "
            "economies in ways that, while sometimes beneficial for consumers who gain "
            "access to cheaper imported goods, may also pose challenges for local industries "
            "that, unable to compete with lower-cost foreign producers, might face declining "
            "sales, leading to job losses in sectors that were previously protected, though "
            "proponents argue that the overall economic efficiency gains and new export "
            "opportunities often outweigh these localized negative effects in the long term."
        )
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "Edge case-Complex sentence structure")
        
        self.assert_pass(result_data)

    def test_edge_case_single_sentence_response(self):
        """Test case: Edge case - Single sentence response.
        
        A minimal single-sentence response that is coherent but very brief.
        """
        query = "What color is the sky?"
        
        response = "The sky appears blue during the day due to the scattering of sunlight by the atmosphere."
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "Edge case-Single sentence response")
        
        self.assert_pass(result_data)

    def test_edge_case_one_word_correct_response(self):
        """Test case: Edge case - One word correct response.
        
        A single word response that directly and coherently answers the question,
        testing the minimum possible coherent response.
        """
        query = "What is the capital of France?"
        
        response = "Paris."
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "Edge case-One word correct response")
        
        self.assert_pass(result_data)

    def test_edge_case_two_word_correct_response(self):
        """Test case: Edge case - Two word correct response.
        
        A two-word response that coherently addresses the question,
        testing minimal but valid coherent expression.
        """
        # TODO: Test fails - evaluator scores 2 (Poorly coherent) for minimal responses.
        # Reason: "The response lacks clarity and coherence due to its incomplete structure."
        # Decision needed: Should minimal but correct responses pass coherence, or is this expected behavior?
        query = "Is Paris the capital of France?"
        
        response = "Yes it's."
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "Edge case-Two word correct response")
        
        self.assert_pass(result_data)