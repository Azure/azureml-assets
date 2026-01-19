# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Relevance Evaluator with real flow execution."""

import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner
from ...builtin.relevance.evaluator._relevance import RelevanceEvaluator


@pytest.mark.quality
class TestRelevanceEvaluatorQuality(BaseQualityEvaluatorRunner):
    """
    Quality tests for Relevance Evaluator.

    Tests actual LLM evaluation with real flow execution (no mocking).

    Relevance Scoring Rubric (from prompty):
    - Score 5: Comprehensive Response with Insights - fully answers and adds meaningful elaboration/context
    - Score 4: Fully Relevant/Sufficient - covers all essential aspects, minor omissions acceptable
    - Score 3: Partially Relevant/Incomplete - addresses query but omits essential components
    - Score 2: Related but Unhelpful/Superficial - vague, non-answers, or failure messages
    - Score 1: Irrelevant Response - completely off-topic, doesn't attempt to address query
    """

    evaluator_type = RelevanceEvaluator

    # ==================== PASS CASES (Score 4-5) ====================

    def test_pass_comprehensive_with_insights(self):
        """Test comprehensive response with valuable insights - should score 5."""
        query = "What amenities does the new apartment complex provide?"
        response = (
            "The apartment complex provides a gym, swimming pool, and 24/7 security, "
            "designed to offer residents a comfortable and active lifestyle while ensuring their safety. "
            "The gym features modern equipment including cardio machines and free weights, "
            "the pool is heated year-round, and security includes both on-site guards and camera monitoring."
        )

        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "PASS-comprehensive with insights")
        self.assert_pass(result_data)

    def test_pass_fully_relevant_complete_answer(self):
        """Test fully relevant response covering all essential aspects - should score 4-5."""
        query = "What services does the premium membership include?"
        response = (
            "The premium membership includes priority customer support, "
            "exclusive content access, and early product releases."
        )

        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "PASS-fully relevant complete answer")
        self.assert_pass(result_data)

    def test_pass_detailed_technical_explanation(self):
        """Test detailed technical response that fully addresses the query."""
        query = "How does photosynthesis work?"
        response = (
            "Photosynthesis is the process by which plants convert light energy into chemical energy. "
            "It occurs in two stages: the light-dependent reactions in the thylakoid membranes, "
            "where water is split and ATP/NADPH are produced, and the Calvin cycle in the stroma, "
            "where CO2 is fixed into glucose using the ATP and NADPH. This process is essential "
            "for producing oxygen and organic compounds that sustain most life on Earth."
        )

        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "PASS-detailed technical explanation")
        self.assert_pass(result_data)

    def test_pass_practical_advice_with_context(self):
        """Test practical response with helpful context and reasoning."""
        query = "What's the best way to learn a new programming language?"
        response = (
            "The most effective approach is to combine structured learning with hands-on practice. "
            "Start with official documentation or a beginner course to understand syntax and concepts. "
            "Then build small projects that interest you - this reinforces learning and builds portfolio work. "
            "Join communities like Stack Overflow or Discord for help and code reviews. "
            "Consistency matters more than intensity; even 30 minutes daily yields better results than sporadic long sessions."
        )

        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "PASS-practical advice with context")
        self.assert_pass(result_data)

    def test_pass_direct_factual_answer(self):
        """Test direct factual answer that fully addresses the query."""
        query = "What is the capital of France?"
        response = "The capital of France is Paris. It's the largest city in France and serves as the country's political, economic, and cultural center."

        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "PASS-direct factual answer")
        self.assert_pass(result_data)

    def test_pass_step_by_step_instructions(self):
        """Test step-by-step instructions that comprehensively address a how-to query."""
        query = "How do I reset my password?"
        response = (
            "To reset your password: 1) Go to the login page and click 'Forgot Password'. "
            "2) Enter your email address and click 'Send Reset Link'. "
            "3) Check your email (including spam folder) for the reset link. "
            "4) Click the link and enter your new password twice. "
            "5) Click 'Save' to confirm. Your new password will be active immediately."
        )

        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "PASS-step by step instructions")
        self.assert_pass(result_data)

    # ==================== FAIL CASES (Score 1-2) ====================

    def test_fail_completely_irrelevant(self):
        """Test completely off-topic response - should score 1."""
        query = "What is the team preparing for?"
        response = "I went grocery shopping yesterday evening."

        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "FAIL-completely irrelevant")
        self.assert_fail(result_data)

    def test_fail_unrelated_topic(self):
        """Test response about unrelated topic - should score 1."""
        query = "When will the company's new product line launch?"
        response = "International travel can be very rewarding and educational."

        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "FAIL-unrelated topic")
        self.assert_fail(result_data)

    def test_fail_vague_non_answer(self):
        """Test vague response that provides no useful information - should score 2."""
        query = "What is the event about?"
        response = "It's something important."

        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "FAIL-vague non-answer")
        self.assert_fail(result_data)

    def test_fail_error_message_response(self):
        """Test error/failure response that provides no useful information - should score 2."""
        query = "What's the weather in Paris?"
        response = "I tried to find the forecast but the query failed."

        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "FAIL-error message response")
        self.assert_fail(result_data)

    def test_fail_random_unrelated_facts(self):
        """Test response with random unrelated facts - should score 1."""
        query = "How do I configure the database connection?"
        response = "The Great Wall of China is over 13,000 miles long and was built over many centuries."

        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "FAIL-random unrelated facts")
        self.assert_fail(result_data)

    def test_fail_deflection_without_info(self):
        """Test deflection that provides no useful information."""
        query = "What are the system requirements for the software?"
        response = "That's a good question. You should definitely look into that."

        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "FAIL-deflection without info")
        self.assert_fail(result_data)

    # ==================== EDGE CASES ====================

    def test_edge_case_partially_relevant(self):
        """Test partially relevant response that omits important details - may score 3."""
        # TODO: Test fails - evaluator scores 3.0 (pass at threshold) instead of expected fail.
        # Reason: "The response mentions one amenity, the gym, but does not provide a complete list.
        # It partially addresses the user's query but lacks comprehensiveness."
        # Decision needed: Should partially relevant responses (score 3) pass or fail?
        # The test expects fail but evaluator gives exactly threshold score (3.0).
        query = "What amenities does the new apartment complex provide?"
        response = "The apartment complex has a gym."

        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "EDGE-partially relevant")
        # Partially relevant should score around 3, which is below threshold
        self.assert_fail(result_data)

    def test_edge_case_single_service_mentioned(self):
        """Test response mentioning only one aspect of a multi-faceted query."""
        query = "What services does the premium membership include?"
        response = "It includes priority customer support."

        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "EDGE-single service mentioned")
        # Partial answer, may score 3
        self.assert_pass(result_data)

    def test_edge_case_correct_but_minimal(self):
        """Test correct but very minimal response."""
        query = "What is the capital of Japan?"
        response = "Tokyo."

        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "EDGE-correct but minimal")
        # Correct but minimal - could pass or be considered too brief
        self.assert_pass(result_data)

    def test_edge_case_related_but_different_aspect(self):
        """Test response about related topic but different aspect than asked."""
        query = "What time does the restaurant open?"
        response = "The restaurant serves Italian cuisine and is known for its pasta dishes."

        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "EDGE-related but different aspect")
        # Related to restaurant but doesn't answer the time question
        self.assert_fail(result_data)

    def test_edge_case_excessive_detail_but_relevant(self):
        """Test response with excessive detail that may overwhelm but is relevant."""
        query = "What is Python?"
        response = (
            "Python is a high-level, interpreted programming language created by Guido van Rossum in 1991. "
            "It emphasizes code readability with significant whitespace and supports multiple programming paradigms "
            "including procedural, object-oriented, and functional programming. Python has a comprehensive standard "
            "library and a vast ecosystem of third-party packages available through PyPI. It's widely used in web "
            "development (Django, Flask), data science (pandas, NumPy, scikit-learn), machine learning (TensorFlow, "
            "PyTorch), automation, and scripting. Python 3 is the current major version, with Python 2 having reached "
            "end of life in 2020. The language is known for its gentle learning curve and is often recommended for beginners."
        )

        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "EDGE-excessive detail but relevant")
        self.assert_pass(result_data)

    def test_edge_case_yes_no_question(self):
        """Test appropriate response to a yes/no question."""
        query = "Is Python free to use?"
        response = "Yes, Python is completely free and open-source under the Python Software Foundation License."

        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "EDGE-yes no question")
        self.assert_pass(result_data)

    def test_edge_case_yes_no_question_minimal_answer(self):
        """Test appropriate response to a yes/no question."""
        query = "Is Python free to use?"
        response = "Yes."

        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "EDGE-yes no question minimal answer")
        self.assert_pass(result_data)

    def test_edge_case_ambiguous_query(self):
        """Test response to ambiguous query that seeks clarification."""
        query = "Tell me about the bank."
        response = (
            "I'd be happy to help, but could you clarify what you mean by 'bank'? "
            "Are you asking about a financial institution, a river bank, a memory bank in computing, "
            "or something else? With more context, I can provide more relevant information."
        )

        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "EDGE-ambiguous query")
        # Seeking clarification for ambiguous query is reasonable
        self.assert_pass(result_data)

    def test_edge_case_multi_part_query_all_answered(self):
        """Test response that addresses all parts of a multi-part query."""
        query = "What is machine learning, and how is it different from traditional programming?"
        response = (
            "Machine learning is a subset of AI where systems learn patterns from data rather than "
            "following explicit instructions. Unlike traditional programming where developers write "
            "specific rules for every scenario, ML algorithms identify patterns in training data and "
            "use those patterns to make predictions on new data. Traditional programming follows: "
            "Input + Rules = Output, while ML follows: Input + Output = Rules (learned patterns)."
        )

        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "EDGE-multi part query all answered")
        self.assert_pass(result_data)

    def test_edge_case_multi_part_query_partial_answer(self):
        """Test response that only addresses one part of a multi-part query."""
        query = "What is the company's revenue and how many employees do they have?"
        response = "The company's annual revenue is approximately $50 million."

        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "EDGE-multi part query partial answer")
        # Only answers one part - should be partial (score 3)
        self.assert_pass(result_data)

    def test_edge_case_speculative_but_relevant(self):
        """Test response that is relevant but necessarily speculative."""
        query = "What will the stock market do tomorrow?"
        response = (
            "I cannot predict stock market movements with certainty. However, market performance "
            "typically depends on factors like economic data releases, corporate earnings, "
            "geopolitical events, and investor sentiment. For tomorrow specifically, consider "
            "checking for any scheduled economic reports or company announcements that might "
            "influence trading."
        )

        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "EDGE-speculative but relevant")
        # Honest about limitations while providing relevant context
        self.assert_pass(result_data)

    def test_edge_case_one_word_correct_answer(self):
        """Test single word response that correctly answers the query."""
        query = "Is the store open on Sundays?"
        response = "Yes."

        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "EDGE-one word correct answer")
        # Correct but extremely brief
        self.assert_pass(result_data)

    def test_edge_case_two_word_answer(self):
        """Test two-word response to a simple query."""
        query = "What color is the sky?"
        response = "It's blue."

        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "EDGE-two word answer")
        # Correct and concise
        self.assert_pass(result_data)

    def test_edge_case_list_format_response(self):
        """Test response in list format."""
        query = "What are the primary colors?"
        response = "The primary colors are: 1) Red, 2) Blue, 3) Yellow. These cannot be created by mixing other colors."

        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "EDGE-list format response")
        self.assert_pass(result_data)

    def test_edge_case_honest_limitation(self):
        """Test response that honestly admits limitation but provides what it can."""
        query = "What is the current temperature in Tokyo?"
        response = (
            "I don't have access to real-time weather data. However, you can check current "
            "Tokyo temperatures on weather websites like weather.com, or apps like Weather Underground. "
            "Generally, Tokyo's climate ranges from cold winters (0-10°C) to hot, humid summers (25-35°C)."
        )

        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "EDGE-honest limitation")
        # Provides helpful alternatives despite limitation
        self.assert_pass(result_data)
