# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Fluency Evaluator with real flow execution."""

import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner
from ...builtin.fluency.evaluator._fluency import FluencyEvaluator


@pytest.mark.quality
class TestFluencyEvaluatorQuality(BaseQualityEvaluatorRunner):
    """
    Quality tests for Fluency Evaluator.

    Tests actual LLM evaluation with real flow execution (no mocking).
    
    Fluency measures the effectiveness and clarity of written communication,
    focusing on grammatical accuracy, vocabulary range, sentence complexity,
    coherence, and overall readability.
    
    Rating scale (default threshold=3):
    - 1: Emergent Fluency - minimal command, pervasive errors, fragmented sentences
    - 2: Basic Fluency - simple ideas, frequent errors, limited vocabulary
    - 3: Competent Fluency - clear ideas, occasional errors, adequate vocabulary
    - 4: Proficient Fluency - well-articulated, good grammar, varied vocabulary
    - 5: Exceptional Fluency - sophisticated vocabulary, flawless grammar, eloquent
    
    Note: Fluency evaluator only requires a response (no query needed).
    """

    evaluator_type = FluencyEvaluator

    # ==================== PASS CASES (Score >= 3) ====================

    def test_pass_exceptional_fluency(self):
        """Test case: PASS - Exceptional fluency (expected score: 5).
        
        Response demonstrates exceptional command of language with sophisticated
        vocabulary, complex varied sentence structures, flawless grammar, and eloquent style.
        """
        response = (
            "Globalization exerts a profound influence on cultural diversity by facilitating "
            "unprecedented cultural exchange while simultaneously risking the homogenization "
            "of distinct cultural identities. This paradoxical phenomenon necessitates a "
            "nuanced approach to preserving indigenous traditions whilst embracing the "
            "inevitable interconnectedness of our modern world. Furthermore, the proliferation "
            "of digital communication platforms has accelerated these cultural dynamics, "
            "creating both opportunities for cross-cultural understanding and challenges "
            "in maintaining the authenticity of local customs and practices."
        )
        
        results = self._run_evaluation(response=response)
        result_data = self._extract_and_print_result(results, "PASS-Exceptional fluency")
        
        self.assert_pass(result_data)

    def test_pass_proficient_fluency(self):
        """Test case: PASS - Proficient fluency (expected score: 4).
        
        Response is well-articulated with good control of grammar, varied vocabulary,
        and complex well-structured sentences. Minor errors may occur but don't affect understanding.
        """
        response = (
            "Environmental conservation is crucial because it protects ecosystems, preserves "
            "biodiversity, and ensures natural resources are available for future generations. "
            "When we take steps to reduce pollution and protect wildlife habitats, we contribute "
            "to a healthier planet. Additionally, sustainable practices in agriculture and "
            "industry can help mitigate the effects of climate change while supporting economic "
            "development in local communities."
        )
        
        results = self._run_evaluation(response=response)
        result_data = self._extract_and_print_result(results, "PASS-Proficient fluency")
        
        self.assert_pass(result_data)

    @pytest.mark.flaky(reruns=3)
    def test_pass_competent_fluency_at_threshold(self):
        """Test case: PASS - Competent fluency at threshold (expected score: 3).
        
        Response clearly conveys ideas with occasional grammatical errors.
        Vocabulary is adequate but not extensive. Sentences are generally correct
        but may lack complexity and variety.
        """
        response = (
            "I'm planning to visit my friends next weekend and maybe see a movie together. "
            "We usually go to the mall and have lunch at a restaurant. Sometimes we play "
            "video games at someone's house. It's always fun to spend time with friends "
            "and relax after a busy week at work."
        )
        
        results = self._run_evaluation(response=response)
        result_data = self._extract_and_print_result(results, "PASS-Competent fluency at threshold")
        
        self.assert_pass(result_data)

    def test_pass_technical_language_fluent(self):
        """Test case: PASS - Technical language with fluent expression.
        
        Response uses technical terminology but maintains grammatical correctness
        and clear expression.
        """
        response = (
            "The neural network architecture employs a series of convolutional layers "
            "followed by batch normalization and ReLU activation functions. During the "
            "forward pass, the input tensor is progressively transformed through these "
            "layers, extracting increasingly abstract feature representations. The final "
            "fully connected layers then map these features to the output classification "
            "probabilities using a softmax activation function."
        )
        
        results = self._run_evaluation(response=response)
        result_data = self._extract_and_print_result(results, "PASS-Technical language fluent")
        
        self.assert_pass(result_data)

    def test_pass_simple_but_grammatically_correct(self):
        """Test case: PASS - Simple but grammatically correct response.
        
        Response uses simple vocabulary and sentence structures but maintains
        grammatical correctness throughout.
        """
        response = (
            "The weather today is sunny and warm. I decided to go for a walk in the park. "
            "There were many people enjoying the nice day. Some children were playing on "
            "the swings while their parents watched nearby. I sat on a bench and read my "
            "book for about an hour. It was a peaceful afternoon."
        )
        
        results = self._run_evaluation(response=response)
        result_data = self._extract_and_print_result(results, "PASS-Simple but grammatically correct")
        
        self.assert_pass(result_data)

    # ==================== FAIL CASES (Score < 3) ====================

    def test_fail_emergent_fluency_fragmented(self):
        """Test case: FAIL - Emergent fluency with fragmented sentences (expected score: 1).
        
        Response shows minimal command of language with pervasive grammatical errors,
        extremely limited vocabulary, and fragmented incoherent sentences.
        """
        response = "Free time I. Go park. Not fun. Alone. Weather good but boring still."
        
        results = self._run_evaluation(response=response)
        result_data = self._extract_and_print_result(results, "FAIL-Emergent fluency fragmented")
        
        self.assert_fail(result_data)

    def test_fail_basic_fluency_frequent_errors(self):
        """Test case: FAIL - Basic fluency with frequent errors (expected score: 2).
        
        Response communicates simple ideas but has frequent grammatical errors
        and limited vocabulary. Sentences are short and improperly constructed.
        """
        response = (
            "I like play soccer very much. Yesterday I go to park with friend. "
            "We play for long time. It fun. Then we eat pizza. Pizza is good very. "
            "My friend he like basketball more. We have different hobby."
        )
        
        results = self._run_evaluation(response=response)
        result_data = self._extract_and_print_result(results, "FAIL-Basic fluency frequent errors")
        
        self.assert_fail(result_data)

    def test_fail_severe_grammatical_errors(self):
        """Test case: FAIL - Severe grammatical errors throughout.
        
        Response has subject-verb disagreement, wrong tenses, and missing articles
        that significantly impair readability.
        """
        response = (
            "Yesterday I goes to store for buy some food. The store it was very crowd "
            "and I waiting in line for long time. When I arrive at home, I cooking dinner "
            "for my family. Everyone they was happy for eat together. We talks about many "
            "thing during dinner time."
        )
        
        results = self._run_evaluation(response=response)
        result_data = self._extract_and_print_result(results, "FAIL-Severe grammatical errors")
        
        self.assert_fail(result_data)

    def test_fail_incomprehensible_word_salad(self):
        """Test case: FAIL - Incomprehensible word arrangement.
        
        Response consists of words that don't form meaningful sentences,
        making the message largely incomprehensible.
        """
        response = "Like food pizza. Good cheese eat. Restaurant go tomorrow yes. Happy very stomach full."
        
        results = self._run_evaluation(response=response)
        result_data = self._extract_and_print_result(results, "FAIL-Incomprehensible word salad")
        
        self.assert_fail(result_data)

    def test_fail_missing_function_words(self):
        """Test case: FAIL - Missing essential function words.
        
        Response lacks articles, prepositions, and auxiliary verbs,
        making it read like a telegram.
        """
        response = (
            "Company need increase profit next quarter. Manager say employee must work harder. "
            "Meeting schedule Monday discuss strategy. Everyone must attend no exception. "
            "Report due Friday deadline strict."
        )
        
        results = self._run_evaluation(response=response)
        result_data = self._extract_and_print_result(results, "FAIL-Missing function words")
        
        self.assert_fail(result_data)

    # ==================== EDGE CASES ====================

    def test_edge_case_single_eloquent_sentence(self):
        """Test case: Edge case - Single eloquent sentence.
        
        A single sentence that is grammatically perfect and sophisticated.
        """
        response = (
            "The inexorable march of technological progress has fundamentally transformed "
            "the way we communicate, work, and perceive the world around us."
        )
        
        results = self._run_evaluation(response=response)
        result_data = self._extract_and_print_result(results, "Edge case-Single eloquent sentence")
        
        self.assert_pass(result_data)

    def test_edge_case_very_short_simple_response(self):
        """Test case: Edge case - Very short simple response.
        
        A minimal response that is grammatically correct but provides little
        content to evaluate.
        """
        response = "The meeting is scheduled for tomorrow at noon."
        
        results = self._run_evaluation(response=response)
        result_data = self._extract_and_print_result(results, "Edge case-Very short simple response")
        
        self.assert_pass(result_data)

    def test_edge_case_one_word_correct_response(self):
        """Test case: Edge case - One word correct response.
        
        A single word response that is grammatically valid in context,
        testing the minimum possible fluent response.
        """
        # TODO: Test fails - evaluator scores 1 (Emergent Fluency) for one-word responses.
        # Reason: "The response consists of only one word, which severely limits expression."
        # Decision needed: Should single-word grammatically correct responses pass fluency,
        # or is more context needed to evaluate fluency?
        response = "Yes."
        
        results = self._run_evaluation(response=response)
        result_data = self._extract_and_print_result(results, "Edge case-One word correct response")
        
        self.assert_pass(result_data)

    def test_edge_case_two_word_correct_response(self):
        """Test case: Edge case - Two word correct response.
        
        A two-word response that is grammatically complete,
        testing minimal but valid fluent expression.
        """
        # TODO: Test fails - evaluator scores 1-2 for two-word responses.
        # Reason: "The response is extremely brief, limiting fluency assessment."
        # Decision needed: Should minimal grammatically correct responses pass fluency,
        # or does fluency require sufficient length to demonstrate language command?
        response = "Absolutely correct."
        
        results = self._run_evaluation(response=response)
        result_data = self._extract_and_print_result(results, "Edge case-Two word correct response")
        
        self.assert_pass(result_data)

    def test_edge_case_mixed_fluency_levels(self):
        """Test case: Edge case - Mixed fluency levels within response.
        
        Response starts with good fluency but degrades to poor grammar,
        testing how the evaluator handles inconsistent quality.
        """
        response = (
            "The implementation of renewable energy sources represents a significant step "
            "toward environmental sustainability. Solar and wind power offer clean alternatives "
            "to fossil fuels. But then problem come when storage. Battery not good enough yet. "
            "Cost too high for many country. Need more research for make better."
        )
        
        results = self._run_evaluation(response=response)
        result_data = self._extract_and_print_result(results, "Edge case-Mixed fluency levels")
        
        self.assert_fail(result_data)

    def test_edge_case_formal_academic_style(self):
        """Test case: Edge case - Formal academic writing style.
        
        Response uses formal academic language with complex sentence structures
        and specialized vocabulary.
        """
        response = (
            "The epistemological implications of quantum mechanics have engendered considerable "
            "debate among philosophers of science, particularly regarding the ontological status "
            "of unobserved quantum states. The Copenhagen interpretation, while mathematically "
            "consistent, raises profound questions about the nature of reality and the role of "
            "the observer in determining physical outcomes. Consequently, alternative interpretations, "
            "such as the many-worlds hypothesis, have been proposed to address these conceptual "
            "difficulties, though they introduce their own philosophical challenges."
        )
        
        results = self._run_evaluation(response=response)
        result_data = self._extract_and_print_result(results, "Edge case-Formal academic style")
        
        self.assert_pass(result_data)

    def test_edge_case_conversational_informal_style(self):
        """Test case: Edge case - Conversational informal style.
        
        Response uses casual, conversational language that is grammatically
        acceptable in informal contexts.
        """
        response = (
            "So yeah, I was thinking about getting a new laptop because my current one is "
            "pretty old now. I've been looking at a few options online, and honestly, there "
            "are so many choices it's kind of overwhelming. I'll probably end up going with "
            "something mid-range since I don't really need anything too fancy for what I do."
        )
        
        results = self._run_evaluation(response=response)
        result_data = self._extract_and_print_result(results, "Edge case-Conversational informal style")
        
        self.assert_pass(result_data)

    def test_edge_case_run_on_sentences(self):
        """Test case: Edge case - Run-on sentences.
        
        Response consists of overly long sentences that lack proper punctuation
        and sentence boundaries. The evaluator penalizes this as basic fluency
        due to lack of clarity and coherence.
        """
        response = (
            "I went to the store yesterday and I bought some groceries and then I went "
            "home and I started cooking dinner and my friend called me and we talked for "
            "a while and then I finished cooking and I ate dinner and watched some TV and "
            "then I went to bed because I was very tired from such a long day."
        )
        
        results = self._run_evaluation(response=response)
        result_data = self._extract_and_print_result(results, "Edge case-Run-on sentences")
        
        self.assert_fail(result_data)

    @pytest.mark.flaky(reruns=3)
    def test_edge_case_non_native_speaker_patterns(self):
        """Test case: Edge case - Common non-native speaker error patterns.
        
        Response contains typical errors made by non-native English speakers
        such as article misuse and preposition errors. The evaluator considers
        this competent fluency since ideas are clearly conveyed despite errors.
        Expected score: 3.
        """
        # TODO: Test fails - evaluator scores 2 (Basic Fluency) instead of expected 3.
        # Reason: "Several grammatical errors are present (e.g., 'since five years' should be 'for five years')."
        # Decision needed: Should non-native speaker patterns with clear communication pass,
        # or is the evaluator correctly penalizing these grammar errors?
        response = (
            "I have been living in the United States since five years. The life here is very "
            "different from my country. In beginning, I had difficulty to understand the people "
            "because they speak very fast. Now I am more comfortable with the language and I "
            "can make myself understood in most of situations."
        )
        
        results = self._run_evaluation(response=response)
        result_data = self._extract_and_print_result(results, "Edge case-Non-native speaker patterns")
        
        self.assert_pass(result_data)

    def test_edge_case_repetitive_vocabulary(self):
        """Test case: Edge case - Correct grammar but limited vocabulary range.
        
        Response is grammatically correct but uses excessive repetition.
        The evaluator considers this basic fluency due to limited vocabulary
        affecting clarity and engagement.
        Expected score: 2.
        """
        response = (
            "The meeting was very good because we discussed many good ideas. "
            "The team presented a good proposal, and the manager thought it was good. "
            "We decided that this good approach would lead to good results for our project. "
            "Everyone felt good about the good progress we made during this good session."
        )
        
        results = self._run_evaluation(response=response)
        result_data = self._extract_and_print_result(results, "Edge case-Repetitive vocabulary")

        # TODO: Decide if this should pass or fail based on repetition handling
        self.assert_fail(result_data)
