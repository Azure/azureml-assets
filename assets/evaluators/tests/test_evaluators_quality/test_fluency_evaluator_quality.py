# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Fluency Evaluator with real flow execution."""

import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner, ExpectedResult
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

    def test_pass_exceptional_fluency(self) -> None:
        """Test case: PASS - Exceptional fluency (expected score: 5).

        Response demonstrates exceptional command of language with sophisticated
        vocabulary, complex varied sentence structures, flawless grammar, and eloquent style.
        """
        self.run_quality_test(
            test_label="PASS-exceptional-fluency",
            expected=ExpectedResult.PASS,
            response=(
                "Globalization exerts a profound influence on cultural diversity by facilitating "
                "unprecedented cultural exchange while simultaneously risking the homogenization "
                "of distinct cultural identities. This paradoxical phenomenon necessitates a "
                "nuanced approach to preserving indigenous traditions whilst embracing the "
                "inevitable interconnectedness of our modern world. Furthermore, the proliferation "
                "of digital communication platforms has accelerated these cultural dynamics, "
                "creating both opportunities for cross-cultural understanding and challenges "
                "in maintaining the authenticity of local customs and practices."
            ),
        )

    def test_pass_proficient_fluency(self) -> None:
        """Test case: PASS - Proficient fluency (expected score: 4).

        Response is well-articulated with good control of grammar, varied vocabulary,
        and complex well-structured sentences. Minor errors may occur but don't affect understanding.
        """
        self.run_quality_test(
            test_label="PASS-proficient-fluency",
            expected=ExpectedResult.PASS,
            response=(
                "Environmental conservation is crucial because it protects ecosystems, preserves "
                "biodiversity, and ensures natural resources are available for future generations. "
                "When we take steps to reduce pollution and protect wildlife habitats, we contribute "
                "to a healthier planet. Additionally, sustainable practices in agriculture and "
                "industry can help mitigate the effects of climate change while supporting economic "
                "development in local communities."
            ),
        )

    @pytest.mark.flaky(reruns=3)
    def test_pass_competent_fluency_at_threshold(self) -> None:
        """Test case: PASS - Competent fluency at threshold (expected score: 3).

        Response clearly conveys ideas with occasional grammatical errors.
        Vocabulary is adequate but not extensive. Sentences are generally correct
        but may lack complexity and variety.
        """
        self.run_quality_test(
            test_label="PASS-competent-fluency-at-threshold",
            expected=ExpectedResult.PASS,
            response=(
                "I'm planning to visit my friends next weekend and maybe see a movie together. "
                "We usually go to the mall and have lunch at a restaurant. Sometimes we play "
                "video games at someone's house. It's always fun to spend time with friends "
                "and relax after a busy week at work."
            ),
        )

    def test_pass_technical_language_fluent(self) -> None:
        """Test case: PASS - Technical language with fluent expression.

        Response uses technical terminology but maintains grammatical correctness
        and clear expression.
        """
        self.run_quality_test(
            test_label="PASS-technical-language-fluent",
            expected=ExpectedResult.PASS,
            response=(
                "The neural network architecture employs a series of convolutional layers "
                "followed by batch normalization and ReLU activation functions. During the "
                "forward pass, the input tensor is progressively transformed through these "
                "layers, extracting increasingly abstract feature representations. The final "
                "fully connected layers then map these features to the output classification "
                "probabilities using a softmax activation function."
            ),
        )

    def test_pass_simple_but_grammatically_correct(self) -> None:
        """Test case: PASS - Simple but grammatically correct response.

        Response uses simple vocabulary and sentence structures but maintains
        grammatical correctness throughout.
        """
        self.run_quality_test(
            test_label="PASS-simple-grammatically-correct",
            expected=ExpectedResult.PASS,
            response=(
                "The weather today is sunny and warm. I decided to go for a walk in the park. "
                "There were many people enjoying the nice day. Some children were playing on "
                "the swings while their parents watched nearby. I sat on a bench and read my "
                "book for about an hour. It was a peaceful afternoon."
            ),
        )

    # ==================== FAIL CASES (Score < 3) ====================

    def test_fail_emergent_fluency_fragmented(self) -> None:
        """Test case: FAIL - Emergent fluency with fragmented sentences (expected score: 1).

        Response shows minimal command of language with pervasive grammatical errors,
        extremely limited vocabulary, and fragmented incoherent sentences.
        """
        self.run_quality_test(
            test_label="FAIL-emergent-fluency-fragmented",
            expected=ExpectedResult.FAIL,
            response="Free time I. Go park. Not fun. Alone. Weather good but boring still.",
        )

    @pytest.mark.flaky(reruns=3)
    def test_fail_basic_fluency_frequent_errors(self) -> None:
        """Test case: FAIL - Basic fluency with frequent errors (expected score: 2).

        Response communicates simple ideas but has frequent grammatical errors
        and limited vocabulary. Sentences are short and improperly constructed.
        """
        self.run_quality_test(
            test_label="FAIL-basic-fluency-frequent-errors",
            expected=ExpectedResult.FAIL,
            response=(
                "I like play soccer very much. Yesterday I go to park with friend. "
                "We play for long time. It fun. Then we eat pizza. Pizza is good very. "
                "My friend he like basketball more. We have different hobby."
            ),
        )

    def test_fail_severe_grammatical_errors(self) -> None:
        """Test case: FAIL - Severe grammatical errors throughout.

        Response has subject-verb disagreement, wrong tenses, and missing articles
        that significantly impair readability.
        """
        self.run_quality_test(
            test_label="FAIL-severe-grammatical-errors",
            expected=ExpectedResult.FAIL,
            response=(
                "Yesterday I goes to store for buy some food. The store it was very crowd "
                "and I waiting in line for long time. When I arrive at home, I cooking dinner "
                "for my family. Everyone they was happy for eat together. We talks about many "
                "thing during dinner time."
            ),
        )

    def test_fail_incomprehensible_word_salad(self) -> None:
        """Test case: FAIL - Incomprehensible word arrangement.

        Response consists of words that don't form meaningful sentences,
        making the message largely incomprehensible.
        """
        self.run_quality_test(
            test_label="FAIL-incomprehensible-word-salad",
            expected=ExpectedResult.FAIL,
            response="Like food pizza. Good cheese eat. Restaurant go tomorrow yes. Happy very stomach full.",
        )

    @pytest.mark.flaky(reruns=3)
    def test_fail_missing_function_words(self) -> None:
        """Test case: FAIL - Missing essential function words.

        Response lacks articles, prepositions, and auxiliary verbs,
        making it read like a telegram.
        """
        self.run_quality_test(
            test_label="FAIL-missing-function-words",
            expected=ExpectedResult.FAIL,
            response=(
                "Company need increase profit next quarter. Manager say employee must work harder. "
                "Meeting schedule Monday discuss strategy. Everyone must attend no exception. "
                "Report due Friday deadline strict."
            ),
        )

    # ==================== EDGE CASES ====================

    def test_edge_case_single_eloquent_sentence(self) -> None:
        """Test case: Edge case - Single eloquent sentence.

        A single sentence that is grammatically perfect and sophisticated.
        """
        self.run_quality_test(
            test_label="EDGE-single-eloquent-sentence",
            expected=ExpectedResult.PASS,
            response=(
                "The inexorable march of technological progress has fundamentally transformed "
                "the way we communicate, work, and perceive the world around us."
            ),
        )

    def test_edge_case_very_short_simple_response(self) -> None:
        """Test case: Edge case - Very short simple response.

        A minimal response that is grammatically correct but provides little
        content to evaluate.
        """
        self.run_quality_test(
            test_label="EDGE-very-short-simple-response",
            expected=ExpectedResult.PASS,
            response="The meeting is scheduled for tomorrow at noon.",
        )

    @pytest.mark.flaky(reruns=3)
    def test_edge_case_one_word_correct_response(self) -> None:
        """Test case: Edge case - One word correct response.

        A single word response that is grammatically valid in context,
        testing the minimum possible fluent response.
        """
        self.run_quality_test(
            test_label="EDGE-one-word-correct-response",
            expected=ExpectedResult.PASS,
            response="Yes.",
        )

    def test_edge_case_two_word_correct_response(self) -> None:
        """Test case: Edge case - Two word correct response.

        A two-word response that is grammatically complete,
        testing minimal but valid fluent expression.
        """
        self.run_quality_test(
            test_label="EDGE-two-word-correct-response",
            expected=ExpectedResult.PASS,
            response="Absolutely correct.",
        )

    @pytest.mark.flaky(reruns=3)
    def test_edge_case_mixed_fluency_levels(self) -> None:
        """Test case: Edge case - Mixed fluency levels within response.

        Response starts with good fluency but degrades to poor grammar,
        testing how the evaluator handles inconsistent quality.
        """
        self.run_quality_test(
            test_label="EDGE-mixed-fluency-levels",
            expected=ExpectedResult.FAIL,
            response=(
                "The implementation of renewable energy sources represents a significant step "
                "toward environmental sustainability. Solar and wind power offer clean alternatives "
                "to fossil fuels. But then problem come when storage. Battery not good enough yet. "
                "Cost too high for many country. Need more research for make better."
            ),
        )

    def test_edge_case_formal_academic_style(self) -> None:
        """Test case: Edge case - Formal academic writing style.

        Response uses formal academic language with complex sentence structures
        and specialized vocabulary.
        """
        self.run_quality_test(
            test_label="EDGE-formal-academic-style",
            expected=ExpectedResult.PASS,
            response=(
                "The epistemological implications of quantum mechanics have engendered considerable "
                "debate among philosophers of science, particularly regarding the ontological status "
                "of unobserved quantum states. The Copenhagen interpretation, while mathematically "
                "consistent, raises profound questions about the nature of reality and the role of "
                "the observer in determining physical outcomes. Consequently, alternative interpretations, "
                "such as the many-worlds hypothesis, have been proposed to address these conceptual "
                "difficulties, though they introduce their own philosophical challenges."
            ),
        )

    def test_edge_case_conversational_informal_style(self) -> None:
        """Test case: Edge case - Conversational informal style.

        Response uses casual, conversational language that is grammatically
        acceptable in informal contexts.
        """
        self.run_quality_test(
            test_label="EDGE-conversational-informal-style",
            expected=ExpectedResult.PASS,
            response=(
                "So yeah, I was thinking about getting a new laptop because my current one is "
                "pretty old now. I've been looking at a few options online, and honestly, there "
                "are so many choices it's kind of overwhelming. I'll probably end up going with "
                "something mid-range since I don't really need anything too fancy for what I do."
            ),
        )

    @pytest.mark.flaky(reruns=3)
    def test_edge_case_run_on_sentences(self) -> None:
        """Test case: Edge case - Run-on sentences.

        Response consists of overly long sentences that lack proper punctuation
        and sentence boundaries. The evaluator penalizes this as basic fluency
        due to lack of clarity and coherence.
        """
        self.run_quality_test(
            test_label="EDGE-run-on-sentences",
            expected=ExpectedResult.PASS_WITH_SCORE_3,
            response=(
                "I went to the store yesterday and I bought some groceries and then I went "
                "home and I started cooking dinner and my friend called me and we talked for "
                "a while and then I finished cooking and I ate dinner and watched some TV and "
                "then I went to bed because I was very tired from such a long day."
            ),
        )

    @pytest.mark.flaky(reruns=3)
    def test_edge_case_non_native_speaker_patterns(self) -> None:
        """Test case: Edge case - Common non-native speaker error patterns.

        Response contains typical errors made by non-native English speakers
        such as article misuse and preposition errors. The evaluator considers
        this competent fluency since ideas are clearly conveyed despite errors.
        Expected score: 3.
        """
        self.run_quality_test(
            test_label="EDGE-non-native-speaker-patterns",
            expected=ExpectedResult.PASS,
            response=(
                "I have been living in the United States since five years. The life here is very "
                "different from my country. In beginning, I had difficulty to understand the people "
                "because they speak very fast. Now I am more comfortable with the language and I "
                "can make myself understood in most of situations."
            ),
        )

    @pytest.mark.flaky(reruns=3)
    def test_edge_case_repetitive_vocabulary(self) -> None:
        """Test case: Edge case - Correct grammar but limited vocabulary range.

        Response is grammatically correct but uses excessive repetition.
        The evaluator considers this basic fluency due to limited vocabulary
        affecting clarity and engagement.
        Expected score: 2.
        """
        self.run_quality_test(
            test_label="EDGE-repetitive-vocabulary",
            expected=ExpectedResult.PASS,
            response=(
                "The meeting was very good because we discussed many good ideas. "
                "The team presented a good proposal, and the manager thought it was good. "
                "We decided that this good approach would lead to good results for our project. "
                "Everyone felt good about the good progress we made during this good session."
            ),
        )
