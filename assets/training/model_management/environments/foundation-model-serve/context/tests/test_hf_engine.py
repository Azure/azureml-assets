import unittest
from foundation.model.serve.engine.hf_engine import HfEngine
from foundation.model.serve.configs import EngineConfig, TaskConfig
from foundation.model.serve.constants import TaskType


class TestHfEngine(unittest.TestCase):
    fill_mask_task_config = TaskConfig(task_type=TaskType.FILL_MASK)
    qna_task_config = TaskConfig(task_type=TaskType.QnA)

    def test_fill_mask(self):
        ml_model_info = {
                "task": "fill-mask",
                "hf_tokenizer_class": "BertTokenizerFast",
                "hf_pretrained_class": "BertForMaskedLM"
        }
        engine_config = EngineConfig(engine_name="hf", model_id="bert-base-cased",
                                     tokenizer="bert-base-cased", hf_config_path="bert-base-cased",
                                     ml_model_info=ml_model_info)
        engine = HfEngine(engine_config, self.fill_mask_task_config)
        test_tokens = engine.generate(["[MASK] is the capital of France."], params={})
        self.assertIsNotNone(test_tokens)

    def test_question_answering(self):
        ml_model_info = {
                "task": "question-answering",
                "hf_tokenizer_class": "DistilBertTokenizerFast",
                "hf_pretrained_class": "DistilBertForQuestionAnswering"
        }
        engine_config = EngineConfig(engine_name="hf", model_id="distilbert-base-cased-distilled-squad",
                                     tokenizer="distilbert-base-cased-distilled-squad",
                                     hf_config_path="distilbert-base-cased-distilled-squad",
                                     ml_model_info=ml_model_info)
        engine = HfEngine(engine_config, self.qna_task_config)
        context = """
            Extractive Question Answering is the task of extracting an answer from a text given a question.
            An example of a question answering dataset is the SQuAD dataset,
            which is entirely based on that task.
            If you would like to fine-tune a model on a SQuAD task,
            you may leverage the examples/pytorch/question-answering/run_squad.py script.
        """
        test_tokens = engine.generate([{"question":"What is a good example of a question answering dataset?",
                                        "context":context}], params={})
        self.assertIsNotNone(test_tokens)



if __name__ == "__main__":
    unittest.main()
