# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
from unittest.mock import MagicMock, patch

from configs import EngineConfig
from configs import TaskConfig
from constants import TaskType
from fm_score import FMScore
from fm_score import get_formatter
from managed_inference import MIRPayload


class TestFMScore(unittest.TestCase):
    def setUp(self):
        self.sample_config = {
            'engine': {
                'engine_name': 'mii',
                'model_id': 'Llama2',
                'tokenizer': 'tokenizer',
                'mii_config': {
                    'deployment_name': 'sample_deployment',
                    'mii_configs': {},
                    'ds_config': None,
                    'ds_zero': False,
                },
            },
            'task': {
                'task_type': TaskType.TEXT_GENERATION,
            },
        }

    @patch('fm_score.get_formatter')
    @patch('fm_score.ReplicaManager')
    def test_init(self, mock_replica_manager, mock_get_formatter):
        mock_formatter = MagicMock()
        mock_replica_manager_instance = MagicMock()
        mock_replica_manager.return_value = mock_replica_manager_instance
        mock_get_formatter.return_value = mock_formatter

        fms = FMScore(self.sample_config)
        fms.init()

        self.assertEqual(fms.engine_config, EngineConfig.from_dict(self.sample_config['engine']))
        self.assertEqual(fms.task_config, TaskConfig.from_dict(self.sample_config['task']))
        mock_get_formatter.assert_called_once_with(model_name='Llama2')
        self.assertEqual(fms.formatter, mock_formatter)
        self.assertEqual(fms.replica_manager, mock_replica_manager_instance)

    @patch('fm_score.get_formatter')
    @patch('fm_score.ReplicaManager')
    def test_run(self, mock_replica_manager, mock_get_formatter):
        mock_formatter = MagicMock()
        mock_replica_manager_instance = MagicMock()
        mock_replica_manager.return_value = mock_replica_manager_instance
        mock_get_formatter.return_value = mock_formatter

        fms = FMScore(self.sample_config)
        fms.init()

        payload = MIRPayload('Today is a wonderful day to ', {'max_length': 128}, fms.task_config.task_type, True)
        payload.convert_query_to_list()
        output = fms.run(payload)

        mock_formatter.format_prompt.assert_called_once_with(
            fms.task_config.task_type,
            'Today is a wonderful day to ', {'max_length': 128},
        )
        mock_replica_manager_instance.get_replica().generate.assert_called_once_with(
            [mock_formatter.format_prompt.return_value], {'max_length': 128}
        )
        self.assertEqual(output, mock_replica_manager_instance.get_replica().generate.return_value)

    def test_invalid_model(self):
        with self.assertRaises(ValueError):
            get_formatter('invalid_model')

    @patch('fm_score.ReplicaManager')
    def test_initialize_formatter_from_custom_model_config_builder(self, mock_replica_manager):
        custom_model_config_builder = MagicMock()
        custom_model_config_builder.get_formatter = MagicMock(return_value='custom formatter')
        self.sample_config["engine"]["custom_model_config_builder"] = custom_model_config_builder
        self.sample_config["task"]["task_type"] = TaskType.TEXT_TO_IMAGE_INPAINTING
        fms = FMScore(self.sample_config)
        fms.init()
        assert fms.formatter == 'custom formatter'


if __name__ == '__main__':
    unittest.main()
=======
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from configs import EngineConfig
from configs import MiiEngineConfig
from configs import TaskConfig
from constants import TaskType
from fm_score import FMScore
from fm_score import get_formatter
from managed_inference import MIRPayload


class TestFMScore(unittest.TestCase):
    def setUp(self):
        self.sample_config = {
            'engine': {
                'engine_name': 'mii',
                'model_id': 'Llama2',
                'tokenizer': 'tokenizer',
                'mii_config': {
                    'deployment_name': 'sample_deployment',
                    'mii_configs': {},
                    'ds_config': None,
                    'ds_zero': False,
                },
            },
            'task': {
                'task_type': TaskType.TEXT_GENERATION,
            },
        }

    @patch('fm_score.get_formatter')
    @patch('fm_score.ReplicaManager')
    def test_init(self, mock_replica_manager, mock_get_formatter):
        mock_formatter = MagicMock()
        mock_replica_manager_instance = MagicMock()
        mock_replica_manager.return_value = mock_replica_manager_instance
        mock_get_formatter.return_value = mock_formatter

        fms = FMScore(self.sample_config)
        fms.init()

        self.assertEqual(fms.engine_config, EngineConfig.from_dict(self.sample_config['engine']))
        self.assertEqual(fms.task_config, TaskConfig.from_dict(self.sample_config['task']))
        mock_get_formatter.assert_called_once_with(model_name='Llama2')
        self.assertEqual(fms.formatter, mock_formatter)
        self.assertEqual(fms.replica_manager, mock_replica_manager_instance)

    @patch('fm_score.get_formatter')
    @patch('fm_score.ReplicaManager')
    def test_run(self, mock_replica_manager, mock_get_formatter):
        mock_formatter = MagicMock()
        mock_replica_manager_instance = MagicMock()
        mock_replica_manager.return_value = mock_replica_manager_instance
        mock_get_formatter.return_value = mock_formatter

        fms = FMScore(self.sample_config)
        fms.init()

        payload = MIRPayload('Today is a wonderful day to ', {'max_length': 128}, fms.task_config.task_type, True)
        payload.convert_query_to_list()
        output = fms.run(payload)

        mock_formatter.format_prompt.assert_called_once_with(
            fms.task_config.task_type,
            'Today is a wonderful day to ', {'max_length': 128},
        )
        mock_replica_manager_instance.get_replica().generate.assert_called_once_with(
            [mock_formatter.format_prompt.return_value], {'max_length': 128}
        )
        self.assertEqual(output, mock_replica_manager_instance.get_replica().generate.return_value)

    def test_invalid_model(self):
        with self.assertRaises(ValueError):
            get_formatter('invalid_model')

    @patch('fm_score.ReplicaManager')
    def test_initialize_formatter_from_custom_model_config_builder(self, mock_replica_manager):
        custom_model_config_builder = MagicMock()
        custom_model_config_builder.get_formatter = MagicMock(return_value='custom formatter')
        self.sample_config["engine"]["custom_model_config_builder"] = custom_model_config_builder
        self.sample_config["task"]["task_type"] = TaskType.TEXT_TO_IMAGE_INPAINTING
        fms = FMScore(self.sample_config)
        fms.init()
        assert fms.formatter == 'custom formatter'


if __name__ == '__main__':
    unittest.main()
