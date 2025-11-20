# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
from unittest.mock import patch

from engine.engine import BaseEngine
from configs import EngineConfig, TaskConfig
from replica_manager import ReplicaManagerConfig, ReplicaManager, get_engine

from typing import List, Dict

class StubEngine(BaseEngine):
    def __init__(self, engine_config):
        self.engine_config = engine_config

    def load_model(self, env=None):
        # store any environment variables for testing
        self.cuda_visible_devices = env.get('CUDA_VISIBLE_DEVICES') if env else None
        pass

    def init_client(self):
        pass

    def generate(self, prompts: List[str], params: Dict):
        pass

    def generate_openai_response(self, request, headers):
        pass

    def is_port_open(self, host, port, timeout=1.0):
        return True

    @property
    def server_url(self) -> str:
        return f"{self.engine_config.host}:{self.engine_config.port}"


class TestReplicaManager(unittest.TestCase):
    engine_config = EngineConfig(
        engine_name="stub_engine",
        model_id="stub_model",
        tokenizer="stub_tokenizer",
        tensor_parallel=1,
    )

    task_config = TaskConfig()

    def setUp(self):
        self.replica_config = ReplicaManagerConfig(
            engine_config=self.engine_config,
            task_config=self.task_config,
            num_replicas=4,
            gpu_ids="0,1,2,3",
        )

        self.mock_get_engine = patch('replica_manager.get_engine').start()

        self.mock_model_size = patch('replica_manager.get_model_size_in_gb').start()
        self.mock_model_size.return_value = 1

        self.mock_verify_fits = patch('replica_manager.verify_model_fits_in_gpu').start()
        self.mock_verify_fits.return_value = None

        def side_effect_func(engine_config, task_config):
            return StubEngine(engine_config)

        self.mock_get_engine.side_effect = side_effect_func

    def tearDown(self):
        patch.stopall

    def test_initialize_creates_correct_number_of_replicas(self):
        replica_manager = ReplicaManager(self.replica_config)
        self.mock_model_size.assert_called()
        replica_manager.initialize()
        self.assertEqual(len(replica_manager.engine_replicas), self.replica_config.num_replicas)

    def test_initialize_assigns_unique_ports(self):
        replica_manager = ReplicaManager(self.replica_config)
        replica_manager.initialize()
        ports = [engine.engine_config.port for engine in replica_manager.engine_replicas]
        self.assertEqual(len(ports), len(set(ports)))

    def test_initialize_raises_error_with_insufficient_gpus(self):
        self.replica_config.num_replicas = 8
        with self.assertRaises(Exception):
            replica_manager = ReplicaManager(self.replica_config)
            replica_manager.initialize()

    def test_get_replica_round_robin(self):
        replica_manager = ReplicaManager(self.replica_config)
        replica_manager.initialize()
        replicas = [replica_manager.get_replica() for _ in range(8)]
        self.assertEqual(replicas, replica_manager.engine_replicas * 2)

    @patch('replica_manager.get_engine', side_effect=Exception("Failed to load model"))
    def test_error_handling_on_model_load_failure(self, mock_get_engine):
        with self.assertRaises(Exception):
            replica_manager = ReplicaManager(self.replica_config)
            replica_manager.initialize()

    def test_invalid_engine(self):
        with self.assertRaises(ValueError):
            get_engine(EngineConfig(engine_name="invalid_engine", model_id="gpt2", tokenizer="tokenizer"), TaskConfig())

    def test_replica_manager_insufficient_gpus(self):
        self.engine_config.tensor_parallel = 4
        self.replica_config.num_replicas = 2
        with self.assertRaisesRegex(ValueError,
                                    expected_regex=r"Insufficient GPUs: Need .* but only .* are available."):
            replica_manager = ReplicaManager(self.replica_config)
            replica_manager.initialize()

    def test_replica_manager_no_user_settings(self):
        self.engine_config.tensor_parallel = None
        self.replica_config.num_replicas = -1
        replica_manager = ReplicaManager(self.replica_config)
        replica_manager.initialize()
        self.assertEqual(len(replica_manager.engine_replicas), 1)

    def test_replica_manager_no_user_settings_special_model(self):
        self.engine_config.model_config = {"model_type": "phi3"}
        self.engine_config.ml_model_info = {"model_id": "microsoft/phi-3-mini-4k-instruct"}
        self.replica_config.num_replicas = -1
        replica_manager = ReplicaManager(self.replica_config)
        replica_manager.initialize()
        self.assertEqual(len(replica_manager.engine_replicas), 4)

    def test_replica_manager_user_set_tensor(self):
        self.engine_config.tensor_parallel = 4
        self.replica_config.num_replicas = -1

        replica_manager = ReplicaManager(self.replica_config)
        replica_manager.initialize()
        self.assertEqual(len(replica_manager.engine_replicas), 1)

    def test_replica_manager_user_set_tensor_special_model(self):
        self.engine_config.model_config = {"model_type": "phi3"}
        self.engine_config.ml_model_info = {"model_id": "microsoft/phi-3-mini-4k-instruct"}
        self.engine_config.tensor_parallel = 4
        self.replica_config.num_replicas = -1

        replica_manager = ReplicaManager(self.replica_config)
        replica_manager.initialize()
        self.assertEqual(len(replica_manager.engine_replicas), 1)

    def test_replica_manager_user_set_replica(self):
        self.engine_config.tensor_parallel = 2
        self.replica_config.num_replicas = 2

        replica_manager = ReplicaManager(self.replica_config)
        replica_manager.initialize()
        self.assertEqual(len(replica_manager.engine_replicas), 2)

    def test_replica_manager_user_set_replica_special_model(self):
        self.engine_config.model_config = {"model_type": "phi3"}
        self.engine_config.ml_model_info = {"model_id": "microsoft/phi-3-mini-4k-instruct"}
        self.engine_config.tensor_parallel = 2
        self.replica_config.num_replicas = 2

        replica_manager = ReplicaManager(self.replica_config)
        replica_manager.initialize()
        self.assertEqual(len(replica_manager.engine_replicas), 2)

    # FIXME: This test fails because we can't check mock environment variables in the engine.
    # def test_correct_gpu_assignment(self):
    #     replica_manager = ReplicaManager(self.replica_config)
    #     replica_manager.initialize()
    #     gpu_ids = [engine.cuda_visible_devices for engine in replica_manager.engine_replicas]
    #     expected_gpu_ids = self.replica_config.gpu_ids.split(',')
    #     self.assertEqual(gpu_ids, expected_gpu_ids)


if __name__ == '__main__':
    unittest.main()
