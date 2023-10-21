# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the Azure Monitor Metric Publisher component."""

# import os
import pytest
from utils.ComponentHelpers import is_environment_valid


@pytest.mark.unit
class TestComponentAssets():

    def test_validate_component_environments(self, asset_lists):
        (comp_assets_all, _) = asset_lists

        for comp_name in comp_assets_all:
            comp = comp_assets_all[comp_name]
            print(f'Checking Componnent {comp_name}')
            if comp_name != "llm_rag_generate_embeddings_parallel":
                assert is_environment_valid(comp.environment)

    def test_validate_versions_match(self, asset_lists):
        (comp_assets_all, pipe_assets_all) = asset_lists

        for pipeline_name in pipe_assets_all:
            pipeline = pipe_assets_all[pipeline_name]
            for comp in pipeline.components:
                ref_comp = comp_assets_all[comp.name]
                print(f'Checking Pipeline {pipeline_name} Componnent {comp.name}')
                assert comp.version == ref_comp.version
