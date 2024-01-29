# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test file for SpecVersionUpgrader."""

import pytest
from packaging.version import Version
from tools.spec_version_upgrader import SpecVersionUpgrader, Spec


@pytest.mark.unit
class TestSpecVersionUpgrader:
    """Test class for SpecVersionUpgrader."""

    @pytest.mark.parametrize(
        "spec_dict_in, specs, spec_dict_out",
        [
            ({}, [], {}),
            ({"s0": Spec("s0", "1.2.3")}, [], {"s0": Spec("s0", "1.2.3")}),
            ({"s0": Spec("s0", "1.2.3")}, ["s0"], {"s0": Spec("s0", "1.2.4", need_to_update=True)}),
            (
                {"s0": Spec("s0", "1.2.3"), "s1": Spec("s1", "3.2.1")},
                ["s0"],
                {"s0": Spec("s0", "1.2.4", need_to_update=True), "s1": Spec("s1", "3.2.1")}
            ),
            (
                {"s0": Spec("s0", "1.2.3"), "s1": Spec("s1", "2.0.2", {"s0": "1.2.2"})},
                ["s0"],
                {
                    "s0": Spec("s0", "1.2.4", need_to_update=True),
                    "s1": Spec("s1", "2.0.3", {"s0": Version("1.2.4")}, need_to_update=True)
                }
            ),
            (
                {
                    "s0": Spec("s0", "2.5.3"),
                    "s1": Spec("s1", "1.0.0"),
                    "s2": Spec("s2", "1.0.1"),
                    "s3": Spec("s3", "1.2.3", {"s1": "1.0.0"}),
                    "s4": Spec("s4", "1.2.0", {"s1": "1.0.0", "s2": "1.0.0"}),
                    "s5": Spec("s5", "2.0.3", {"s2": "1.0.1"}),
                    "s6": Spec("s6", "2.1.4", {"s4": "1.1.5", "s2": "1.0.1"}),
                    "s7": Spec("s7", "0.4.3", {"s0": "2.5.2"})
                },
                ["s1", "s2"],
                {
                    "s0": Spec("s0", "2.5.3"),
                    "s1": Spec("s1", "1.0.1", need_to_update=True),
                    "s2": Spec("s2", "1.0.2", need_to_update=True),
                    "s3": Spec("s3", "1.2.4", {"s1": Version("1.0.1")}, need_to_update=True),
                    "s4": Spec("s4", "1.2.1", {"s1": Version("1.0.1"), "s2": Version("1.0.2")}, need_to_update=True),
                    "s5": Spec("s5", "2.0.4", {"s2": Version("1.0.2")}, need_to_update=True),
                    "s6": Spec("s6", "2.1.5", {"s4": Version("1.2.1"), "s2": Version("1.0.2")}, need_to_update=True),
                    "s7": Spec("s7", "0.4.3", {"s0": "2.5.2"})
                }
            )
        ]
    )
    def test_upgrade_versions(self, spec_dict_in, specs, spec_dict_out):
        """Test _upgrade_versions."""
        spec_dict = spec_dict_in
        spec_version_upgrader = SpecVersionUpgrader("spec_folder", spec_dict)

        spec_version_upgrader._upgrade_versions(specs)

        assert spec_dict == spec_dict_out, "Spec version not upgraded correctly"
