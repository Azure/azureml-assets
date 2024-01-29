import pytest
from packaging.version import Version
from tools.spec_version_upgrader import SpecVersionUpgrader, Spec


@pytest.mark.unit
class TestSpecVersionUpgrader:
    def test_upgrade_versions(self):
        spec_dict = {
            "s1": Spec("s1", "1.0.0"),
            "s2": Spec("s2", "1.0.1"),
            "s3": Spec("s3", "1.2.3", {"s1": "1.0.0"}),
            "s4": Spec("s4", "1.2.0", {"s1": "1.0.0", "s2": "1.0.0"}),
            "s5": Spec("s5", "2.0.3", {"s2": "1.0.1"}),
            "s6": Spec("s6", "2.1.4", {"s4": "1.1.5", "s2": "1.0.1"}),
            "s7": Spec("s7", "2.5.3"),
            "s8": Spec("s8", "0.4.3", {"s7": "2.5.2"})
        }
        spec_version_upgrader = SpecVersionUpgrader("spec_folder", spec_dict)

        spec_version_upgrader._upgrade_versions(["s1", "s2"])

        assert spec_dict == {
            "s1": Spec("s1", "1.0.1", need_to_update=True),
            "s2": Spec("s2", "1.0.2", need_to_update=True),
            "s3": Spec("s3", "1.2.4", {"s1": Version("1.0.1")}, need_to_update=True),
            "s4": Spec("s4", "1.2.1", {"s1": Version("1.0.1"), "s2": Version("1.0.2")}, need_to_update=True),
            "s5": Spec("s5", "2.0.4", {"s2": Version("1.0.2")}, need_to_update=True),
            "s6": Spec("s6", "2.1.5", {"s4": Version("1.2.1"), "s2": Version("1.0.2")}, need_to_update=True),
            "s7": Spec("s7", "2.5.3"),
            "s8": Spec("s8", "0.4.3", {"s7": "2.5.2"})
        }, "Spec version not upgraded correctly"
