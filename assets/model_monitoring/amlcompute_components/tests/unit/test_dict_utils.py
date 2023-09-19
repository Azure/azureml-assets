# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the dictionary utilities."""

import pytest

from shared_utilities.dict_utils import merge_dicts


@pytest.mark.unit
class TestDictUtils:
    """Test class for dictionary utilities."""

    def test_merge_with_empty_dictionary(self):
        """Test merging dictionaries with an empty dictionary."""
        left = {
            "num_calls": {
                "groups": {
                    "group_1": {
                        "value": 71.0,
                        "threshold": 100.0,
                    },
                },
            },
        }

        right = {}

        result = merge_dicts(left, right)

        assert result == left

    def test_merge_with_none_dictionary(self):
        """Test merging dictionaries with a None dictionary."""
        left = {
            "num_calls": {
                "groups": {
                    "group_1": {
                        "value": 71.0,
                        "threshold": 100.0,
                    },
                },
            },
        }

        right = None

        with pytest.raises(Exception):
            merge_dicts(left, right)

    def test_merge_with_int_values(self):
        """Test merging dictionaries with a None dictionary."""
        left = {
            "num_calls": {
                "groups": {
                    "group_1": {
                        "value": 71.0,
                        "threshold": 100.0,
                    },
                },
            },
        }

        right = 1

        with pytest.raises(Exception):
            merge_dicts(left, right)

    def test_merge_with_two_empty_dictionary(self):
        """Test merging dictionaries with two empty dictionaries."""
        left = {}
        right = {}

        result = merge_dicts(left, right)

        assert result == {}

    def test_merge_with_conflicting_keys(self):
        """Test merging dictionaries with an empty dictionary."""
        left = {
            "num_calls": {
                "groups": {
                    "group_1": {
                        "value": 71.0,
                        "threshold": 100.0,
                    },
                },
            },
        }

        right = {
            "num_calls": {
                "groups": {
                    "group_1": {
                        "value": 11.0,
                    },
                },
            },
        }

        with pytest.raises(Exception) as e:
            merge_dicts(left, right)

        # Make sure the error message contains the conflicting key
        assert "num_calls.groups.group_1.value" in e.value.args[0]

    def test_merge_with_different_tree_keys(self):
        """Test merging dictionaries with an empty dictionary."""
        left = {
            "num_calls1": {
                "groups": {
                    "group_1": {
                        "value": 71.0,
                    },
                },
            },
        }

        right = {
            "num_calls2": {
                "groups": {
                    "group_1": {
                        "value": 72.0,
                    },
                },
            },
        }

        result = merge_dicts(left, right)

        assert result == {
            "num_calls1": {
                "groups": {
                    "group_1": {
                        "value": 71.0,
                    },
                },
            },
            "num_calls2": {
                "groups": {
                    "group_1": {
                        "value": 72.0,
                    },
                },
            },
        }

    def test_merge_with_different_leaf_values(self):
        """Test merging dictionaries with an empty dictionary."""
        left = {
            "num_calls1": {
                "groups": {
                    "group_1": {
                        "value": 71.0,
                    },
                },
            },
        }

        right = {
            "num_calls1": {
                "groups": {
                    "group_1": {
                        "threshold": 72.0,
                    },
                },
            },
        }

        result = merge_dicts(left, right)

        assert result == {
            "num_calls1": {
                "groups": {
                    "group_1": {
                        "value": 71.0,
                        "threshold": 72.0,
                    },
                },
            },
        }

    def test_merge_with_different_branch_values(self):
        """Test merging dictionaries with an empty dictionary."""
        left = {
            "num_calls1": {
                "groups": {
                    "group_2": {
                        "value": 71.0,
                    },
                },
            },
        }

        right = {
            "num_calls1": {
                "groups": {
                    "group_1": {
                        "value": 72.0,
                    },
                },
            },
        }

        result = merge_dicts(left, right)

        assert result == {
            "num_calls1": {
                "groups": {
                    "group_2": {
                        "value": 71.0,
                    },
                    "group_1": {
                        "value": 72.0,
                    },
                },
            },
        }
