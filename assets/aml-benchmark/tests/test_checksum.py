import os
import sys

package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, package_path)  

from package_3p.checksum import SHA256Checksum
import pytest

@pytest.fixture
def checksum1():
    return SHA256Checksum()

@pytest.fixture
def checksum2():
    return SHA256Checksum()

def test_different_for_different_lines(checksum1, checksum2):
    checksum1.update({'a': 1})
    checksum2.update({'a': 2})

    assert checksum1.digest() != checksum2.digest()

def test_independent_of_key_order(checksum1, checksum2):
    checksum1.update({'a': 1, 'b': 'x'})
    checksum2.update({'b': 'x', 'a': 1})

    assert checksum1.digest() == checksum2.digest()

def test_works_with_structures(checksum1):
    jsonl_line = {
        'a': [1,2,3],
        'b': {
            'x': 'y',
            'z': [7,8]
        }
    }
    checksum1.update(jsonl_line)
    
    _ = checksum1.digest()

def test_works_with_empty(checksum1):
    checksum1.update({})
    
    _ = checksum1.digest()

def test_works_with_none(checksum1):
    checksum1.update(None)
    
    _ = checksum1.digest()