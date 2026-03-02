import pytest
import os
import sys

# Add src to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/2d')))

import params

def test_load_params_file_not_found():
    params._params_cache = None # reset cache
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        params.load_params("non_existent_file.json")

def test_get_param_group_file_not_found(monkeypatch):
    # monkeypatch os.path.join inside params module to return a non-existent file path
    original_join = os.path.join

    def mocked_join(*args, **kwargs):
        if len(args) > 0 and args[-1] == 'faghri.json':
            return "non_existent_file.json"
        return original_join(*args, **kwargs)

    monkeypatch.setattr(os.path, "join", mocked_join)

    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        params.get_param_group("some_group")
