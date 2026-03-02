import unittest
import tempfile
import os
import sys
import importlib.util

# Add src to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import params using importlib because of '2d' package name
spec = importlib.util.spec_from_file_location("params", os.path.join(os.path.dirname(__file__), '..', 'src', '2d', 'params.py'))
params = importlib.util.module_from_spec(spec)
sys.modules["params"] = params
spec.loader.exec_module(params)

class TestParams(unittest.TestCase):
    def test_load_params_invalid_json(self):
        # Create a temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            temp_file.write('{"invalid json: [1, 2, 3]')
            temp_file_path = temp_file.name

        try:
            # Clear params cache to ensure we read from the file
            params._params_cache = None

            # Test that loading the invalid JSON raises a ValueError
            with self.assertRaises(ValueError) as context:
                params.load_params(temp_file_path)

            self.assertIn(f"Invalid JSON in configuration file: {temp_file_path}", str(context.exception))
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

if __name__ == '__main__':
    unittest.main()
