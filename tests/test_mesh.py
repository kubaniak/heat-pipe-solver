import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add src to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, src_path)
sys.path.insert(0, os.path.join(src_path, '2d'))

# We must mock sys.modules dynamically during the tests to avoid polluting the test suite.
# Let's create a fixture or just use a context manager to import the module safely.

@pytest.fixture
def mock_mesh_module():
    import importlib

    mock_modules = {
        'fipy': MagicMock(),
        'fipy.meshes.nonUniformGrid2D': MagicMock(),
        'fipy.meshes': MagicMock(),
        'numpy': MagicMock(),
        'matplotlib': MagicMock(),
        'matplotlib.pyplot': MagicMock(),
        'params': MagicMock()
    }

    with patch.dict('sys.modules', mock_modules):
        # Force reload to ensure it uses the mocked modules if it was already loaded
        if '2d.mesh' in sys.modules:
            del sys.modules['2d.mesh']
        mesh_module = importlib.import_module("2d.mesh")
        yield mesh_module

def test_generate_mesh_2d_valid(mock_mesh_module):
    """Test generating a mesh with valid positive dimensions."""
    with patch.object(mock_mesh_module, 'Grid2D') as mock_grid2d:
        L_x, L_y = 10.0, 5.0
        nx, ny = 100, 50

        # generate_mesh_2d calls Grid2D
        mesh = mock_mesh_module.generate_mesh_2d(L_x, L_y, nx, ny)

        # Since Grid2D is mocked, mesh will be a MagicMock object.
        # We check if Grid2D was called with correct arguments
        mock_grid2d.assert_called_once_with(dx=L_x/nx, dy=L_y/ny, nx=nx, ny=ny)

def test_generate_mesh_2d_zero_cells(mock_mesh_module):
    """Test edge cases where nx=0 or ny=0 should raise ZeroDivisionError."""
    L_x, L_y = 10.0, 5.0

    # nx = 0
    with pytest.raises(ZeroDivisionError):
        mock_mesh_module.generate_mesh_2d(L_x, L_y, 0, 50)

    # ny = 0
    with pytest.raises(ZeroDivisionError):
        mock_mesh_module.generate_mesh_2d(L_x, L_y, 100, 0)
