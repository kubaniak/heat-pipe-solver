import sys
import math
import os
from unittest.mock import MagicMock

# Mock fipy because it's not installed in this environment
sys.modules['fipy'] = MagicMock()
sys.modules['fipy.tools'] = MagicMock()
sys.modules['fipy.tools.numerix'] = MagicMock()

# Adjust path to import from src/2d
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/2d')))

from material_properties import rho_sodium_s

def test_rho_sodium_s_freezing_point():
    """Test the density of solid sodium at 0 C (273.15 K)."""
    # At T = 273.15, the equation becomes 972.70 - 0.2154 * (273.15 - 273.15)
    expected_density = 972.70
    assert math.isclose(rho_sodium_s(273.15), expected_density, rel_tol=1e-9)

def test_rho_sodium_s_higher_temp():
    """Test the density of solid sodium at a higher temperature."""
    # At T = 373.15 (100 C)
    # 972.70 - 0.2154 * (100) = 972.70 - 21.54 = 951.16
    expected_density = 951.16
    assert math.isclose(rho_sodium_s(373.15), expected_density, rel_tol=1e-9)

def test_rho_sodium_s_lower_temp():
    """Test the density of solid sodium at a lower temperature."""
    # At T = 173.15 (-100 C)
    # 972.70 - 0.2154 * (-100) = 972.70 + 21.54 = 994.24
    expected_density = 994.24
    assert math.isclose(rho_sodium_s(173.15), expected_density, rel_tol=1e-9)
