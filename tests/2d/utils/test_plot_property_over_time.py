import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import sys
import importlib.util
from unittest.mock import MagicMock

# Mock matplotlib before importing the module
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()

# We need to use importlib because of the "2d" directory name which is not a valid python identifier
spec = importlib.util.spec_from_file_location("plot_property_over_time", "src/2d/utils/plot_property_over_time.py")
plot_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_module)

process_data_file = plot_module.process_data_file

def test_process_data_file_success():
    # Create a temporary file with mock data
    with tempfile.NamedTemporaryFile(suffix="_h5_1.23_s.csv", delete=False, mode='w') as f:
        file_path = f.name
        # Write some data mimicking the expected CSV format
        # Columns: 0, 1, 2 (drop), 3, 4 (drop), 5 -> total 6 columns initially
        f.write("Col0,Col1,Col2ToDrop,Col3,Col4ToDrop,Col5\n")
        f.write("1.0,10.0,DROP_THIS,100.0,DROP_THIS,1000.0\n")
        f.write("0.5,5.0,DROP_THIS,50.0,DROP_THIS,500.0\n")
        f.write("2.0,20.0,DROP_THIS,200.0,DROP_THIS,2000.0\n")

    try:
        # Call the function
        df, time = process_data_file(file_path)

        # Assertions
        # 1. Check time extraction
        assert time == 1.23

        # 2. Check columns after dropping and renaming
        expected_columns = ['Axial Location (m)', 'Density (kg/m^3)',
                            'Specific Heat (J/kg-K)', 'Thermal Conductivity (W/m-K)']
        assert list(df.columns) == expected_columns

        # 3. Check data
        # Data was originally:
        # Col0, Col1, Col3, Col5
        # 1.0, 10.0, 100.0, 1000.0
        # 0.5, 5.0,  50.0,  500.0
        # 2.0, 20.0, 200.0, 2000.0

        # Should be sorted by 'Axial Location (m)' (which corresponds to Col0)
        assert df.iloc[0]['Axial Location (m)'] == 0.5
        assert df.iloc[1]['Axial Location (m)'] == 1.0
        assert df.iloc[2]['Axial Location (m)'] == 2.0

        assert df.iloc[0]['Density (kg/m^3)'] == 5.0
        assert df.iloc[1]['Density (kg/m^3)'] == 10.0
        assert df.iloc[2]['Density (kg/m^3)'] == 20.0

    finally:
        # Clean up
        os.remove(file_path)

def test_process_data_file_no_time_match():
    # Create a temporary file without the expected time format in name
    with tempfile.NamedTemporaryFile(suffix="_wrong_format.csv", delete=False, mode='w') as f:
        file_path = f.name
        f.write("Col0,Col1,Col2ToDrop,Col3,Col4ToDrop,Col5\n")
        f.write("1.0,10.0,DROP_THIS,100.0,DROP_THIS,1000.0\n")

    try:
        # Call the function
        df, time = process_data_file(file_path)

        # Assertions
        assert time is None
        assert len(df.columns) == 4 # still drops and renames
    finally:
        os.remove(file_path)
