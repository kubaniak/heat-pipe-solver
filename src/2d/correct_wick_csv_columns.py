import pandas as pd
import glob
import os

def swap_columns_in_wick_csv(file_path):
    """
    Reads a Wick CSV file, swaps the 2nd and 6th columns, 
    and saves the modified file back.
    Column indices are 0-based, so 2nd is index 1, and 6th is index 5.
    """
    try:
        # Read with no header initially to avoid issues if column names are duplicated
        # or to preserve the exact header line as is after modification.
        df = pd.read_csv(file_path, header=None)

        if df.shape[1] < 6:
            print(f"  Skipping {os.path.basename(file_path)}: File has fewer than 6 columns.")
            return

        # Column indices to swap (0-based)
        col1_idx = 1  # Second column
        col2_idx = 5  # Sixth column

        # Store the columns
        col1_data = df.iloc[:, col1_idx].copy()
        col2_data = df.iloc[:, col2_idx].copy()

        # Perform the swap
        df.iloc[:, col1_idx] = col2_data
        df.iloc[:, col2_idx] = col1_data

        # Save the modified DataFrame back to the CSV file, without index and header
        # to preserve the original header row if it was read as data (header=None)
        df.to_csv(file_path, index=False, header=False)
        print(f"  Corrected columns in: {os.path.basename(file_path)}")

    except Exception as e:
        print(f"  Error processing {os.path.basename(file_path)}: {e}")

if __name__ == '__main__':
    # Construct the path relative to the script's location or use an absolute path
    # Assuming the script is in src/2d/ and Properties_riccardo is two levels up
    script_dir = os.path.dirname(__file__)
    base_path = os.path.join(script_dir, '..', '..', 'Short_Properties_Riccardo', 'Properties')
    
    # Find all Wick property files
    # Adjust the pattern if the naming convention is different
    wick_files_pattern = os.path.join(base_path, '*Wick*_properties_Faghri_ax1000_TS_bra_StepFct05_properties_h5_*.000_s.csv')
    wick_files = glob.glob(wick_files_pattern)

    if not wick_files:
        print(f"No Wick CSV files found matching pattern: {wick_files_pattern}")
    else:
        print(f"Found {len(wick_files)} Wick CSV files. Starting column correction...")
        for file_path in wick_files:
            swap_columns_in_wick_csv(file_path)
        print("Finished processing all found Wick CSV files.")
