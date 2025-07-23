import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re

def process_data_file(file_path):
    """
    Reads a properties CSV file, drops redundant columns, 
    renames columns, sorts by axial location, and extracts the simulation time from the filename.
    """
    df = pd.read_csv(file_path)
    # Drop the 3rd and 5th columns (0-indexed)
    df = df.drop(df.columns[[2, 4]], axis=1)
    
    # Rename columns
    df.columns = ['Axial Location (m)', 'Density (kg/m^3)', 
                  'Specific Heat (J/kg-K)', 'Thermal Conductivity (W/m-K)']
    
    # Sort by Axial Location
    df = df.sort_values(by='Axial Location (m)')
    
    # Extract time from filename
    match = re.search(r'_h5_(\d+\.\d+)_s\.csv$', file_path)
    time = float(match.group(1)) if match else None
    
    return df, time

def plot_property_over_time(file_paths, property_column_name, y_label, title, output_filename, output_dir="plots", ax=None, label_prefix="", plot_kwargs=None):
    """
    Plots a specified property vs. axial location for different times.
    Can plot on a provided matplotlib.axes.Axes object or create a new figure.
    Allows specifying a label_prefix for legend entries and custom plot_kwargs.
    """
    created_figure = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6)) # Use fig, ax = plt.subplots()
        created_figure = True

    if plot_kwargs is None:
        plot_kwargs = {}

    if not os.path.exists(output_dir) and created_figure:
        os.makedirs(output_dir)
    
    all_times_data = []
    for file_path in file_paths:
        df, time = process_data_file(file_path)
        if time is not None:
            all_times_data.append({'time': time, 'df': df})
            
    # Sort data by time to ensure consistent plotting order for legend
    all_times_data.sort(key=lambda x: x['time'])

    for data_entry in all_times_data:
        time = data_entry['time']
        df = data_entry['df']
        ax.plot(df['Axial Location (m)'], df[property_column_name], label=f'{label_prefix}{time} s', **plot_kwargs)
    
    # Sort legend by time (handles cases where lines might be added out of order by other means if ax is passed)
    handles, labels = ax.get_legend_handles_labels()
    
    # Filter out duplicate labels if any (can happen if ax is reused and function called multiple times)
    unique_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels[label] = handle
    
    # Create a list of (time, handle, label) tuples from unique labels
    legend_info = []
    for label, handle in unique_labels.items():
        # Extract time for sorting. Assumes label format '{time} s'
        try:
            time_str = label.split(' ')[0]
            legend_info.append((float(time_str), handle, label))
        except ValueError:
            # Handle labels not matching the expected format (e.g., from other plots on the same ax)
            legend_info.append((float('inf'), handle, label)) # Put non-time labels at the end
            
    legend_info.sort(key=lambda x: x[0])
    
    sorted_handles = [info[1] for info in legend_info]
    sorted_labels = [info[2] for info in legend_info]

    ax.set_title(title)
    ax.set_xlabel('Axial Location (m)')
    ax.set_ylabel(y_label)
    # if sorted_handles:
    #     ax.legend(sorted_handles, sorted_labels, title="Time") # Legend removed
    ax.grid(True)
    
    if created_figure:
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
        plt.show()
    else: # If ax is provided, ensure the legend is updated if new items were added
        # if handles: # Only update legend if there are new handles/labels from this call # Legend removed
            # current_legend = ax.get_legend() # Legend removed
            # if current_legend: # if a legend already exists, try to update it # Legend removed
            #     # This part can be tricky; for now, let's assume the calling script handles the final legend.
            #     # Or, we can simply re-apply it with all items.
            #     all_handles, all_labels = ax.get_legend_handles_labels()
            #     # Filter unique labels to avoid duplicates if called multiple times on same ax with same data
            #     unique_entries = {}
            #     for h, l in zip(all_handles, all_labels):
            #         if l not in unique_entries:
            #             unique_entries[l] = h
                
            #     # Re-sort all collected labels if needed (especially if mixing sim and Riccardo)
            #     # The legend_info sorting logic from earlier could be reused here if complex sorting is needed.
            #     # For now, a simple legend update:
            #     if unique_entries:
            #         ax.legend(unique_entries.values(), unique_entries.keys(), title="Time") # Legend removed

            # else: # If no legend exists, create one # Legend removed
            #      ax.legend(handles, labels, title="Time") # Legend removed
            pass # Legend removed, ensure block is not empty if all lines are commented


    return ax

if __name__ == '__main__':
    base_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Short_Properties_Riccardo', 'Properties')
    output_dir_main = "plots_riccardo"
    if not os.path.exists(output_dir_main):
        os.makedirs(output_dir_main)

    regions = {
        "VC": "Vapor Core",
        "Wick": "Wick",
        "Wall": "Wall"
    }

    properties_to_plot = [
        {
            'column_name': 'Density (kg/m^3)',
            'y_label': 'Density (kg/m^3)',
            'title_infix': 'Density',
            'output_suffix': 'density_riccardo.png'
        },
        {
            'column_name': 'Specific Heat (J/kg-K)',
            'y_label': 'Specific Heat (J/kg-K)',
            'title_infix': 'Specific Heat',
            'output_suffix': 'cp_riccardo.png'
        },
        {
            'column_name': 'Thermal Conductivity (W/m-K)',
            'y_label': 'Thermal Conductivity (W/m-K)',
            'title_infix': 'Thermal Conductivity',
            'output_suffix': 'k_riccardo.png'
        }
    ]

    for region_code, region_name in regions.items():
        file_pattern = os.path.join(base_path, f'{region_code}_properties_Faghri_ax1000_TS_bra_StepFct05_properties_h5_*.000_s.csv')
        region_files = glob.glob(file_pattern)

        if not region_files:
            print(f"No files found for region {region_name} ({region_code}) with pattern: {file_pattern}")
            continue
        else:
            print(f"Found {len(region_files)} files for region {region_name} ({region_code}). Processing...")

        for prop_info in properties_to_plot:
            plot_title = f'{region_name} {prop_info["title_infix"]} vs. Axial Location (Riccardo Data)'
            output_filename = f'{region_code.lower()}_{prop_info["output_suffix"]}'
            
            print(f"  Plotting {prop_info['title_infix']}...")
            plot_property_over_time(
                file_paths=region_files,
                property_column_name=prop_info['column_name'],
                y_label=prop_info['y_label'],
                title=plot_title,
                output_filename=output_filename,
                output_dir=output_dir_main
            )
            # Ensure plots are closed to free memory if many are generated in a loop
            plt.close('all') 

    print(f"All plots saved to {output_dir_main}")

