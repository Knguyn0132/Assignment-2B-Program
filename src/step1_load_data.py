# Import required libraries
import pandas as pd
import os

def load_basic_data():
    # Print start message
    print("Attempting to load data files and examine their structure...\n")
    
    # Set up file paths for the data files
    data_dir = os.path.join('data', 'raw')
    # Path for traffic data Excel file
    scats_data_path = os.path.join(data_dir, 'Scats Data October 2006.xlsx')
    # Path for site listing Excel file
    sites_path = os.path.join(data_dir, 'SCATSSiteListingSpreadsheet_VicRoads.xlsx')
    # Path for locations CSV file
    locations_path = os.path.join(data_dir, 'Traffic_Count_Locations_with_LONG_LAT.csv')
    
    # Print header for SCATS data section
    print("=" * 50)
    print("EXAMINING: Scats Data October 2006.xlsx")
    print("=" * 50)

    try:
        # Open Excel file
        xls = pd.ExcelFile(scats_data_path, engine='openpyxl')
        # Look at each sheet in the Excel file
        for sheet_name in xls.sheet_names:
            print(f"\nExamining sheet: '{sheet_name}'")
            
            # Handle 'Data' sheet
            if sheet_name.lower() == "data":
                # Read data sheet starting from row 2
                df = pd.read_excel(xls, sheet_name=sheet_name, header=1)
                # Show table size
                print(f"  Shape: {df.shape}")
                # Show first 2 rows and first 15 columns
                print(df.iloc[0:2, 0:15])
            
            # Handle 'Summary of Data' sheet
            elif sheet_name.lower() == "summary of data":
                # Read summary sheet starting from row 3
                df = pd.read_excel(xls, sheet_name=sheet_name, header=2)
                # Show table size
                print(f"  Shape: {df.shape}")
                # Show first 3 rows
                print("  Sample rows:")
                print(df.head(3))
            
            # Handle 'Notes' sheet
            elif sheet_name.lower() == "notes":
                # Read notes sheet
                df = pd.read_excel(xls, sheet_name=sheet_name)
                # Show number of rows
                print(f"  Notes sheet: {df.shape[0]} rows")
    except Exception as e:
        # Show error if file can't be loaded
        print(f"Error loading SCATS data file: {e}")
    
    # Print header for Sites data section
    print("\n" + "=" * 50)
    print("EXAMINING: SCATSSiteListingSpreadsheet_VicRoads.xlsx")
    print("=" * 50)

    try:
        # Open sites Excel file
        xls = pd.ExcelFile(sites_path, engine='openpyxl')
        # Show all sheet names
        print("Available sheets:", xls.sheet_names)

        # Check if the needed sheet exists
        if "SCATS Site Numbers" in xls.sheet_names:
            print("\nSheet: SCATS Site Numbers")
            try:
                # Read sites data, skip first 9 rows
                df_sites = pd.read_excel(xls, sheet_name="SCATS Site Numbers", skiprows=9, engine='openpyxl')
                # Show table size
                print(f"  Shape: {df_sites.shape}")
                # Show first 5 rows
                print(df_sites.head(5))
            except Exception as se:
                # Show error if sheet can't be read
                print(f"  Failed reading SCATS Site Numbers: {se}")
        else:
            # Show message if sheet not found
            print("  Sheet 'SCATS Site Numbers' not found.")
    except Exception as e:
        # Show error if file can't be loaded
        print(f"Error loading SCATS Site Listing file: {e}")
    
    # Print header for Traffic Locations section
    print("\n" + "=" * 50)
    print("EXAMINING: Traffic_Count_Locations_with_LONG_LAT.csv")
    print("=" * 50)

    try:
        # Read CSV file
        traffic_locations = pd.read_csv(locations_path)
        # Show table size
        print(f"  Shape: {traffic_locations.shape}")
        # Show column names
        print("  Columns:")
        print(traffic_locations.columns.tolist())
        # Show first 3 rows
        print("  First 3 rows:")
        print(traffic_locations.head(3))
    except Exception as e:
        # Show error if file can't be loaded
        print(f"Error loading traffic locations CSV: {e}")

    # Print completion message
    print("\nAll files loaded and examined.")

# Run the function if this file is run directly
if __name__ == "__main__":
    load_basic_data()