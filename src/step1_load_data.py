import pandas as pd
import os

def load_basic_data():
    print("Attempting to load data files and examine their structure...\n")
    
    # File paths
    data_dir = os.path.join('data', 'raw')
    scats_data_path = os.path.join(data_dir, 'Scats Data October 2006.xlsx')  # updated to .xlsx
    sites_path = os.path.join(data_dir, 'SCATSSiteListingSpreadsheet_VicRoads.xlsx')
    locations_path = os.path.join(data_dir, 'Traffic_Count_Locations_with_LONG_LAT.csv')
    
    # === SCATS DATA OCTOBER (.xlsx version) ===
    print("=" * 50)
    print("EXAMINING: Scats Data October 2006.xlsx")
    print("=" * 50)

    try:
        xls = pd.ExcelFile(scats_data_path, engine='openpyxl')
        for sheet_name in xls.sheet_names:
            print(f"\nExamining sheet: '{sheet_name}'")
            if sheet_name.lower() == "data":
                df = pd.read_excel(xls, sheet_name=sheet_name, header=1)
                print(f"  Shape: {df.shape}")
                print(df.iloc[0:2, 0:15])
            elif sheet_name.lower() == "summary of data":
                df = pd.read_excel(xls, sheet_name=sheet_name, header=2)
                print(f"  Shape: {df.shape}")
                print("  Sample rows:")
                print(df.head(3))
            elif sheet_name.lower() == "notes":
                df = pd.read_excel(xls, sheet_name=sheet_name)
                print(f"  Notes sheet: {df.shape[0]} rows")
    except Exception as e:
        print(f"Error loading SCATS data file: {e}")
    
    # === SCATS SITE LISTING ===
    print("\n" + "=" * 50)
    print("EXAMINING: SCATSSiteListingSpreadsheet_VicRoads.xlsx")
    print("=" * 50)

    try:
        xls = pd.ExcelFile(sites_path, engine='openpyxl')
        print("Available sheets:", xls.sheet_names)

        if "SCATS Site Numbers" in xls.sheet_names:
            print("\nSheet: SCATS Site Numbers")
            try:
                df_sites = pd.read_excel(xls, sheet_name="SCATS Site Numbers", skiprows=9, engine='openpyxl')
                print(f"  Shape: {df_sites.shape}")
                print(df_sites.head(5))
            except Exception as se:
                print(f"  Failed reading SCATS Site Numbers: {se}")
        else:
            print("  Sheet 'SCATS Site Numbers' not found.")
    except Exception as e:
        print(f"Error loading SCATS Site Listing file: {e}")
    
    # === TRAFFIC LOCATION CSV ===
    print("\n" + "=" * 50)
    print("EXAMINING: Traffic_Count_Locations_with_LONG_LAT.csv")
    print("=" * 50)

    try:
        traffic_locations = pd.read_csv(locations_path)
        print(f"  Shape: {traffic_locations.shape}")
        print("  Columns:")
        print(traffic_locations.columns.tolist())
        print("  First 3 rows:")
        print(traffic_locations.head(3))
    except Exception as e:
        print(f"Error loading traffic locations CSV: {e}")

    print("\nAll files loaded and examined.")

if __name__ == "__main__":
    load_basic_data()
