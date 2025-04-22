# Assignment-2B-Program
 Tree Based Routing Guidance System

# Traffic Data Visualization App

An interactive web application that displays traffic monitoring sites on a map using Streamlit and Folium. This application allows you to explore traffic data collected from various monitoring sites.

## Features

- Interactive map with traffic monitoring site markers
- Search for locations by name using geocoding
- Filter sites based on various criteria
- View detailed information about each site, including traffic patterns
- Multiple map styles (street map, terrain, etc.)
- Responsive design for different screen sizes

## Setup and Installation

1. Clone this repository
2. If you're using Python 3.12 or newer, run the fix-environment script first:
   - On Windows: `fix_environment.bat`
   - On Linux/Mac: `bash fix_environment.sh`
3. Otherwise, install the required packages directly:
   ```
   pip install -r requirements.txt
   ```
4. Process the traffic data (if not already processed):
   ```
   python main/data_processor.py
   ```
5. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Troubleshooting

If you encounter the error `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'`:

1. This is a known issue with Python 3.12+ and certain package versions
2. Run the fix-environment script provided in this repository
3. If the issue persists, try installing a newer version of NumPy:
   ```
   pip install numpy>=1.26.0
   pip install --upgrade setuptools
   ```

## Data Structure

The application expects the following data structure:

```
data/
├── raw/
│   ├── Scats Data October 2006.xlsx
│   └── Traffic_Count_Locations_with_LONG_LAT.csv
└── processed/
    ├── sites_metadata.json
    └── sites/
        └── [site_id]/
            ├── daily_pattern.csv
            ├── daily_pattern.png
            └── various .npy and .pkl files
```

## Usage

1. Open the application in your web browser (default: http://localhost:8501)
2. Use the sidebar to search for locations or filter sites
3. Click on map markers to view site information
4. Select a site from the dropdown to see detailed traffic patterns

## Dependencies

- streamlit
- folium
- streamlit-folium
- pandas
- numpy
- requests
- matplotlib
- networkx
- scikit-learn

## License

This project is licensed under the MIT License - see the LICENSE file for details.
