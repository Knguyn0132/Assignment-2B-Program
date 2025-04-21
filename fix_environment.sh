#!/bin/bash
echo "Fixing Python environment for Python 3.12+ compatibility..."

# Upgrade pip
python -m ensurepip --upgrade
python -m pip install --upgrade pip

# Upgrade setuptools
python -m pip install --upgrade setuptools

# Install required packages
python -m pip install -r requirements.txt

echo
echo "Environment setup complete! You can now run the app using 'streamlit run app.py'"
echo 