#!/bin/bash
set -e

# Default to python3, allow override
PYTHON_EXE=${PYTHON_EXE:-python3}

if ! command -v "$PYTHON_EXE" &> /dev/null; then
    echo "Command '$PYTHON_EXE' not found."
    
    if command -v apt-get &> /dev/null; then
        echo "Attempting to install python3-full, python3-pip, and python3-venv..."
        # sudo is required, assuming the user has permissions or is running as root (CI)
        sudo apt-get update
        sudo apt-get install -y python3-full python3-pip python3-venv
    elif command -v yum &> /dev/null; then
        sudo yum install -y python3 python3-pip
    elif command -v brew &> /dev/null; then
        brew install python
    else
        echo "Error: Python not found and package manager could not be identified."
        exit 1
    fi
fi

# Determine absolute path
PYTHON_PATH=$(which "$PYTHON_EXE")
echo "Using python: $PYTHON_PATH"

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    "$PYTHON_EXE" -m venv venv
fi

# Activate venv
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    source venv/Scripts/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install scipy

echo "Setup complete."
echo "To use with Julia, set: export JULIA_PYTHONCALL_EXE=$(pwd)/venv/bin/python"
