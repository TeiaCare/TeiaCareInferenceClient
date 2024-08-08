#!/bin/bash

# Upgrade pip
python -m pip install --upgrade pip

# Create a virtual environment
python -m venv .venv

# Set CONAN_USER_HOME environment variable in the activate script
echo "export CONAN_USER_HOME=\$PWD" >> .venv/bin/activate

# Activate the virtual environment
source .venv/bin/activate

# Install the required packages
pip install -r scripts/requirements.txt

# Install pre-commit hooks
pre-commit install


# chmod +x scripts/env/setup.sh
# ./scripts/env/setup.sh
