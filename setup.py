#!/usr/bin/env python3
"""
Setup script for StatisticalDependencyRegularization project.

This script:
1. Initializes conda environment if it doesn't exist
2. Adds project path to the environment
3. Downloads AIF360 datasets used in the project
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path

# Project configuration
PROJECT_NAME = "StatisticalDependencyRegularization"
PROJECT_ROOT = Path(__file__).parent.absolute()
CONDA_ENV_NAME = "StatisticalDependencyRegularization"
ENVIRONMENT_YML = PROJECT_ROOT / "environment.yml"

# AIF360 data paths
AIF360_DATA_PATH = Path.home() / f".conda/envs/{CONDA_ENV_NAME}/lib/python3.11/site-packages/aif360/data/raw"

# Dataset URLs and configurations
DATASETS = {
    'adult': {
        'urls': [
            'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
            'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
            'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names'
        ],
        'path': AIF360_DATA_PATH / 'adult'
    },
    'german': {
        'urls': [
            'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
            'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc'
        ],
        'path': AIF360_DATA_PATH / 'german'
    },
    'compas': {
        'urls': [
            'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
        ],
        'path': AIF360_DATA_PATH / 'compas'
    },
    'bank': {
        'urls': [
            'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip'
        ],
        'path': AIF360_DATA_PATH / 'bank',
        'extract': True,
        'extract_files': ['bank-additional/bank-additional-full.csv', 'bank-additional/bank-additional-names.txt']
    }
}


def run_command(cmd, check=True, shell=True):
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=shell, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error output: {e.stderr}")
        if check:
            raise
        return e


def check_conda_installed():
    """Check if conda is installed."""
    try:
        subprocess.run(['conda', '--version'], check=True, 
                      capture_output=True, text=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def environment_exists(env_name):
    """Check if conda environment exists."""
    try:
        result = subprocess.run(['conda', 'env', 'list'], 
                              capture_output=True, text=True, check=True)
        return env_name in result.stdout
    except subprocess.CalledProcessError:
        return False


def create_conda_environment():
    """Create conda environment from environment.yml."""
    print(f"Creating conda environment '{CONDA_ENV_NAME}'...")
    
    if not ENVIRONMENT_YML.exists():
        print(f"Error: {ENVIRONMENT_YML} not found!")
        sys.exit(1)
    
    if environment_exists(CONDA_ENV_NAME):
        print(f"Environment '{CONDA_ENV_NAME}' already exists.")
        response = input("Do you want to update it? (y/N): ")
        if response.lower() == 'y':
            run_command(f"conda env update -f {ENVIRONMENT_YML}")
        return
    
    run_command(f"conda env create -f {ENVIRONMENT_YML}")
    print(f"Environment '{CONDA_ENV_NAME}' created successfully!")


def configure_pythonpath():
    """Configure PYTHONPATH for the conda environment."""
    print("Configuring PYTHONPATH...")
    
    # Set PYTHONPATH variable for the environment
    cmd = f'conda env config vars set PYTHONPATH="${{PYTHONPATH}}:{PROJECT_ROOT}" -n {CONDA_ENV_NAME}'
    run_command(cmd)
    
    print(f"PYTHONPATH configured to include: {PROJECT_ROOT}")
    print("Note: You need to reactivate the environment for changes to take effect.")


def download_file(url, destination):
    """Download a file from URL to destination."""
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"Downloaded to {destination}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def extract_zip(zip_path, extract_to, files_to_extract=None):
    """Extract specific files from a zip archive."""
    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            if files_to_extract:
                for file in files_to_extract:
                    try:
                        zip_ref.extract(file, extract_to)
                        # Move extracted file to the target directory
                        src = extract_to / file
                        dst = extract_to / Path(file).name
                        if src.exists():
                            shutil.move(str(src), str(dst))
                        print(f"Extracted: {file}")
                    except KeyError:
                        print(f"File {file} not found in archive")
            else:
                zip_ref.extractall(extract_to)
        
        # Clean up extracted directories
        for item in extract_to.iterdir():
            if item.is_dir() and item.name != extract_to.name:
                shutil.rmtree(item)
        
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False


def download_datasets():
    """Download all required datasets."""
    print("Downloading AIF360 datasets...")
    
    if not AIF360_DATA_PATH.exists():
        print(f"AIF360 data path not found: {AIF360_DATA_PATH}")
        print("Make sure the conda environment is created and AIF360 is installed.")
        return False
    
    success_count = 0
    total_datasets = len(DATASETS)
    
    for dataset_name, config in DATASETS.items():
        print(f"\n--- Downloading {dataset_name} dataset ---")
        
        dataset_path = config['path']
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        dataset_success = True
        
        for url in config['urls']:
            filename = Path(url).name
            destination = dataset_path / filename
            
            # Skip if file already exists
            if destination.exists():
                print(f"File {filename} already exists, skipping...")
                continue
            
            if not download_file(url, destination):
                dataset_success = False
                break
            
            # Handle zip extraction for bank dataset
            if config.get('extract') and filename.endswith('.zip'):
                if extract_zip(destination, dataset_path, config.get('extract_files')):
                    # Remove the zip file after extraction
                    destination.unlink()
                else:
                    dataset_success = False
        
        if dataset_success:
            success_count += 1
            print(f"✓ {dataset_name} dataset downloaded successfully!")
        else:
            print(f"✗ Failed to download {dataset_name} dataset")
    
    print(f"\nDataset download summary: {success_count}/{total_datasets} successful")
    return success_count == total_datasets


def install_project_in_dev_mode():
    """Install the project in development mode."""
    print("Installing project in development mode...")
    
    # Create a basic setup.py content for setuptools
    setup_content = '''
from setuptools import setup, find_packages

setup(
    name="redlining-penalty-regularizer",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[],
    author="Your Name",
    description="Statistical Dependency Regularization for Fair Machine Learning",
)
'''
    
    # Write temporary setup.py for installation
    temp_setup = PROJECT_ROOT / "temp_setup.py"
    with open(temp_setup, 'w') as f:
        f.write(setup_content)
    
    try:
        # Install in development mode
        cmd = f"conda run -n {CONDA_ENV_NAME} pip install -e . --no-deps"
        run_command(cmd)
        print("✓ Project installed in development mode!")
        return True
    except Exception as e:
        print(f"✗ Failed to install project: {e}")
        return False
    finally:
        # Clean up temporary file
        if temp_setup.exists():
            temp_setup.unlink()


def main():
    """Main setup function."""
    print(f"=== StatisticalDependencyRegularization Setup ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Environment name: {CONDA_ENV_NAME}")
    
    # Check if conda is installed
    if not check_conda_installed():
        print("Error: Conda is not installed or not in PATH!")
        print("Please install Miniconda or Anaconda first.")
        sys.exit(1)
    
    # Step 1: Create/update conda environment
    try:
        create_conda_environment()
    except Exception as e:
        print(f"Failed to create conda environment: {e}")
        sys.exit(1)
    
    # Step 2: Configure PYTHONPATH
    try:
        configure_pythonpath()
    except Exception as e:
        print(f"Failed to configure PYTHONPATH: {e}")
        print("You can manually set it later with:")
        print(f'conda env config vars set PYTHONPATH="${{PYTHONPATH}}:{PROJECT_ROOT}" -n {CONDA_ENV_NAME}')
    
    # Step 3: Install project in development mode
    try:
        install_project_in_dev_mode()
    except Exception as e:
        print(f"Failed to install project: {e}")
    
    # Step 4: Download datasets
    try:
        download_datasets()
    except Exception as e:
        print(f"Failed to download datasets: {e}")
        print("You can download them manually later.")
    
    print("\n=== Setup Complete ===")
    print(f"To activate the environment, run:")
    print(f"conda activate {CONDA_ENV_NAME}")
    print("\nTo verify the setup, try:")
    print("python -c 'from dataset_readers import AdultDatasetReader; print(\"✓ Import successful!\")'")


if __name__ == "__main__":
    main()
