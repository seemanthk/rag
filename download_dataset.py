"""
Helper script to download the Amazon Products dataset from Kaggle

Prerequisites:
1. Install kaggle: pip install kaggle
2. Set up Kaggle API credentials:
   - Go to https://www.kaggle.com/account
   - Create new API token (downloads kaggle.json)
   - Place kaggle.json in ~/.kaggle/ (Linux/Mac) or %USERPROFILE%\.kaggle\ (Windows)

Usage:
    python download_dataset.py
"""

import os
import sys
import subprocess
from pathlib import Path


def check_kaggle_installed():
    """Check if kaggle CLI is installed"""
    try:
        import kaggle
        return True
    except ImportError:
        return False


def check_kaggle_credentials():
    """Check if Kaggle API credentials are configured"""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    return kaggle_json.exists()


def download_dataset():
    """Download the Amazon Products dataset"""
    dataset_name = "lokeshparab/amazon-products-dataset"
    output_dir = "data"

    print("=" * 80)
    print("Amazon Products Dataset Downloader")
    print("=" * 80)

    # Check prerequisites
    if not check_kaggle_installed():
        print("\n❌ Error: Kaggle package not installed")
        print("\nTo install:")
        print("  pip install kaggle")
        sys.exit(1)

    if not check_kaggle_credentials():
        print("\n❌ Error: Kaggle API credentials not found")
        print("\nTo set up:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Save kaggle.json to:")
        print(f"   {Path.home() / '.kaggle' / 'kaggle.json'}")
        sys.exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n✓ Kaggle CLI installed")
    print(f"✓ Kaggle credentials found")
    print(f"\nDownloading dataset: {dataset_name}")
    print(f"Output directory: {output_dir}")
    print("\nThis may take a few minutes...\n")

    try:
        # Download dataset
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        api.dataset_download_files(
            dataset_name,
            path=output_dir,
            unzip=True
        )

        print("\n✓ Dataset downloaded successfully!")

        # List downloaded files
        print(f"\nFiles in {output_dir}:")
        for file in os.listdir(output_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(output_dir, file)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  - {file} ({size_mb:.2f} MB)")

        # Update config if needed
        csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
        if csv_files:
            csv_file = csv_files[0]
            print(f"\n✓ Dataset ready: {os.path.join(output_dir, csv_file)}")
            print(f"\nUpdate config.yaml to use this file:")
            print(f"  dataset:")
            print(f"    path: \"data/{csv_file}\"")

    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify Kaggle credentials are correct")
        print("3. Make sure you have accepted the dataset terms on Kaggle website")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Next steps:")
    print("=" * 80)
    print("1. Verify the CSV file in data/ directory")
    print("2. Update config.yaml with correct file path")
    print("3. Run: python demo.py")
    print("=" * 80)


if __name__ == "__main__":
    download_dataset()
