#!/usr/bin/env python3
"""
DRIFT Dataset Downloader

Download traffic trajectory data from the DRIFT dataset by site.

Usage:
    python eda/scripts/download_data.py --site A B C    # Download specific sites
    python eda/scripts/download_data.py --all           # Download all sites
    python eda/scripts/download_data.py --site A --output ./data/drift  # Custom output directory
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
import warnings

# Disable progress bars for cleaner output
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
warnings.filterwarnings('ignore')

# Dataset configuration
REPO_ID = "Hj-Lee/The-DRIFT"
DEFAULT_OUTPUT_DIR = "../data/drift"  # Relative to eda/scripts/


def get_available_sites():
    """Get list of all available sites in the DRIFT dataset."""
    print("🔍 Discovering available sites...")
    files = list_repo_files(REPO_ID, repo_type="dataset")
    csv_files = [f for f in files if f.endswith('.csv')]
    
    sites = set()
    for f in csv_files:
        if '/' in f:
            site = f.split('/')[0]
            sites.add(site)
    
    return sorted(sites), csv_files


def download_site_data(site_name, csv_files, output_dir, max_files=None):
    """Download all CSV files for a specific site."""
    site_files = [f for f in csv_files if f.startswith(f"{site_name}/") and f.endswith('.csv')]
    
    if not site_files:
        print(f"⚠️  No files found for Site {site_name}")
        return 0
    
    # Limit number of files if specified
    total_available = len(site_files)
    if max_files is not None and max_files < len(site_files):
        site_files = site_files[:max_files]
        print(f"\n📦 Downloading Site {site_name} ({len(site_files)}/{total_available} files)...")
    else:
        print(f"\n📦 Downloading Site {site_name} ({len(site_files)} files)...")
    
    # Create site directory
    site_dir = Path(output_dir) / f"site_{site_name}"
    site_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_count = 0
    for file in site_files:
        try:
            # Download from HuggingFace
            file_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=file,
                repo_type="dataset"
            )
            
            # Read and save to output directory
            df = pd.read_csv(file_path)
            
            # Save to output directory
            output_file = site_dir / Path(file).name
            df.to_csv(output_file, index=False)
            
            print(f"  ✓ {Path(file).name}: {df.shape[0]:,} rows, {df.shape[1]} columns")
            downloaded_count += 1
            
        except Exception as e:
            print(f"  ✗ {Path(file).name}: {e}")
    
    print(f"✅ Site {site_name}: Downloaded {downloaded_count}/{len(site_files)} files to {site_dir}")
    return downloaded_count


def main():
    parser = argparse.ArgumentParser(
        description="Download DRIFT dataset by site",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eda/scripts/download_data.py --site A              # Download Site A only
  python eda/scripts/download_data.py --site A B C          # Download Sites A, B, and C
  python eda/scripts/download_data.py --all                 # Download all sites
  python eda/scripts/download_data.py --site A --files 5    # Download first 5 files from Site A
  python eda/scripts/download_data.py --site A --output ./my_data  # Custom output directory
        """
    )
    
    parser.add_argument(
        '--site',
        nargs='+',
        metavar='SITE',
        help='Site(s) to download (e.g., A B C)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all available sites'
    )
    
    parser.add_argument(
        '--output',
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available sites without downloading'
    )
    
    parser.add_argument(
        '--files',
        type=int,
        metavar='N',
        help='Download only first N files per site (useful for sampling)'
    )
    
    args = parser.parse_args()
    
    # Get available sites
    available_sites, csv_files = get_available_sites()
    
    print(f"\n📊 DRIFT Dataset Info:")
    print(f"   Repository: {REPO_ID}")
    print(f"   Available sites: {', '.join(available_sites)}")
    print(f"   Total files: {len(csv_files)}")
    
    # List only mode
    if args.list:
        print("\n📋 Available sites:")
        for site in available_sites:
            count = len([f for f in csv_files if f.startswith(f"{site}/")])
            print(f"   - Site {site}: {count} files")
        return
    
    # Determine which sites to download
    if args.all:
        sites_to_download = available_sites
        print(f"\n🌐 Downloading ALL sites...")
    elif args.site:
        sites_to_download = [s.upper() for s in args.site]
        # Validate sites
        invalid_sites = [s for s in sites_to_download if s not in available_sites]
        if invalid_sites:
            print(f"\n❌ Error: Invalid site(s): {', '.join(invalid_sites)}")
            print(f"   Available sites: {', '.join(available_sites)}")
            return
    else:
        parser.print_help()
        return
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir.absolute()}")
    
    # Download each site
    total_files = 0
    print("\n" + "="*60)
    
    if args.files:
        print(f"⚙️  Limiting to {args.files} file(s) per site\n")
    
    for site in sites_to_download:
        count = download_site_data(site, csv_files, output_dir, max_files=args.files)
        total_files += count
    
    print("="*60)
    print(f"\n🎉 Download complete!")
    print(f"   Sites downloaded: {len(sites_to_download)}")
    print(f"   Total files: {total_files}")
    print(f"   Location: {output_dir.absolute()}")


if __name__ == "__main__":
    main()

