"""
Batch processor for cancer cell survival analysis
"""

import os
import glob
from cancer_survival_analyzer import analyze_cancer_survival

def process_all_nd2_files(directory_path):
    """Process all ND2 files for survival analysis."""
    # Find all ND2 files
    nd2_files = glob.glob(os.path.join(directory_path, "*.nd2"))
    
    if not nd2_files:
        print(f"No ND2 files found in {directory_path}")
        return
    
    print(f"Found {len(nd2_files)} ND2 files")
    print("="*60)
    
    # Process each file
    for i, nd2_file in enumerate(nd2_files):
        print(f"\nProcessing file {i+1}/{len(nd2_files)}: {os.path.basename(nd2_file)}")
        print("-"*60)
        
        try:
            analyze_cancer_survival(nd2_file)
            print(f"✓ Successfully processed {os.path.basename(nd2_file)}")
        except Exception as e:
            print(f"✗ Error processing {os.path.basename(nd2_file)}: {str(e)}")
    
    print("\n" + "="*60)
    print("Batch processing complete!")

if __name__ == "__main__":
    # Update this path to your data directory
    data_directory = r"D:\New\BrainBites\Cell"
    process_all_nd2_files(data_directory)