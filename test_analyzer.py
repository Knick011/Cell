"""
Test script for the NK Cell Cancer Analyzer
"""

from nk_cancer_analyzer import ND2Analyzer
import os

def test_single_file():
    """Test the analyzer with a single ND2 file."""
    
    # IMPORTANT: Replace this with the actual path to your ND2 file
    # Example paths:
    # Windows: r"C:\Users\YourName\Documents\Data\experiment1.nd2"
    # Mac/Linux: "/Users/YourName/Documents/Data/experiment1.nd2"
    
    nd2_file = r"D:\New\BrainBites\Cell\2.nd2"  # <-- CHANGE THIS!
    
    # Check if file exists
    if not os.path.exists(nd2_file):
        print(f"ERROR: File not found: {nd2_file}")
        print("Please update the file path in this script!")
        return
    
    print(f"Testing with file: {nd2_file}")
    print("-" * 50)
    
    # Create analyzer instance
    analyzer = ND2Analyzer(nd2_file)
    
    # Load the file
    if analyzer.load_file():
        print("\n✓ File loaded successfully!")
        
        # Test 1: Visualize the first timepoint
        print("\nTest 1: Showing first timepoint...")
        analyzer.visualize_timepoint(timepoint=0)
        
        # Test 2: Visualize a middle timepoint
        middle_timepoint = analyzer.metadata['frames'] // 2
        print(f"\nTest 2: Showing middle timepoint (t={middle_timepoint})...")
        analyzer.visualize_timepoint(timepoint=middle_timepoint)
        
        # Test 3: Create time montage
        print("\nTest 3: Creating time montage...")
        analyzer.create_time_montage(channel='TRITC', num_timepoints=6)
        
        # Test 4: Save a visualization
        print("\nTest 4: Saving visualization to file...")
        analyzer.visualize_timepoint(timepoint=0, save=True)
        
        # Clean up
        analyzer.close()
        print("\n✓ All tests completed!")
        
    else:
        print("✗ Failed to load file. Check the error message above.")

def test_batch_loading():
    """Test loading multiple ND2 files in a directory."""
    
    # IMPORTANT: Replace this with your data directory
    data_directory = r"C:\path\to\your\nd2\files"  # <-- CHANGE THIS!
    
    if not os.path.exists(data_directory):
        print(f"ERROR: Directory not found: {data_directory}")
        return
    
    # Find all ND2 files
    nd2_files = list(os.glob(os.path.join(data_directory, "*.nd2")))
    
    if not nd2_files:
        print(f"No ND2 files found in {data_directory}")
        return
    
    print(f"Found {len(nd2_files)} ND2 files")
    print("-" * 50)
    
    # Test loading each file
    for i, file_path in enumerate(nd2_files[:3]):  # Test first 3 files
        print(f"\nFile {i+1}: {os.path.basename(file_path)}")
        
        analyzer = ND2Analyzer(file_path)
        if analyzer.load_file():
            print("✓ Loaded successfully")
            analyzer.close()
        else:
            print("✗ Failed to load")

if __name__ == "__main__":
    print("NK Cell Cancer Analyzer - Test Script")
    print("=" * 50)
    
    # Run the single file test
    test_single_file()
    
    # Uncomment to test batch loading
    # print("\n\nTesting batch loading...")
    # test_batch_loading()