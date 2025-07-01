"""
Complete test script for NK Cancer Cell Analysis
Run this to test both Phase 1 and Phase 2
"""

import os
import sys

# Test Phase 1
def test_phase1():
    print("="*60)
    print("TESTING PHASE 1 - Basic ND2 Reading and Visualization")
    print("="*60)
    
    from nk_cancer_analyzer import ND2Analyzer
    
    # Your ND2 file
    nd2_file = r"D:\New\BrainBites\Cell\2.nd2"
    
    # Check if file exists
    if not os.path.exists(nd2_file):
        print(f"ERROR: Cannot find file: {nd2_file}")
        return False
    
    print(f"Loading file: {nd2_file}")
    
    # Create analyzer
    analyzer = ND2Analyzer(nd2_file)
    
    # Load file
    if analyzer.load_file():
        print("✓ File loaded successfully!\n")
        
        # Show first timepoint
        print("Showing first timepoint...")
        analyzer.visualize_timepoint(0)
        
        # Show time montage
        print("\nCreating time montage...")
        analyzer.create_time_montage(channel='TRITC', num_timepoints=6)
        
        analyzer.close()
        return True
    else:
        print("✗ Failed to load file")
        return False

# Test Phase 2
def test_phase2():
    print("\n" + "="*60)
    print("TESTING PHASE 2 - Cell Detection and Analysis")
    print("="*60)
    
    from nk_cancer_analyzer import ND2Analyzer
    from nk_cancer_analyzer_phase2 import CellDetector, CellTracker, analyze_cell_dynamics
    
    # Your ND2 file
    nd2_file = r"D:\New\BrainBites\Cell\2.nd2"
    
    print(f"Loading file: {nd2_file}")
    
    # Load file
    analyzer = ND2Analyzer(nd2_file)
    if not analyzer.load_file():
        print("Failed to load file!")
        return False
    
    # Create detector and tracker
    print("\nInitializing cell detector...")
    detector = CellDetector(min_cell_area=50, max_cell_area=2000)
    tracker = CellTracker(max_distance=50)
    
    # Test on first frame
    print("\nDetecting cells in first frame...")
    tritc_frame = analyzer.get_frame(0, 'TRITC')
    labeled_cells, cell_properties = detector.detect_cells_adaptive(tritc_frame)
    
    print(f"✓ Detected {len(cell_properties)} cells")
    
    # Visualize detection
    print("\nVisualizing cell detection...")
    detector.visualize_detection(tritc_frame, labeled_cells, cell_properties,
                               title="Cell Detection - Frame 0")
    
    # Analyze cell properties
    print("\nAnalyzing cell properties...")
    detector.analyze_intensity_distribution(cell_properties)
    
    # Analyze time series (first 5 frames)
    print("\nAnalyzing time series (first 5 frames)...")
    results_df = analyze_cell_dynamics(analyzer, detector, tracker, 
                                     start_frame=0, end_frame=5)
    
    print("\nResults summary:")
    print(results_df)
    
    analyzer.close()
    return True

# Main function
def main():
    print("NK CELL CANCER ANALYSIS - COMPLETE TEST")
    print("="*60)
    print("This will test both Phase 1 and Phase 2")
    print("Close each plot window to continue to the next test\n")
    
    # Check current directory
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in directory: {[f for f in os.listdir('.') if f.endswith('.py') or f.endswith('.nd2')]}\n")
    
    # Run Phase 1 test
    phase1_success = test_phase1()
    
    if phase1_success:
        # Ask user if they want to continue to Phase 2
        input("\nPress Enter to continue to Phase 2 testing...")
        
        # Run Phase 2 test
        phase2_success = test_phase2()
        
        if phase2_success:
            print("\n✓ All tests completed successfully!")
        else:
            print("\n✗ Phase 2 tests failed")
    else:
        print("\n✗ Phase 1 tests failed - fix this before testing Phase 2")

if __name__ == "__main__":
    main()