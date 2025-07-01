from nk_cancer_analyzer import ND2Analyzer

# Run the test with the existing analyzer
def test_phase2(file_path):
    analyzer = ND2Analyzer(file_path)
    if analyzer.load_file():
        print("✓ File loaded successfully!")
        analyzer.visualize_timepoint(timepoint=0)
        analyzer.close()
    else:
        print("✗ Failed to load file.")

# Run the test
test_phase2(r"D:\New\BrainBites\Cell\2.nd2")