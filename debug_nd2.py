"""
Debug script to check if timepoints are actually changing
"""

from nk_cancer_analyzer import ND2Analyzer
import numpy as np
import matplotlib.pyplot as plt

# Your ND2 file
nd2_file = r"D:\New\BrainBites\Cell\2.nd2"

print("Testing timepoint access...")
print("="*50)

analyzer = ND2Analyzer(nd2_file)
analyzer.load_file()

# Test multiple timepoints
timepoints_to_test = [0, 10, 20, 30, 40]

# Get frames and calculate differences
frames = {}
for t in timepoints_to_test:
    if t < analyzer.metadata['frames']:
        frames[t] = analyzer.get_frame(t, 'TRITC')
        print(f"Timepoint {t}: Got frame with shape {frames[t].shape}")

# Check if frames are different
print("\nChecking frame differences:")
for i in range(len(timepoints_to_test)-1):
    t1, t2 = timepoints_to_test[i], timepoints_to_test[i+1]
    if t1 in frames and t2 in frames:
        diff = np.mean(np.abs(frames[t1] - frames[t2]))
        are_same = np.array_equal(frames[t1], frames[t2])
        print(f"T{t1} vs T{t2}: Mean difference = {diff:.2f}, Identical = {are_same}")

# Visual comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
test_points = [0, 20, 40] if 40 < analyzer.metadata['frames'] else [0, analyzer.metadata['frames']//2, analyzer.metadata['frames']-1]

for i, t in enumerate(test_points):
    if t in frames:
        vmin, vmax = np.percentile(frames[t], [5, 99.5])
        axes[i].imshow(frames[t], cmap='hot', vmin=vmin, vmax=vmax)
        axes[i].set_title(f'Timepoint {t} (t={t*15} min)')
        axes[i].axis('off')

plt.suptitle('Timepoint Comparison')
plt.tight_layout()
plt.show()

# Also test with nd2reader directly
print("\nTesting with nd2reader directly:")
from nd2reader import ND2Reader
reader = ND2Reader(nd2_file)

for t in [0, 20, 40]:
    if t < reader.metadata['total_images_per_channel']:
        reader.iter_axes = 't'
        reader.default_coords['c'] = 1  # TRITC
        frame = reader[t]
        print(f"Direct access T{t}: shape = {frame.shape if frame is not None else 'None'}")

reader.close()
analyzer.close()