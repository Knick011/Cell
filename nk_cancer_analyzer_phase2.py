"""
NK Cell Cancer Analysis Tool - Phase 2: Cell Detection and Counting
Author: AI Assistant
Date: 2025-01-20

This module adds advanced cell detection, segmentation, and counting capabilities.
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import morphology
from skimage import measure, segmentation, feature
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from skimage.filters import threshold_otsu, gaussian
import matplotlib.pyplot as plt
from matplotlib import patches
import pandas as pd

class CellDetector:
    """Advanced cell detection and analysis for TRITC channel."""
    
    def __init__(self, min_cell_area=50, max_cell_area=5000):
        """
        Initialize detector with parameters.
        
        Args:
            min_cell_area: Minimum area (pixels) for valid cell
            max_cell_area: Maximum area (pixels) for valid cell
        """
        self.min_cell_area = min_cell_area
        self.max_cell_area = max_cell_area
        
    def preprocess_image(self, image, sigma=1.0):
        """Apply preprocessing to enhance cell detection."""
        # Convert to float
        img_float = image.astype(np.float32)
        
        # Apply Gaussian blur to reduce noise
        img_smooth = gaussian(img_float, sigma=sigma)
        
        # Normalize to 0-1 range
        img_norm = (img_smooth - img_smooth.min()) / (img_smooth.max() - img_smooth.min())
        
        return img_norm
    
    def detect_cells_adaptive(self, image, method='watershed'):
        """
        Detect cells using adaptive thresholding and watershed segmentation.
        
        Args:
            image: TRITC channel image
            method: 'watershed' or 'simple' detection method
            
        Returns:
            labeled_cells: Labeled image where each cell has unique ID
            cell_properties: List of cell properties (area, centroid, etc.)
        """
        # Preprocess
        img_processed = self.preprocess_image(image)
        
        # Calculate adaptive threshold
        # Use percentile-based threshold for fluorescence
        threshold_value = np.percentile(img_processed, 90)
        binary = img_processed > threshold_value
        
        # Morphological operations to clean up
        binary = morphology.binary_opening(binary, iterations=1)
        binary = morphology.binary_closing(binary, iterations=1)
        
        # Remove small objects
        binary = remove_small_objects(binary, min_size=self.min_cell_area)
        
        if method == 'watershed':
            # Use watershed for touching cells
            labeled_cells = self._watershed_segmentation(img_processed, binary)
        else:
            # Simple connected component labeling
            labeled_cells, _ = ndimage.label(binary)
        
        # Get cell properties
        cell_properties = self._measure_cells(labeled_cells, image)
        
        return labeled_cells, cell_properties
    
    def _watershed_segmentation(self, image, binary_mask):
        """Apply watershed segmentation to separate touching cells."""
        # Calculate distance transform
        distance = ndimage.distance_transform_edt(binary_mask)
        
        # Find local maxima (cell centers)
        coordinates = feature.peak_local_max(
            distance, 
            min_distance=10,
            labels=binary_mask
        )
        
        # Convert coordinates to boolean mask
        local_maxima = np.zeros_like(distance, dtype=bool)
        local_maxima[tuple(coordinates.T)] = True
        
        # Create markers for watershed
        markers = ndimage.label(local_maxima)[0]
        
        # Apply watershed
        labels = watershed(-distance, markers, mask=binary_mask)
        
        return labels
    
    def _measure_cells(self, labeled_image, intensity_image):
        """Measure properties of detected cells."""
        # Get region properties
        props = measure.regionprops(labeled_image, intensity_image=intensity_image)
        
        cell_data = []
        for prop in props:
            # Filter by size
            if self.min_cell_area <= prop.area <= self.max_cell_area:
                cell_data.append({
                    'label': prop.label,
                    'area': prop.area,
                    'centroid_x': prop.centroid[1],
                    'centroid_y': prop.centroid[0],
                    'mean_intensity': prop.mean_intensity,
                    'max_intensity': prop.max_intensity,
                    'eccentricity': prop.eccentricity,
                    'perimeter': prop.perimeter,
                    'solidity': prop.solidity,
                    'bbox': prop.bbox
                })
        
        return cell_data
    
    def visualize_detection(self, image, labeled_cells, cell_properties, title="Cell Detection"):
        """Visualize detected cells with bounding boxes and labels."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        vmin, vmax = np.percentile(image, [5, 99.5])
        axes[0].imshow(image, cmap='hot', vmin=vmin, vmax=vmax)
        axes[0].set_title('Original TRITC')
        axes[0].axis('off')
        
        # Labeled cells
        axes[1].imshow(labeled_cells, cmap='tab20')
        axes[1].set_title(f'Detected Cells (n={len(cell_properties)})')
        axes[1].axis('off')
        
        # Overlay with annotations
        axes[2].imshow(image, cmap='hot', vmin=vmin, vmax=vmax)
        
        # Add bounding boxes and cell numbers
        for cell in cell_properties:
            # Bounding box
            minr, minc, maxr, maxc = cell['bbox']
            rect = patches.Rectangle((minc, minr), maxc-minc, maxr-minr,
                                   linewidth=1, edgecolor='lime', facecolor='none')
            axes[2].add_patch(rect)
            
            # Cell number
            axes[2].text(cell['centroid_x'], cell['centroid_y'], 
                        str(cell['label']), 
                        color='white', fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.5))
        
        axes[2].set_title('Annotated Cells')
        axes[2].axis('off')
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def analyze_intensity_distribution(self, cell_properties):
        """Analyze and plot intensity distribution of cells."""
        if not cell_properties:
            print("No cells detected!")
            return
        
        df = pd.DataFrame(cell_properties)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Intensity distribution
        axes[0, 0].hist(df['mean_intensity'], bins=30, alpha=0.7, color='red')
        axes[0, 0].set_xlabel('Mean Intensity')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Cell Intensity Distribution')
        
        # Area distribution
        axes[0, 1].hist(df['area'], bins=30, alpha=0.7, color='blue')
        axes[0, 1].set_xlabel('Area (pixels)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Cell Size Distribution')
        
        # Intensity vs Area scatter
        axes[1, 0].scatter(df['area'], df['mean_intensity'], alpha=0.6)
        axes[1, 0].set_xlabel('Area (pixels)')
        axes[1, 0].set_ylabel('Mean Intensity')
        axes[1, 0].set_title('Cell Size vs Intensity')
        
        # Eccentricity distribution (cell shape)
        axes[1, 1].hist(df['eccentricity'], bins=20, alpha=0.7, color='green')
        axes[1, 1].set_xlabel('Eccentricity (0=circle, 1=line)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Cell Shape Distribution')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\nCell Detection Summary:")
        print(f"Total cells detected: {len(df)}")
        print(f"Mean cell area: {df['area'].mean():.1f} ± {df['area'].std():.1f} pixels")
        print(f"Mean intensity: {df['mean_intensity'].mean():.1f} ± {df['mean_intensity'].std():.1f}")
        
        return df


class CellTracker:
    """Track cells across time points to measure death timing."""
    
    def __init__(self, max_distance=50):
        """
        Initialize tracker.
        
        Args:
            max_distance: Maximum distance (pixels) to link cells between frames
        """
        self.max_distance = max_distance
        self.tracks = {}
        self.next_track_id = 1
        
    def track_cells(self, previous_cells, current_cells):
        """
        Simple nearest-neighbor tracking between frames.
        
        Returns:
            track_assignments: Dict mapping current cell labels to track IDs
        """
        if not previous_cells:
            # First frame - assign new track IDs
            track_assignments = {}
            for cell in current_cells:
                track_assignments[cell['label']] = self.next_track_id
                self.tracks[self.next_track_id] = {
                    'start_frame': 0,
                    'positions': [(cell['centroid_x'], cell['centroid_y'])],
                    'intensities': [cell['mean_intensity']],
                    'status': 'alive'
                }
                self.next_track_id += 1
            return track_assignments
        
        # Create cost matrix for assignment
        track_assignments = {}
        used_previous = set()
        
        # For each current cell, find nearest previous cell
        for curr_cell in current_cells:
            best_dist = float('inf')
            best_prev = None
            
            for prev_cell in previous_cells:
                if prev_cell['label'] in used_previous:
                    continue
                    
                # Calculate distance
                dist = np.sqrt(
                    (curr_cell['centroid_x'] - prev_cell['centroid_x'])**2 +
                    (curr_cell['centroid_y'] - prev_cell['centroid_y'])**2
                )
                
                if dist < best_dist and dist < self.max_distance:
                    best_dist = dist
                    best_prev = prev_cell
            
            if best_prev:
                # Found a match
                track_id = best_prev.get('track_id', self.next_track_id)
                track_assignments[curr_cell['label']] = track_id
                used_previous.add(best_prev['label'])
                
                # Update track
                if track_id in self.tracks:
                    self.tracks[track_id]['positions'].append(
                        (curr_cell['centroid_x'], curr_cell['centroid_y'])
                    )
                    self.tracks[track_id]['intensities'].append(curr_cell['mean_intensity'])
            else:
                # New cell appeared
                track_assignments[curr_cell['label']] = self.next_track_id
                self.tracks[self.next_track_id] = {
                    'start_frame': len(self.tracks[self.next_track_id]['positions']) 
                                  if self.next_track_id in self.tracks else 0,
                    'positions': [(curr_cell['centroid_x'], curr_cell['centroid_y'])],
                    'intensities': [curr_cell['mean_intensity']],
                    'status': 'alive'
                }
                self.next_track_id += 1
        
        return track_assignments
    
    def identify_dead_cells(self, frame_num, intensity_threshold=0.3):
        """
        Identify cells that have died based on intensity drop.
        
        Args:
            frame_num: Current frame number
            intensity_threshold: Relative intensity threshold for death
        """
        dead_cells = []
        
        for track_id, track_data in self.tracks.items():
            if track_data['status'] == 'alive' and len(track_data['intensities']) > 1:
                # Check for significant intensity drop
                current_intensity = track_data['intensities'][-1]
                max_intensity = max(track_data['intensities'])
                
                if current_intensity < intensity_threshold * max_intensity:
                    track_data['status'] = 'dead'
                    track_data['death_frame'] = frame_num
                    dead_cells.append(track_id)
        
        return dead_cells


# Integration with Phase 1 analyzer
def analyze_cell_dynamics(analyzer, detector, tracker, start_frame=0, end_frame=None):
    """
    Analyze cell dynamics over time using the ND2Analyzer and CellDetector.
    
    Args:
        analyzer: ND2Analyzer instance with loaded file
        detector: CellDetector instance
        tracker: CellTracker instance
        start_frame: Starting frame for analysis
        end_frame: Ending frame (None for all frames)
    
    Returns:
        results: DataFrame with time series data
    """
    if end_frame is None:
        end_frame = analyzer.metadata['frames']
    
    results = []
    previous_cells = None
    
    print(f"Analyzing frames {start_frame} to {end_frame-1}...")
    
    for frame_idx in range(start_frame, end_frame):
        # Get TRITC frame
        tritc_frame = analyzer.get_frame(frame_idx, 'TRITC')
        
        if tritc_frame is None:
            continue
        
        # Detect cells
        labeled_cells, cell_properties = detector.detect_cells_adaptive(tritc_frame)
        
        # Track cells
        if previous_cells is not None:
            track_assignments = tracker.track_cells(previous_cells, cell_properties)
            
            # Add track IDs to cell properties
            for cell in cell_properties:
                cell['track_id'] = track_assignments.get(cell['label'], -1)
        
        # Record results
        results.append({
            'frame': frame_idx,
            'time_min': frame_idx * analyzer.metadata['time_interval'],
            'cell_count': len(cell_properties),
            'mean_intensity': np.mean([c['mean_intensity'] for c in cell_properties]) 
                           if cell_properties else 0,
            'total_area': sum([c['area'] for c in cell_properties])
        })
        
        # Visualize every 10th frame
        if frame_idx % 10 == 0:
            print(f"Frame {frame_idx}: {len(cell_properties)} cells detected")
            detector.visualize_detection(tritc_frame, labeled_cells, cell_properties,
                                       title=f"Frame {frame_idx} (t={frame_idx*15} min)")
        
        previous_cells = cell_properties
    
    return pd.DataFrame(results)


# Test function for Phase 2
def test_phase2():
    """Test the Phase 2 cell detection capabilities."""
    print("Phase 2 Cell Detection Test")
    print("=" * 50)
    
    # You'll need to import the Phase 1 analyzer
    from nk_cancer_analyzer import ND2Analyzer
    
    # Load your ND2 file
    nd2_file = r"D:\New\BrainBites\Cell\2.nd2"  # Update with your path
    analyzer = ND2Analyzer(nd2_file)
    
    if not analyzer.load_file():
        print("Failed to load file!")
        return
    
    # Create detector and tracker
    detector = CellDetector(min_cell_area=50, max_cell_area=2000)
    tracker = CellTracker(max_distance=50)
    
    # Test on single frame
    print("\nTesting single frame detection...")
    tritc_frame = analyzer.get_frame(0, 'TRITC')
    labeled_cells, cell_properties = detector.detect_cells_adaptive(tritc_frame)
    
    # Visualize
    detector.visualize_detection(tritc_frame, labeled_cells, cell_properties)
    
    # Analyze properties
    df_cells = detector.analyze_intensity_distribution(cell_properties)
    
    # Analyze first 10 frames
    print("\nAnalyzing time series (first 10 frames)...")
    results_df = analyze_cell_dynamics(analyzer, detector, tracker, 
                                     start_frame=0, end_frame=10)
    
    # Plot time series
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(results_df['time_min'], results_df['cell_count'], 'b-o')
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Cell Count')
    ax1.set_title('Cancer Cell Count Over Time')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(results_df['time_min'], results_df['mean_intensity'], 'r-o')
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Mean Intensity')
    ax2.set_title('Average Cell Intensity Over Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Clean up
    analyzer.close()
    
    print("\nPhase 2 testing complete!")
    print(f"Initial cell count: {results_df.iloc[0]['cell_count']}")
    print(f"Final cell count: {results_df.iloc[-1]['cell_count']}")

if __name__ == "__main__":
    test_phase2()