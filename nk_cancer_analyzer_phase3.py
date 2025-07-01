"""
NK Cell Cancer Analysis Tool - Phase 3: Droplet Detection and Cell Analysis
Author: AI Assistant
Date: 2025-01-20

This module adds droplet detection to analyze cells only within droplets.
Droplets are 160-180 micrometers in diameter with black rings.
"""

import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib import patches
import pandas as pd
from nk_cancer_analyzer_phase2 import CellDetector

class DropletDetector:
    """Detect and analyze droplets in brightfield images."""
    
    def __init__(self, pixel_size_um=0.65, expected_diameter_um=170):
        """
        Initialize droplet detector.
        
        Args:
            pixel_size_um: Pixel size in micrometers (10x objective typically ~0.65)
            expected_diameter_um: Expected droplet diameter in micrometers (160-180)
        """
        self.pixel_size_um = pixel_size_um
        self.expected_diameter_um = expected_diameter_um
        self.expected_diameter_px = expected_diameter_um / pixel_size_um
        self.expected_radius_px = self.expected_diameter_px / 2
        
        print(f"Droplet detector initialized:")
        print(f"  Expected diameter: {expected_diameter_um} µm ({self.expected_diameter_px:.0f} pixels)")
        print(f"  Expected radius: {self.expected_radius_px:.0f} pixels")
        
    def detect_droplets(self, brightfield_image, method='hough'):
        """
        Detect circular droplets in brightfield image.
        
        Args:
            brightfield_image: Brightfield channel image
            method: 'hough' for Hough circles or 'threshold' for edge detection
            
        Returns:
            droplets: List of dictionaries with droplet properties
        """
        # Convert to 8-bit for OpenCV
        img_norm = cv2.normalize(brightfield_image, None, 0, 255, cv2.NORM_MINMAX)
        img_8bit = img_norm.astype(np.uint8)
        
        if method == 'hough':
            droplets = self._detect_droplets_hough(img_8bit)
        else:
            droplets = self._detect_droplets_threshold(brightfield_image)
            
        return droplets
    
    def _detect_droplets_hough(self, img_8bit):
        """Detect droplets using Hough Circle Transform."""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(img_8bit, (9, 9), 2)
        
        # Edge detection for better circle finding
        edges = cv2.Canny(blurred, 30, 100)
        
        # Parameters for Hough circles
        min_radius = int(self.expected_radius_px * 0.7)  # 70% of expected
        max_radius = int(self.expected_radius_px * 1.3)  # 130% of expected
        
        all_droplets = []
        
        # First pass: detect strong circles (droplets in wells)
        circles_strong = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(self.expected_diameter_px * 0.8),
            param1=50,
            param2=30,  # Higher threshold for strong circles
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles_strong is not None:
            circles_strong = np.round(circles_strong[0, :]).astype("int")
            for x, y, r in circles_strong:
                all_droplets.append((x, y, r, 'strong'))
        
        # Second pass: detect weaker circles (floating droplets)
        circles_weak = cv2.HoughCircles(
            edges,  # Use edge image for floating droplets
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(self.expected_diameter_px * 0.8),
            param1=30,  # Lower threshold
            param2=20,  # Much lower threshold for weak circles
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles_weak is not None:
            circles_weak = np.round(circles_weak[0, :]).astype("int")
            
            # Add weak circles that don't overlap with strong ones
            for x, y, r in circles_weak:
                is_duplicate = False
                for sx, sy, sr, _ in all_droplets:
                    dist = np.sqrt((x - sx)**2 + (y - sy)**2)
                    if dist < (r + sr) * 0.5:  # Overlapping
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    all_droplets.append((x, y, r, 'weak'))
        
        # Convert to droplet dictionaries
        droplets = []
        for i, (x, y, r, detection_type) in enumerate(all_droplets):
            # Determine droplet type based on ring intensity
            droplet_type = self._classify_droplet_type(img_8bit, x, y, r)
            
            droplets.append({
                'id': i + 1,
                'center_x': x,
                'center_y': y,
                'radius_px': r,
                'radius_um': r * self.pixel_size_um,
                'diameter_um': 2 * r * self.pixel_size_um,
                'type': droplet_type,
                'detection': detection_type
            })
        
        return droplets
    
    def _classify_droplet_type(self, image, cx, cy, r):
        """Classify droplet as 'well' or 'floating' based on ring characteristics."""
        # Sample ring intensities at different radii
        angles = np.linspace(0, 2*np.pi, 36)
        
        # Inner ring (just inside the edge)
        inner_r = r - int(r * 0.1)
        inner_intensities = []
        
        # Outer ring (at the edge)
        outer_intensities = []
        
        for angle in angles:
            # Inner point
            ix = int(cx + inner_r * np.cos(angle))
            iy = int(cy + inner_r * np.sin(angle))
            if 0 <= ix < image.shape[1] and 0 <= iy < image.shape[0]:
                inner_intensities.append(image[iy, ix])
            
            # Outer point
            ox = int(cx + r * np.cos(angle))
            oy = int(cy + r * np.sin(angle))
            if 0 <= ox < image.shape[1] and 0 <= oy < image.shape[0]:
                outer_intensities.append(image[oy, ox])
        
        if not inner_intensities or not outer_intensities:
            return 'unknown'
        
        # Calculate ring darkness
        inner_mean = np.mean(inner_intensities)
        outer_mean = np.mean(outer_intensities)
        ring_darkness = outer_mean
        
        # Well droplets have very dark, thick rings
        # Floating droplets have lighter, thinner rings
        if ring_darkness < 80:  # Very dark ring
            return 'well'
        else:
            return 'floating'
    
    def _detect_droplets_threshold(self, brightfield_image):
        """Alternative detection using edge detection and contours."""
        # Normalize image
        img_norm = (brightfield_image - brightfield_image.min()) / (brightfield_image.max() - brightfield_image.min())
        
        # Edge detection to find dark rings
        edges = cv2.Canny((img_norm * 255).astype(np.uint8), 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        droplets = []
        for i, contour in enumerate(contours):
            # Fit circle to contour
            area = cv2.contourArea(contour)
            if area < np.pi * (self.expected_radius_px * 0.5)**2:  # Too small
                continue
                
            # Get enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            # Check if it's roughly the right size
            if self.expected_radius_px * 0.7 < radius < self.expected_radius_px * 1.3:
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                if circularity > 0.7:  # Reasonably circular
                    droplets.append({
                        'id': i + 1,
                        'center_x': int(x),
                        'center_y': int(y),
                        'radius_px': int(radius),
                        'radius_um': radius * self.pixel_size_um,
                        'diameter_um': 2 * radius * self.pixel_size_um,
                        'type': 'detected',
                        'circularity': circularity
                    })
        
        return droplets
    
    def _verify_droplet_ring(self, image, cx, cy, r, thickness_ratio=0.1):
        """Verify droplet by checking for dark ring at edge."""
        # Create masks for ring region
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        
        # Outer and inner radius for ring
        outer_r = r
        inner_r = r - int(r * thickness_ratio)
        
        # Ring mask
        ring_mask = ((x - cx)**2 + (y - cy)**2 <= outer_r**2) & \
                   ((x - cx)**2 + (y - cy)**2 >= inner_r**2)
        
        # Inside mask
        inside_mask = (x - cx)**2 + (y - cy)**2 < inner_r**2
        
        if np.sum(ring_mask) == 0 or np.sum(inside_mask) == 0:
            return False
        
        # Compare intensities
        ring_intensity = np.mean(image[ring_mask])
        inside_intensity = np.mean(image[inside_mask])
        
        # Ring should be darker than inside
        return ring_intensity < inside_intensity * 0.8
    
    def _is_well_droplet(self, image, cx, cy, r):
        """Determine if droplet is in a well (thick black ring) or floating."""
        # Check ring thickness
        outer_r = r
        inner_r = r - int(r * 0.15)  # Thicker ring for wells
        
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        ring_mask = ((x - cx)**2 + (y - cy)**2 <= outer_r**2) & \
                   ((x - cx)**2 + (y - cy)**2 >= inner_r**2)
        
        if np.sum(ring_mask) == 0:
            return False
            
        ring_intensity = np.mean(image[ring_mask])
        
        # Well droplets have darker, thicker rings
        return ring_intensity < 100  # Adjust threshold as needed
    
    def create_droplet_masks(self, image_shape, droplets):
        """Create binary masks for each droplet."""
        masks = {}
        combined_mask = np.zeros(image_shape, dtype=bool)
        
        for droplet in droplets:
            # Create circular mask
            y, x = np.ogrid[:image_shape[0], :image_shape[1]]
            mask = (x - droplet['center_x'])**2 + (y - droplet['center_y'])**2 <= droplet['radius_px']**2
            
            masks[droplet['id']] = mask
            combined_mask |= mask
        
        return masks, combined_mask
    
    def visualize_droplets(self, brightfield_image, droplets, title="Detected Droplets"):
        """Visualize detected droplets on brightfield image."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original image
        ax1.imshow(brightfield_image, cmap='gray')
        ax1.set_title('Original Brightfield')
        ax1.axis('off')
        
        # Image with droplet annotations
        ax2.imshow(brightfield_image, cmap='gray')
        
        for droplet in droplets:
            # Draw circle
            color = 'lime' if droplet['type'] == 'well' else 'yellow'
            if droplet['type'] == 'floating':
                linestyle = '--'  # Dashed line for floating droplets
            else:
                linestyle = '-'
            
            circle = plt.Circle((droplet['center_x'], droplet['center_y']), 
                              droplet['radius_px'], 
                              color=color,
                              fill=False, linewidth=2, linestyle=linestyle)
            ax2.add_patch(circle)
            
            # Add label
            label_text = f"{droplet['id']}\n{droplet['diameter_um']:.0f}µm"
            if 'detection' in droplet:
                label_text += f"\n{droplet['type']}"
            
            ax2.text(droplet['center_x'], droplet['center_y'], 
                    label_text, 
                    color='white', fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        ax2.set_title(f'Detected Droplets (n={len(droplets)})')
        ax2.axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


class DropletCellAnalyzer:
    """Analyze cells within individual droplets."""
    
    def __init__(self, droplet_detector, cell_detector):
        """Initialize with existing detectors."""
        self.droplet_detector = droplet_detector
        self.cell_detector = cell_detector
        
    def analyze_cells_in_droplets(self, brightfield_image, tritc_image, enhance_cells=True):
        """
        Detect droplets and analyze cells within each droplet.
        
        Args:
            enhance_cells: Apply enhancement to detect faint cells
            
        Returns:
            droplet_results: Dictionary with results for each droplet
        """
        # Detect droplets
        droplets = self.droplet_detector.detect_droplets(brightfield_image)
        
        if not droplets:
            print("No droplets detected!")
            return {}
        
        print(f"Detected {len(droplets)} droplets")
        
        # Create masks for each droplet
        masks, combined_mask = self.droplet_detector.create_droplet_masks(
            brightfield_image.shape, droplets
        )
        
        droplet_results = {}
        
        for droplet in droplets:
            droplet_id = droplet['id']
            mask = masks[droplet_id]
            
            # Extract TRITC signal within droplet
            masked_tritc = tritc_image.copy()
            masked_tritc[~mask] = 0
            
            # Extract brightfield within droplet for combined detection
            masked_bf = brightfield_image.copy()
            masked_bf[~mask] = 0
            
            # Detect nuclei using specialized nuclear detection
            nuclei_positions = self._detect_nuclei(masked_tritc, masked_bf, mask, droplet)
            
            # Convert nuclei to cell properties format
            valid_cells = []
            for i, (cx, cy, intensity, area) in enumerate(nuclei_positions):
                dist_from_center = np.sqrt(
                    (cx - droplet['center_x'])**2 + 
                    (cy - droplet['center_y'])**2
                )
                
                # Keep cells well within droplet
                if dist_from_center < droplet['radius_px'] * 0.9:
                    valid_cells.append({
                        'label': i + 1,
                        'centroid_x': cx,
                        'centroid_y': cy,
                        'mean_intensity': intensity,
                        'area': area,
                        'distance_from_center': dist_from_center
                    })
            
            droplet_results[droplet_id] = {
                'droplet_info': droplet,
                'cell_count': len(valid_cells),
                'cells': valid_cells,
                'mean_intensity': np.mean([c['mean_intensity'] for c in valid_cells]) 
                                if valid_cells else 0
            }
            
            print(f"Droplet {droplet_id} ({droplet['type']}): {len(valid_cells)} cells detected")
        
        return droplet_results
    
    def _detect_nuclei(self, tritc_image, brightfield_image, mask, droplet):
        """
        Specialized detection for mKate2 nuclear staining.
        
        Returns:
            List of (x, y, intensity, area) tuples for detected nuclei
        """
        from scipy.ndimage import label, center_of_mass, distance_transform_edt
        from skimage.feature import peak_local_max
        from skimage.filters import gaussian, median
        from skimage.segmentation import watershed
        import numpy as np

        # Get the region of interest
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            return []
        
        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()
        
        # Crop to ROI
        roi_tritc = tritc_image[y_min:y_max+1, x_min:x_max+1]
        roi_mask = mask[y_min:y_max+1, x_min:x_max+1]

        # Step 1: Enhance nuclear signal
        denoised = median(roi_tritc, footprint=np.ones((3, 3)))
        background = gaussian(denoised, sigma=10)
        signal = np.maximum(0, denoised - background * 0.8)

        # Normalize within droplet
        if signal[roi_mask].max() > 0:
            signal = signal / signal[roi_mask].max()

        # Step 2: Thresholding to find large aggregates
        min_area = 85  # Only keep aggregates of at least 200 pixels (adjust as needed)
        threshold = np.percentile(signal[roi_mask], 99)  # Use a high threshold for strong staining
        binary = (signal > threshold) & roi_mask

        # Remove small objects
        from skimage.morphology import remove_small_objects
        binary = remove_small_objects(binary, min_size=min_area)

        # --- Watershed to split aggregates ---
        # Compute distance transform
        distance = distance_transform_edt(binary)
        # Find local maxima (cell centers)
        coordinates = peak_local_max(distance, min_distance=10, labels=binary)
        # Create marker image
        markers = np.zeros_like(distance, dtype=int)
        for idx, (y, x) in enumerate(coordinates, 1):
            markers[y, x] = idx
        # Watershed segmentation
        labels_ws = watershed(-distance, markers, mask=binary)

        # Label and filter regions
        nuclei = []
        for i in range(1, labels_ws.max() + 1):
            nucleus_mask = labels_ws == i
            area = np.sum(nucleus_mask)
            if area >= min_area:
                cy, cx = center_of_mass(nucleus_mask)
                intensity = np.mean(signal[nucleus_mask])
                nuclei.append((cx + x_min, cy + y_min, intensity, area))

        return nuclei
    
    def visualize_droplet_cells(self, brightfield_image, tritc_image, droplet_results):
        """Visualize cells within each droplet."""
        n_droplets = len(droplet_results)
        if n_droplets == 0:
            print("No droplets to visualize!")
            return
        
        # Create grid layout
        cols = min(3, n_droplets)
        rows = (n_droplets + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if n_droplets == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if rows > 1 else axes
        
        for idx, (droplet_id, results) in enumerate(droplet_results.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx] if n_droplets > 1 else axes
            droplet = results['droplet_info']
            
            # Create crop window
            margin = int(droplet['radius_px'] * 1.2)
            x0 = max(0, droplet['center_x'] - margin)
            x1 = min(brightfield_image.shape[1], droplet['center_x'] + margin)
            y0 = max(0, droplet['center_y'] - margin)
            y1 = min(brightfield_image.shape[0], droplet['center_y'] + margin)
            
            # Enhanced TRITC visualization for nuclear staining
            tritc_crop = tritc_image[y0:y1, x0:x1].copy()
            
            # Enhance contrast for visualization
            if tritc_crop.max() > tritc_crop.min():
                # Use adaptive enhancement
                p_low = np.percentile(tritc_crop[tritc_crop > 0], 10) if np.any(tritc_crop > 0) else 0
                p_high = np.percentile(tritc_crop, 99.8)
                tritc_enhanced = np.clip((tritc_crop - p_low) / (p_high - p_low + 1e-8), 0, 1)
            else:
                tritc_enhanced = tritc_crop
            
            # Create colored overlay
            # Red channel: TRITC (nuclei)
            # Green channel: slight brightfield for context
            # Blue channel: brightfield
            bf_crop = brightfield_image[y0:y1, x0:x1]
            bf_norm = (bf_crop - bf_crop.min()) / (bf_crop.max() - bf_crop.min() + 1e-8)
            
            rgb_image = np.zeros((*tritc_enhanced.shape, 3))
            rgb_image[:, :, 0] = tritc_enhanced * 2  # Amplify red for visibility
            rgb_image[:, :, 1] = bf_norm * 0.3  # Dim green
            rgb_image[:, :, 2] = bf_norm * 0.5  # Medium blue
            
            # Clip to valid range
            rgb_image = np.clip(rgb_image, 0, 1)
            
            ax.imshow(rgb_image)
            
            # Draw droplet circle
            circle = plt.Circle((droplet['center_x'] - x0, droplet['center_y'] - y0), 
                              droplet['radius_px'], 
                              color='lime', fill=False, linewidth=2)
            ax.add_patch(circle)
            
            # Mark detected nuclei with better visibility
            for cell in results['cells']:
                cx = cell['centroid_x'] - x0
                cy = cell['centroid_y'] - y0
                
                # White cross with black outline for visibility
                ax.plot(cx, cy, 'k+', markersize=12, markeredgewidth=3)  # Black outline
                ax.plot(cx, cy, 'w+', markersize=10, markeredgewidth=2)  # White cross
                
                # Small circle around nucleus
                nucleus_circle = plt.Circle((cx, cy), 5, 
                                          color='yellow', fill=False, 
                                          linewidth=1, alpha=0.8)
                ax.add_patch(nucleus_circle)
            
            # Add title with count
            title = f'Droplet {droplet_id} ({droplet["type"]})\n'
            title += f'{results["cell_count"]} cells, {droplet["diameter_um"]:.0f}µm'
            ax.set_title(title, fontsize=10)
            ax.axis('off')
            
            # Add scale bar (50 µm)
            scale_length_px = 50 / self.droplet_detector.pixel_size_um
            scale_y = tritc_enhanced.shape[0] - 20
            ax.plot([10, 10 + scale_length_px], [scale_y, scale_y], 
                   'white', linewidth=3)
            ax.text(10 + scale_length_px/2, scale_y - 10, '50 µm', 
                   color='white', ha='center', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        # Hide unused subplots
        for idx in range(len(droplet_results), len(axes)):
            axes[idx].axis('off') if n_droplets > 1 else None
        
        plt.suptitle('Cancer Cells (mKate2+ Nuclei) in Individual Droplets', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def summarize_results(self, droplet_results):
        """Create summary statistics of droplet analysis."""
        summary = []
        
        for droplet_id, results in droplet_results.items():
            droplet = results['droplet_info']
            summary.append({
                'droplet_id': droplet_id,
                'type': droplet['type'],
                'diameter_um': droplet['diameter_um'],
                'cell_count': results['cell_count'],
                'mean_intensity': results['mean_intensity']
            })
        
        df = pd.DataFrame(summary)
        
        if len(df) > 0:
            print("\nDroplet Analysis Summary:")
            print("=" * 50)
            print(f"Total droplets: {len(df)}")
            print(f"Droplets with cells: {len(df[df['cell_count'] > 0])}")
            print(f"Total cells: {df['cell_count'].sum()}")
            print(f"\nCells per droplet: {df['cell_count'].mean():.1f} ± {df['cell_count'].std():.1f}")
            print(f"Droplet diameter: {df['diameter_um'].mean():.1f} ± {df['diameter_um'].std():.1f} µm")
            
            # Group by type if available
            if 'well' in df['type'].values:
                print("\nBy droplet type:")
                print(df.groupby('type')[['cell_count', 'diameter_um']].mean())
        
        return df


# Test function for Phase 3
def test_phase3():
    """Test Phase 3 droplet detection and cell analysis."""
    print("Phase 3 Droplet Detection Test")
    print("=" * 50)
    
    from nk_cancer_analyzer import ND2Analyzer
    
    # Load your ND2 file
    nd2_file = r"D:\New\BrainBites\Cell\2.nd2"
    analyzer = ND2Analyzer(nd2_file)
    
    if not analyzer.load_file():
        print("Failed to load file!")
        return
    
    # Get frames
    print("\nLoading frames...")
    bf_frame = analyzer.get_frame(0, 'brightfield')
    tritc_frame = analyzer.get_frame(0, 'TRITC')
    
    # Initialize detectors
    print("\nInitializing detectors...")
    droplet_detector = DropletDetector(pixel_size_um=0.65, expected_diameter_um=170)
    cell_detector = CellDetector(min_cell_area=30, max_cell_area=500)  # Adjusted for cell size
    
    # Detect droplets
    print("\nDetecting droplets...")
    droplets = droplet_detector.detect_droplets(bf_frame, method='hough')
    print(f"Found {len(droplets)} droplets")
    
    # Visualize droplets
    droplet_detector.visualize_droplets(bf_frame, droplets)
    
    # Analyze cells in droplets
    print("\nAnalyzing cells in droplets...")
    analyzer_tool = DropletCellAnalyzer(droplet_detector, cell_detector)
    droplet_results = analyzer_tool.analyze_cells_in_droplets(bf_frame, tritc_frame)
    
    # Visualize results
    analyzer_tool.visualize_droplet_cells(bf_frame, tritc_frame, droplet_results)
    
    # Summary
    summary_df = analyzer_tool.summarize_results(droplet_results)
    
    # Clean up
    analyzer.close()
    
    print("\nPhase 3 testing complete!")

if __name__ == "__main__":
    test_phase3()