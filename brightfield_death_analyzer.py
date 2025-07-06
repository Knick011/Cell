"""
Complete Cancer Cell Analysis with Brightfield Death Detection
This integrates brightfield morphology-based death detection into your existing pipeline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches
from matplotlib.gridspec import GridSpec
import cv2
import os
from datetime import datetime
from scipy import ndimage
from scipy.ndimage import label, binary_erosion, binary_dilation
from skimage import morphology, measure, filters
from skimage.feature import graycomatrix, graycoprops
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment

# Import your existing modules
from nk_cancer_analyzer import ND2Analyzer
from nk_cancer_analyzer_phase3 import DropletDetector, DropletCellAnalyzer

class BrightfieldDeathDetector:
    """Detect cell death using brightfield morphology at TRITC-marked locations."""
    
    def __init__(self):
        self.cell_reference_states = {}  # Store healthy cell appearance
        self.death_threshold_scores = {
            'disappeared': 0.9,      # Cell completely gone
            'blebbing': 0.8,         # Membrane bubbles
            'shrunk': 0.7,           # Significant size reduction
            'shape_change': 0.6,     # Irregular shape
            'texture_change': 0.5    # Granular appearance
        }
    
    def analyze_cell_morphology(self, bf_image, tritc_image, cell_position, cell_id, 
                               previous_bf=None, store_reference=False):
        """
        Analyze morphology at a TRITC-identified cell position.
        
        Args:
            bf_image: Brightfield image
            tritc_image: TRITC fluorescence image
            cell_position: (x, y) center from TRITC detection
            cell_id: Unique cell identifier
            previous_bf: Previous brightfield frame
            store_reference: Store this as healthy reference
            
        Returns:
            dict: Morphology analysis results and death score
        """
        x, y = int(cell_position[0]), int(cell_position[1])
        
        # Define analysis region (larger than TRITC to catch blebs)
        roi_size = 40  # pixels, adjust based on cell size
        x0 = max(0, x - roi_size)
        x1 = min(bf_image.shape[1], x + roi_size)
        y0 = max(0, y - roi_size)
        y1 = min(bf_image.shape[0], y + roi_size)
        
        # Extract regions
        bf_roi = bf_image[y0:y1, x0:x1].copy()
        tritc_roi = tritc_image[y0:y1, x0:x1].copy()
        
        # Get TRITC mask for cell location
        if np.any(tritc_roi > 0):
            tritc_thresh = filters.threshold_otsu(tritc_roi)
            tritc_mask = tritc_roi > tritc_thresh
        else:
            # Create small circular mask if TRITC is too weak
            center_x, center_y = roi_size, roi_size
            y_grid, x_grid = np.ogrid[:bf_roi.shape[0], :bf_roi.shape[1]]
            tritc_mask = (x_grid - center_x)**2 + (y_grid - center_y)**2 <= 100
        
        # Analyze brightfield morphology
        features = {}
        
        # 1. Check if cell disappeared
        features['disappeared_score'] = self._check_disappearance(bf_roi, tritc_mask)
        
        # 2. Detect blebbing
        features['blebbing_score'] = self._detect_blebbing(bf_roi, tritc_mask)
        
        # 3. Measure shrinkage
        features['shrinkage_score'] = self._measure_shrinkage(bf_roi, tritc_mask, cell_id)
        
        # 4. Shape irregularity
        features['shape_score'] = self._analyze_shape(bf_roi, tritc_mask)
        
        # 5. Texture changes
        features['texture_score'] = self._analyze_texture(bf_roi, tritc_mask, cell_id)
        
        # 6. Temporal changes if previous frame available
        if previous_bf is not None:
            prev_roi = previous_bf[y0:y1, x0:x1]
            features['temporal_score'] = self._temporal_changes(bf_roi, prev_roi, tritc_mask)
        else:
            features['temporal_score'] = 0
        
        # Calculate overall death score
        death_score = self._calculate_death_score(features)
        
        # Store reference if requested (for healthy cells)
        if store_reference and death_score < 0.3:
            self.cell_reference_states[cell_id] = {
                'texture': self._get_texture_features(bf_roi, tritc_mask),
                'size': np.sum(tritc_mask),
                'shape': self._get_shape_features(bf_roi, tritc_mask)
            }
        
        return {
            'features': features,
            'death_score': death_score,
            'state': self._classify_state(death_score),
            'roi': bf_roi,
            'mask': tritc_mask
        }
    
    def _check_disappearance(self, bf_roi, tritc_mask):
        """Check if cell has disappeared/lysed."""
        masked_bf = bf_roi[tritc_mask]
        if len(masked_bf) == 0:
            return 1.0
        
        # Contrast measure
        contrast = np.std(masked_bf) / (np.mean(masked_bf) + 1e-6)
        
        # Edge strength
        edges = cv2.Canny(bf_roi.astype(np.uint8), 30, 100)
        edge_density = np.sum(edges[tritc_mask]) / np.sum(tritc_mask)
        
        # Low contrast + low edges = disappeared
        disappearance_score = 1.0 - min(contrast * 10, 1.0) * min(edge_density * 5, 1.0)
        
        return disappearance_score
    
    def _detect_blebbing(self, bf_roi, tritc_mask):
        """Detect membrane blebbing (bubble-like protrusions)."""
        # Find cell boundary
        dilated = morphology.dilation(tritc_mask, morphology.disk(3))
        boundary = dilated & ~tritc_mask
        
        if np.sum(boundary) == 0:
            return 0.0
        
        # Detect circular structures using Hough transform
        edges = cv2.Canny(bf_roi.astype(np.uint8), 50, 150)
        
        # Look for small circles (blebs)
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=5,
            param1=50,
            param2=20,
            minRadius=2,
            maxRadius=8
        )
        
        bleb_score = 0.0
        if circles is not None:
            circles = circles[0]
            # Count circles near boundary
            for cx, cy, r in circles:
                if 0 <= int(cy) < boundary.shape[0] and 0 <= int(cx) < boundary.shape[1]:
                    if boundary[int(cy), int(cx)]:
                        bleb_score += 0.2
        
        # Also check for intensity variations at boundary
        boundary_bf = bf_roi[boundary]
        if len(boundary_bf) > 0:
            boundary_std = np.std(boundary_bf)
            bleb_score += min(boundary_std / 30, 0.5)
        
        return min(bleb_score, 1.0)
    
    def _measure_shrinkage(self, bf_roi, tritc_mask, cell_id):
        """Measure cell shrinkage compared to reference."""
        # Find actual cell area in brightfield
        bf_smooth = cv2.GaussianBlur(bf_roi, (5, 5), 1)
        bf_thresh = cv2.adaptiveThreshold(
            bf_smooth.astype(np.uint8),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Find cell body (should overlap with TRITC)
        cell_body = bf_thresh & (tritc_mask.astype(np.uint8) * 255)
        current_size = np.sum(cell_body > 0)
        
        # Compare to reference size
        if cell_id in self.cell_reference_states:
            ref_size = self.cell_reference_states[cell_id]['size']
            shrinkage_ratio = 1.0 - (current_size / (ref_size + 1e-6))
            return max(0, min(shrinkage_ratio, 1.0))
        else:
            # Compare to TRITC area
            tritc_size = np.sum(tritc_mask)
            size_ratio = current_size / (tritc_size + 1e-6)
            # Healthy cells should fill most of TRITC area
            return max(0, 1.0 - size_ratio)
    
    def _analyze_shape(self, bf_roi, tritc_mask):
        """Analyze shape irregularity."""
        # Extract cell contour from brightfield
        bf_smooth = cv2.GaussianBlur(bf_roi, (5, 5), 1)
        edges = cv2.Canny(bf_smooth.astype(np.uint8), 30, 100)
        
        # Focus on edges within/near TRITC region
        dilated_mask = morphology.dilation(tritc_mask, morphology.disk(5))
        relevant_edges = edges & (dilated_mask.astype(np.uint8) * 255)
        
        # Find contours
        contours, _ = cv2.findContours(
            relevant_edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return 0.5  # No clear shape
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate shape metrics
        area = cv2.contourArea(largest_contour)
        if area < 10:
            return 0.5
        
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Circularity (perfect circle = 1)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Convexity
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Irregular shapes have low circularity and solidity
        irregularity = 1.0 - (circularity * 0.5 + solidity * 0.5)
        
        return irregularity
    
    def _analyze_texture(self, bf_roi, tritc_mask, cell_id):
        """Analyze texture changes indicating death."""
        features = self._get_texture_features(bf_roi, tritc_mask)
        
        if cell_id in self.cell_reference_states:
            ref_features = self.cell_reference_states[cell_id]['texture']
            
            # Compare features
            changes = []
            for key in features:
                if key in ref_features and ref_features[key] > 0:
                    change = abs(features[key] - ref_features[key]) / ref_features[key]
                    changes.append(change)
            
            return min(np.mean(changes) * 2, 1.0) if changes else 0.5
        else:
            # Dead cells typically have higher contrast and energy
            return min(features.get('contrast', 0) / 50 + features.get('energy', 0) * 2, 1.0)
    
    def _get_texture_features(self, bf_roi, mask):
        """Extract texture features using GLCM."""
        masked = bf_roi.copy()
        masked[~mask] = 0
        
        # Normalize to 0-255
        if masked.max() > masked.min():
            masked_norm = ((masked - masked.min()) / (masked.max() - masked.min()) * 255).astype(np.uint8)
        else:
            return {'contrast': 0, 'energy': 0, 'homogeneity': 0, 'correlation': 0}
        
        try:
            # Calculate GLCM
            glcm = graycomatrix(
                masked_norm,
                distances=[1],
                angles=[0],
                levels=256,
                symmetric=True,
                normed=True
            )
            
            return {
                'contrast': graycoprops(glcm, 'contrast')[0, 0],
                'energy': graycoprops(glcm, 'energy')[0, 0],
                'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
                'correlation': graycoprops(glcm, 'correlation')[0, 0]
            }
        except:
            return {'contrast': 0, 'energy': 0, 'homogeneity': 0, 'correlation': 0}
    
    def _get_shape_features(self, bf_roi, mask):
        """Extract shape features."""
        # Simplified implementation
        return {'circularity': 0.5, 'solidity': 0.5}
    
    def _temporal_changes(self, current_roi, previous_roi, mask):
        """Analyze temporal changes."""
        # Calculate difference
        diff = np.abs(current_roi.astype(float) - previous_roi.astype(float))
        masked_diff = diff[mask]
        
        if len(masked_diff) == 0:
            return 0.5
        
        # High changes indicate death process
        mean_change = np.mean(masked_diff)
        return min(mean_change / 50, 1.0)
    
    def _calculate_death_score(self, features):
        """Calculate overall death score from features."""
        weights = {
            'disappeared_score': 0.3,
            'blebbing_score': 0.2,
            'shrinkage_score': 0.2,
            'shape_score': 0.15,
            'texture_score': 0.1,
            'temporal_score': 0.05
        }
        
        total_score = sum(features.get(k, 0) * v for k, v in weights.items())
        return min(total_score, 1.0)
    
    def _classify_state(self, death_score):
        """Classify cell state based on death score."""
        if death_score >= 0.8:
            return 'dead'
        elif death_score >= 0.5:
            return 'dying'
        else:
            return 'alive'


class BrightfieldTracker:
    """Simple tracker that maintains cell identities across frames."""
    
    def __init__(self, max_distance=30):
        self.max_distance = max_distance
        self.tracks = {}
        self.next_id = 1
        self.global_id_map = {}  # Maps (droplet_id, local_id) to global_id
        
    def update(self, current_cells, droplet_id, frame_num):
        """Update tracks for cells in a specific droplet."""
        # Get previous cells for this droplet
        droplet_tracks = {tid: track for tid, track in self.tracks.items() 
                         if track.get('droplet_id') == droplet_id and 
                         track['last_seen'] == frame_num - 1}
        
        assignments = {}
        
        if not droplet_tracks:
            # First frame or no previous tracks - assign new IDs
            for i, cell in enumerate(current_cells):
                global_id = self.next_id
                self.next_id += 1
                
                self.tracks[global_id] = {
                    'droplet_id': droplet_id,
                    'first_frame': frame_num,
                    'last_seen': frame_num,
                    'positions': [(cell['centroid_x'], cell['centroid_y'])],
                    'death_scores': [cell.get('death_score', 0)],
                    'states': [cell.get('state', 'alive')]
                }
                
                assignments[i] = global_id
                self.global_id_map[(droplet_id, i)] = global_id
        else:
            # Match cells to existing tracks
            used_tracks = set()
            
            for i, cell in enumerate(current_cells):
                best_dist = float('inf')
                best_track = None
                
                for tid, track in droplet_tracks.items():
                    if tid in used_tracks:
                        continue
                    
                    last_pos = track['positions'][-1]
                    dist = np.sqrt((cell['centroid_x'] - last_pos[0])**2 + 
                                 (cell['centroid_y'] - last_pos[1])**2)
                    
                    if dist < best_dist and dist < self.max_distance:
                        best_dist = dist
                        best_track = tid
                
                if best_track:
                    # Update existing track
                    self.tracks[best_track]['last_seen'] = frame_num
                    self.tracks[best_track]['positions'].append(
                        (cell['centroid_x'], cell['centroid_y'])
                    )
                    self.tracks[best_track]['death_scores'].append(
                        cell.get('death_score', 0)
                    )
                    self.tracks[best_track]['states'].append(
                        cell.get('state', 'alive')
                    )
                    
                    assignments[i] = best_track
                    used_tracks.add(best_track)
                else:
                    # New cell
                    global_id = self.next_id
                    self.next_id += 1
                    
                    self.tracks[global_id] = {
                        'droplet_id': droplet_id,
                        'first_frame': frame_num,
                        'last_seen': frame_num,
                        'positions': [(cell['centroid_x'], cell['centroid_y'])],
                        'death_scores': [cell.get('death_score', 0)],
                        'states': [cell.get('state', 'alive')]
                    }
                    
                    assignments[i] = global_id
        
        return assignments


class CancerSurvivalAnalyzerBF:
    """Complete analyzer using brightfield death detection."""
    
    def __init__(self, nd2_file, time_interval_min=15):
        self.nd2_file = nd2_file
        self.time_interval = time_interval_min
        self.analyzer = None
        self.death_detector = BrightfieldDeathDetector()
        self.tracker = BrightfieldTracker()
        self.droplets = None
        self.masks = None
        self.frame_cache = []
        self.results = []
        
    def analyze(self, output_dir=None):
        """Run complete analysis with brightfield death detection."""
        print(f"Loading {self.nd2_file}...")
        
        # Load ND2 file
        self.analyzer = ND2Analyzer(self.nd2_file)
        if not self.analyzer.load_file():
            return False
        
        # Create output directory
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(self.nd2_file), "bf_death_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        # Detect droplets
        print("Detecting droplets...")
        droplet_detector = DropletDetector()
        bf_frame = self.analyzer.get_frame(0, 'brightfield')
        self.droplets = droplet_detector.detect_droplets(bf_frame)
        self.masks, _ = droplet_detector.create_droplet_masks(bf_frame.shape, self.droplets)
        
        print(f"Found {len(self.droplets)} droplets")
        
        # Initialize cell analyzer
        cell_analyzer = DropletCellAnalyzer(droplet_detector, None)
        
        # Process all frames
        previous_bf = None
        frame_results = []
        
        for t in range(self.analyzer.metadata['frames']):
            if t % 10 == 0:
                print(f"Processing frame {t}/{self.analyzer.metadata['frames']-1}")
            
            # Get frames
            bf_frame = self.analyzer.get_frame(t, 'brightfield')
            tritc_frame = self.analyzer.get_frame(t, 'TRITC')
            
            if bf_frame is None or tritc_frame is None:
                continue
            
            # Store for movie generation
            self.frame_cache.append({
                'timepoint': t,
                'brightfield': bf_frame,
                'tritc': tritc_frame
            })
            
            # Process each droplet
            droplet_results = {}
            
            for droplet in self.droplets:
                did = droplet['id']
                mask = self.masks[did]
                
                # Extract regions
                masked_tritc = tritc_frame.copy()
                masked_tritc[~mask] = 0
                masked_bf = bf_frame.copy()
                masked_bf[~mask] = 0
                
                # Detect cells using TRITC
                nuclei = cell_analyzer._detect_nuclei(masked_tritc, masked_bf, mask, droplet)
                
                # Convert to cell list with brightfield analysis
                cancer_cells = []
                for nx, ny, intensity, area in nuclei:
                    cancer_cells.append({
                        'centroid_x': nx,
                        'centroid_y': ny,
                        'mean_intensity': intensity,
                        'area': area
                    })
                
                # Apply brightfield death detection
                for i, cell in enumerate(cancer_cells):
                    # Analyze brightfield morphology
                    death_result = self.death_detector.analyze_cell_morphology(
                        bf_frame,
                        tritc_frame,
                        (cell['centroid_x'], cell['centroid_y']),
                        f"{did}_{i}_{t}",  # Unique cell ID
                        previous_bf=previous_bf,
                        store_reference=(t < 5)  # Store early frames as reference
                    )
                    
                    # Add death detection results to cell
                    cell['death_score'] = death_result['death_score']
                    cell['state'] = death_result['state']
                    cell['death_features'] = death_result['features']
                
                # Update tracking
                track_assignments = self.tracker.update(cancer_cells, did, t)
                
                # Calculate statistics
                alive = sum(1 for c in cancer_cells if c['state'] == 'alive')
                dying = sum(1 for c in cancer_cells if c['state'] == 'dying')
                dead = sum(1 for c in cancer_cells if c['state'] == 'dead')
                
                # Also count cells that disappeared (from tracking history)
                disappeared = 0
                for tid, track in self.tracker.tracks.items():
                    if (track['droplet_id'] == did and 
                        track['last_seen'] < t - 3 and 
                        'dead' in track['states'][-1:]):
                        disappeared += 1
                
                droplet_results[did] = {
                    'cells': cancer_cells,
                    'track_assignments': track_assignments,
                    'alive': alive,
                    'dying': dying,
                    'dead': dead,
                    'disappeared': disappeared,
                    'total': alive + dying
                }
            
            # Store frame results
            frame_results.append({
                'timepoint': t,
                'time_min': t * self.time_interval,
                'droplet_results': droplet_results
            })
            
            # Store previous brightfield
            previous_bf = bf_frame
        
        self.results = frame_results
        
        # Generate outputs
        print("Generating outputs...")
        self._save_results(output_dir)
        self._create_movie(output_dir)
        self._create_plots(output_dir)
        
        # Close analyzer
        self.analyzer.close()
        
        print(f"Analysis complete! Results saved to: {output_dir}")
        return True
    
    def _save_results(self, output_dir):
        """Save analysis results to Excel."""
        base_name = os.path.splitext(os.path.basename(self.nd2_file))[0]
        excel_path = os.path.join(output_dir, f"{base_name}_bf_death_analysis.xlsx")
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Time series data
            rows = []
            for frame in self.results:
                for did, data in frame['droplet_results'].items():
                    rows.append({
                        'timepoint': frame['timepoint'],
                        'time_min': frame['time_min'],
                        'droplet_id': did,
                        'alive': data['alive'],
                        'dying': data['dying'],
                        'dead': data['dead'],
                        'disappeared': data['disappeared'],
                        'total': data['total']
                    })
            
            df_timeseries = pd.DataFrame(rows)
            df_timeseries.to_excel(writer, sheet_name='Time_Series', index=False)
            
            # Death events
            death_rows = []
            for tid, track in self.tracker.tracks.items():
                # Find when cell died
                for i, state in enumerate(track['states']):
                    if state == 'dead' and (i == 0 or track['states'][i-1] != 'dead'):
                        death_rows.append({
                            'cell_id': tid,
                            'droplet_id': track['droplet_id'],
                            'death_frame': track['first_frame'] + i,
                            'death_time_min': (track['first_frame'] + i) * self.time_interval,
                            'max_death_score': max(track['death_scores']),
                            'lifespan_frames': i
                        })
                        break
            
            if death_rows:
                df_deaths = pd.DataFrame(death_rows)
                df_deaths.to_excel(writer, sheet_name='Death_Events', index=False)
            
            # Summary
            summary_rows = []
            for did in self.masks.keys():
                # Get initial and final counts
                initial = next((r for r in rows if r['droplet_id'] == did and r['timepoint'] == 0), {})
                final = next((r for r in rows if r['droplet_id'] == did and 
                            r['timepoint'] == self.results[-1]['timepoint']), {})
                
                droplet_deaths = [d for d in death_rows if d.get('droplet_id') == did]
                
                summary_rows.append({
                    'droplet_id': did,
                    'initial_cells': initial.get('total', 0),
                    'final_alive': final.get('alive', 0),
                    'total_deaths': len(droplet_deaths) + final.get('disappeared', 0),
                    'survival_rate_%': (final.get('alive', 0) / initial.get('total', 1) * 100) 
                                     if initial.get('total', 0) > 0 else 0
                })
            
            df_summary = pd.DataFrame(summary_rows)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"Results saved to: {excel_path}")
    
    def _create_movie(self, output_dir, fps=3):
        """Create movie showing brightfield death detection."""
        base_name = os.path.splitext(os.path.basename(self.nd2_file))[0]
        movie_path = os.path.join(output_dir, f"{base_name}_bf_death_movie.mp4")
        
        print(f"Creating movie: {movie_path}")
        
        # Setup figure
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 3, figure=fig)
        
        # Axes
        ax_bf = fig.add_subplot(gs[0, 0])
        ax_tritc = fig.add_subplot(gs[0, 1])
        ax_overlay = fig.add_subplot(gs[0, 2])
        ax_survival = fig.add_subplot(gs[1, :2])
        ax_info = fig.add_subplot(gs[1, 2])
        
        # Initialize plots
        frame0 = self.frame_cache[0]
        
        # Brightfield
        bf_img = ax_bf.imshow(frame0['brightfield'], cmap='gray')
        ax_bf.set_title('Brightfield')
        ax_bf.axis('off')
        
        # TRITC
        tritc = frame0['tritc']
        vmin, vmax = np.percentile(tritc[tritc > 0], [5, 99.5]) if np.any(tritc > 0) else (0, 1)
        tritc_enhanced = np.clip((tritc - vmin) / (vmax - vmin + 1e-8), 0, 1)
        tritc_img = ax_tritc.imshow(tritc_enhanced, cmap='hot', vmin=0, vmax=1)
        ax_tritc.set_title('TRITC (Cell Nuclei)')
        ax_tritc.axis('off')
        
        # Overlay
        overlay_img = ax_overlay.imshow(frame0['brightfield'], cmap='gray')
        ax_overlay.set_title('Death Detection')
        ax_overlay.axis('off')
        
        # Draw droplets
        for droplet in self.droplets:
            for ax in [ax_bf, ax_tritc, ax_overlay]:
                circle = plt.Circle((droplet['center_x'], droplet['center_y']), 
                                  droplet['radius_px'], 
                                  color='lime', fill=False, linewidth=1)
                ax.add_patch(circle)
        
        # Survival curves
        ax_survival.set_xlabel('Time (minutes)')
        ax_survival.set_ylabel('Live Cells')
        ax_survival.set_title('Cell Survival by Droplet')
        ax_survival.grid(True, alpha=0.3)
        
        # Info text
        ax_info.axis('off')
        info_text = ax_info.text(0.1, 0.9, '', transform=ax_info.transAxes, 
                               fontsize=10, verticalalignment='top')
        
        # Cell markers
        cell_markers = []
        survival_lines = {}
        
        # Animation function
        def update_frame(frame_idx):
            frame_data = self.results[frame_idx]
            frame = self.frame_cache[frame_idx]
            
            # Update images
            bf_img.set_data(frame['brightfield'])
            
            # Update TRITC
            tritc = frame['tritc']
            vmin, vmax = np.percentile(tritc[tritc > 0], [5, 99.5]) if np.any(tritc > 0) else (0, 1)
            tritc_enhanced = np.clip((tritc - vmin) / (vmax - vmin + 1e-8), 0, 1)
            tritc_img.set_data(tritc_enhanced)
            
            # Update overlay
            overlay_img.set_data(frame['brightfield'])
            
            # Clear old markers
            for marker in cell_markers:
                marker.remove()
            cell_markers.clear()
            
            # Add cell state markers
            total_alive = 0
            total_dying = 0
            total_dead = 0
            
            for did, data in frame_data['droplet_results'].items():
                for cell in data['cells']:
                    x, y = cell['centroid_x'], cell['centroid_y']
                    state = cell['state']
                    
                    if state == 'alive':
                        marker = ax_overlay.plot(x, y, 'o', color='green', 
                                               markersize=8, markeredgecolor='white',
                                               markeredgewidth=1)[0]
                        total_alive += 1
                    elif state == 'dying':
                        marker = ax_overlay.plot(x, y, 's', color='yellow', 
                                               markersize=10, markeredgecolor='black',
                                               markeredgewidth=1)[0]
                        total_dying += 1
                        
                        # Add death score
                        score_text = ax_overlay.text(x, y-15, f"{cell['death_score']:.2f}", 
                                                   color='yellow', fontsize=6, ha='center',
                                                   bbox=dict(boxstyle="round,pad=0.2", 
                                                           facecolor='black', alpha=0.7))
                        cell_markers.append(score_text)
                    else:  # dead
                        marker = ax_overlay.plot(x, y, 'x', color='red', 
                                               markersize=12, markeredgewidth=3)[0]
                        total_dead += 1
                    
                    cell_markers.append(marker)
                
                total_dead += data['disappeared']
            
            # Update survival curves
            times = [r['time_min'] for r in self.results[:frame_idx+1]]
            
            for did in self.masks.keys():
                counts = []
                for r in self.results[:frame_idx+1]:
                    if did in r['droplet_results']:
                        counts.append(r['droplet_results'][did]['alive'])
                    else:
                        counts.append(0)
                
                if did in survival_lines:
                    survival_lines[did].set_data(times, counts)
                else:
                    line, = ax_survival.plot(times, counts, marker='o', markersize=4,
                                           label=f'Droplet {did}')
                    survival_lines[did] = line
            
            # Update legend once
            if frame_idx == 0:
                ax_survival.legend(loc='upper right')
            
            # Update info text
            info_str = f"Time: {frame_data['time_min']:.0f} min\n\n"
            info_str += f"Total Cells:\n"
            info_str += f"  Alive: {total_alive}\n"
            info_str += f"  Dying: {total_dying}\n"
            info_str += f"  Dead: {total_dead}\n\n"
            
            # Add death detection info
            info_str += "Death Detection Features:\n"
            info_str += "• Disappearance/Lysis\n"
            info_str += "• Membrane Blebbing\n"
            info_str += "• Cell Shrinkage\n"
            info_str += "• Shape Changes\n"
            info_str += "• Texture Changes\n\n"
            
            # Survival rate
            initial_total = sum(self.results[0]['droplet_results'][d]['total'] 
                              for d in self.results[0]['droplet_results'])
            if initial_total > 0:
                survival_rate = (total_alive / initial_total) * 100
                info_str += f"Survival Rate: {survival_rate:.1f}%"
            
            info_text.set_text(info_str)
            
            # Update title
            fig.suptitle(f'Brightfield Death Detection - T = {frame_data["time_min"]:.0f} minutes', 
                        fontsize=16)
            
            return [bf_img, tritc_img, overlay_img, info_text] + cell_markers + list(survival_lines.values())
        
        # Create animation
        anim = animation.FuncAnimation(fig, update_frame, frames=len(self.results),
                                     interval=1000/fps, blit=False)
        
        # Save movie
        self._save_animation(anim, movie_path, fps)
        plt.close()
        
        print(f"Movie saved: {movie_path}")
    
    def _save_animation(self, anim, output_path, fps):
        """Save animation with proper ffmpeg handling."""
        ffmpeg_path = r"D:\ffmpeg-2025-06-28-git-cfd1f81e7d-full_build\bin\ffmpeg.exe"
        
        if os.path.exists(ffmpeg_path):
            # Temporarily add FFmpeg to PATH
            original_path = os.environ.get('PATH', '')
            os.environ['PATH'] = os.path.dirname(ffmpeg_path) + os.pathsep + original_path
            
            try:
                writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
                anim.save(output_path, writer=writer)
            finally:
                # Restore original PATH
                os.environ['PATH'] = original_path
        else:
            # Try default writer
            try:
                writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
                anim.save(output_path, writer=writer)
            except:
                print("Warning: Could not save movie. FFmpeg not found.")
    
    def _create_plots(self, output_dir):
        """Create analysis plots."""
        base_name = os.path.splitext(os.path.basename(self.nd2_file))[0]
        
        # Survival curves plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Individual droplet survival
        for did in self.masks.keys():
            times = []
            alive_counts = []
            
            for frame in self.results:
                times.append(frame['time_min'])
                if did in frame['droplet_results']:
                    alive_counts.append(frame['droplet_results'][did]['alive'])
                else:
                    alive_counts.append(0)
            
            if max(alive_counts) > 0:  # Only plot droplets that had cells
                ax1.plot(times, alive_counts, marker='o', markersize=4, 
                        label=f'Droplet {did}', linewidth=2)
        
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Number of Live Cells')
        ax1.set_title('Cell Survival by Droplet')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Total survival and death states
        times = []
        total_alive = []
        total_dying = []
        total_dead = []
        
        for frame in self.results:
            times.append(frame['time_min'])
            alive = dying = dead = 0
            
            for did, data in frame['droplet_results'].items():
                alive += data['alive']
                dying += data['dying']
                dead += data['dead'] + data['disappeared']
            
            total_alive.append(alive)
            total_dying.append(dying)
            total_dead.append(dead)
        
        ax2.plot(times, total_alive, 'g-', label='Alive', linewidth=3, marker='o')
        ax2.plot(times, total_dying, 'y-', label='Dying', linewidth=3, marker='s')
        ax2.plot(times, total_dead, 'r-', label='Dead', linewidth=3, marker='x')
        
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Number of Cells')
        ax2.set_title('Cell States Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_survival_curves.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Death feature analysis
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Collect all death features
        all_features = {
            'disappeared_score': [],
            'blebbing_score': [],
            'shrinkage_score': [],
            'shape_score': [],
            'texture_score': [],
            'temporal_score': []
        }
        
        for frame in self.results:
            for did, data in frame['droplet_results'].items():
                for cell in data['cells']:
                    if 'death_features' in cell and cell['state'] in ['dying', 'dead']:
                        for feat, val in cell['death_features'].items():
                            if feat in all_features:
                                all_features[feat].append(val)
        
        # Plot histograms
        for i, (feat_name, values) in enumerate(all_features.items()):
            if i < len(axes) and values:
                axes[i].hist(values, bins=20, alpha=0.7, color='red')
                axes[i].set_xlabel('Score')
                axes[i].set_ylabel('Count')
                axes[i].set_title(feat_name.replace('_', ' ').title())
                axes[i].axvline(0.5, color='black', linestyle='--', alpha=0.5)
        
        plt.suptitle('Death Feature Distributions for Dying/Dead Cells', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_death_features.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Plots saved to output directory")


def run_brightfield_death_analysis(nd2_file, output_dir=None):
    """
    Main function to run brightfield-based death detection analysis.
    
    Args:
        nd2_file: Path to ND2 file
        output_dir: Output directory (optional)
    
    Returns:
        analyzer object if successful, None otherwise
    """
    analyzer = CancerSurvivalAnalyzerBF(nd2_file)
    
    if analyzer.analyze(output_dir):
        return analyzer
    else:
        return None


# Example usage to integrate with your existing code
def integrate_with_existing_pipeline(nd2_file):
    """
    Example of how to integrate this with your existing analysis pipeline.
    """
    # Run the brightfield death analysis
    analyzer = run_brightfield_death_analysis(nd2_file)
    
    if analyzer:
        # Access results
        results = analyzer.results
        tracks = analyzer.tracker.tracks
        
        # You can now use these results in your other analyses
        print(f"Analysis complete. Tracked {len(tracks)} cells across all droplets.")
        
        # Example: Get death times
        death_times = []
        for tid, track in tracks.items():
            for i, state in enumerate(track['states']):
                if state == 'dead' and (i == 0 or track['states'][i-1] != 'dead'):
                    death_time = (track['first_frame'] + i) * analyzer.time_interval
                    death_times.append(death_time)
                    break
        
        if death_times:
            print(f"Average death time: {np.mean(death_times):.1f} minutes")
        
        return analyzer
    else:
        print("Analysis failed")
        return None


if __name__ == "__main__":
    # Test the analyzer
    nd2_file = r"D:\New\BrainBites\Cell\2.nd2"
    
    print("Running Brightfield Death Detection Analysis...")
    print("=" * 60)
    
    # Run analysis
    analyzer = run_brightfield_death_analysis(nd2_file)
    
    if analyzer:
        print("\nAnalysis completed successfully!")
        print("\nKey features of this analysis:")
        print("1. Uses TRITC to locate cells")
        print("2. Analyzes brightfield morphology for death detection")
        print("3. Detects: disappearance, blebbing, shrinkage, shape changes")
        print("4. Tracks cells across frames")
        print("5. Generates movies and Excel reports")
    else:
        print("\nAnalysis failed. Please check your ND2 file path.")