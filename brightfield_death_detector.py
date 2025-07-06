"""
Standalone Brightfield Cell Death Detector
=========================================
Complete application for detecting cell death using brightfield morphology
at TRITC-identified cell locations.

Author: AI Assistant
Date: 2025-01-20

Requirements:
- numpy
- opencv-python
- scipy
- scikit-image
- matplotlib
- pandas
- nd2reader

Usage:
    python brightfield_death_detector.py --input your_file.nd2 --output results_folder
"""

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches, animation
from matplotlib.gridspec import GridSpec
import os
import argparse
from datetime import datetime
from pathlib import Path

from nd2reader import ND2Reader
from scipy import ndimage
from scipy.ndimage import label, binary_erosion, binary_dilation, distance_transform_edt
from skimage import morphology, measure, filters, feature
from skimage.filters import threshold_otsu, gaussian
from skimage.feature import graycomatrix, graycoprops, peak_local_max
from skimage.segmentation import watershed


class CellDeathDetector:
    """Main class for detecting cell death using brightfield morphology."""
    
    def __init__(self, nd2_file_path):
        self.file_path = Path(nd2_file_path)
        self.reader = None
        self.metadata = {}
        self.cell_tracker = CellTracker()
        self.morphology_analyzer = MorphologyAnalyzer()
        self.results = []
        
    def load_file(self):
        """Load ND2 file and extract metadata."""
        try:
            self.reader = ND2Reader(str(self.file_path))
            self.metadata = {
                'width': self.reader.metadata['width'],
                'height': self.reader.metadata['height'],
                'frames': self.reader.metadata['total_images_per_channel'],
                'channels': self.reader.metadata['channels'],
                'time_interval': 15  # minutes
            }
            print(f"Successfully loaded: {self.file_path.name}")
            print(f"Dimensions: {self.metadata['width']} x {self.metadata['height']}")
            print(f"Time points: {self.metadata['frames']}")
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def get_frame(self, timepoint=0, channel='brightfield'):
        """Get specific frame from ND2 file."""
        if not self.reader:
            return None
        
        try:
            self.reader.iter_axes = 't'
            channel_idx = 0 if channel.lower() == 'brightfield' else 1
            self.reader.default_coords['c'] = channel_idx
            frame = self.reader[timepoint]
            return np.array(frame) if frame is not None else None
        except Exception as e:
            print(f"Error getting frame: {e}")
            return None
    
    def detect_cells_in_tritc(self, tritc_image):
        """Detect cell nuclei in TRITC channel."""
        # Denoise
        denoised = gaussian(tritc_image, sigma=1)
        
        # Threshold
        thresh = threshold_otsu(denoised)
        binary = denoised > thresh
        
        # Remove small objects
        binary = morphology.remove_small_objects(binary, min_size=50)
        
        # Watershed to separate touching nuclei
        distance = distance_transform_edt(binary)
        local_maxima_coords = peak_local_max(distance, min_distance=10)
        local_maxima = np.zeros_like(distance, dtype=bool)
        local_maxima[local_maxima_coords[:, 0], local_maxima_coords[:, 1]] = True
        markers = label(local_maxima)[0]
        labels = watershed(-distance, markers, mask=binary)
        
        # Extract cell properties
        props = measure.regionprops(labels, intensity_image=tritc_image)
        
        cells = []
        for prop in props:
            cells.append({
                'id': prop.label,
                'centroid': prop.centroid,
                'area': prop.area,
                'intensity': prop.mean_intensity,
                'bbox': prop.bbox
            })
        
        return cells, labels
    
    def analyze_all_frames(self):
        """Analyze all frames in the ND2 file."""
        print("\nStarting analysis...")
        
        for t in range(self.metadata['frames']):
            if t % 10 == 0:
                print(f"Processing frame {t}/{self.metadata['frames']-1}")
            
            # Get both channels
            bf_frame = self.get_frame(t, 'brightfield')
            tritc_frame = self.get_frame(t, 'TRITC')
            
            if bf_frame is None or tritc_frame is None:
                continue
            
            # Detect cells in TRITC
            cells, labels = self.detect_cells_in_tritc(tritc_frame)
            
            # Update tracking
            self.cell_tracker.update(cells, t)
            
            # Analyze each cell in brightfield
            frame_results = {
                'timepoint': t,
                'time_min': t * self.metadata['time_interval'],
                'cells': []
            }
            
            for cell in cells:
                # Get cell track
                track_id = self.cell_tracker.get_track_id(cell['id'], t)
                
                # Analyze morphology in brightfield
                morphology_result = self.morphology_analyzer.analyze_cell(
                    bf_frame, 
                    tritc_frame,
                    cell,
                    track_id,
                    t
                )
                
                # Store results
                cell_result = {
                    'track_id': track_id,
                    'position': cell['centroid'],
                    'tritc_area': cell['area'],
                    'tritc_intensity': cell['intensity'],
                    **morphology_result
                }
                
                frame_results['cells'].append(cell_result)
            
            self.results.append(frame_results)
        
        print("Analysis complete!")
        return self.results
    
    def save_results(self, output_dir):
        """Save analysis results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create summary DataFrame
        rows = []
        for frame in self.results:
            for cell in frame['cells']:
                row = {
                    'timepoint': frame['timepoint'],
                    'time_min': frame['time_min'],
                    'track_id': cell['track_id'],
                    'x': cell['position'][1],
                    'y': cell['position'][0],
                    'state': cell['state'],
                    'death_score': cell['death_score'],
                    'disappeared': cell['features']['disappeared'],
                    'blebbing': cell['features']['blebbing'],
                    'shrinkage': cell['features']['shrinkage'],
                    'shape_change': cell['features']['shape_change'],
                    'texture_change': cell['features']['texture_change']
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save to Excel
        excel_path = output_path / f"{self.file_path.stem}_death_analysis.xlsx"
        with pd.ExcelWriter(excel_path) as writer:
            # Time series data
            df.to_excel(writer, sheet_name='Time_Series', index=False)
            
            # Summary by cell
            cell_summary = self._create_cell_summary(df)
            cell_summary.to_excel(writer, sheet_name='Cell_Summary', index=False)
            
            # Death events
            death_events = self._extract_death_events(df)
            death_events.to_excel(writer, sheet_name='Death_Events', index=False)
        
        print(f"Results saved to: {excel_path}")
        
        # Create visualizations
        self._create_summary_plots(df, output_path)
        
        return df
    
    def _create_cell_summary(self, df):
        """Create summary statistics for each cell."""
        summary = []
        
        for track_id in df['track_id'].unique():
            cell_data = df[df['track_id'] == track_id]
            
            # Get first and last states
            first_state = cell_data.iloc[0]['state']
            last_state = cell_data.iloc[-1]['state']
            
            # Find death time if applicable
            death_time = None
            if 'dead' in cell_data['state'].values:
                death_idx = cell_data[cell_data['state'] == 'dead'].iloc[0]['timepoint']
                death_time = death_idx * self.metadata['time_interval']
            
            summary.append({
                'track_id': track_id,
                'first_seen': cell_data.iloc[0]['time_min'],
                'last_seen': cell_data.iloc[-1]['time_min'],
                'initial_state': first_state,
                'final_state': last_state,
                'death_time_min': death_time,
                'max_death_score': cell_data['death_score'].max(),
                'showed_blebbing': cell_data['blebbing'].max() > 0.5,
                'showed_shrinkage': cell_data['shrinkage'].max() > 0.5
            })
        
        return pd.DataFrame(summary)
    
    def _extract_death_events(self, df):
        """Extract death events from time series."""
        events = []
        
        for track_id in df['track_id'].unique():
            cell_data = df[df['track_id'] == track_id].sort_values('timepoint')
            
            # Find state transitions
            states = cell_data['state'].values
            times = cell_data['time_min'].values
            
            for i in range(1, len(states)):
                if states[i-1] != 'dead' and states[i] == 'dead':
                    events.append({
                        'track_id': track_id,
                        'death_time_min': times[i],
                        'death_timepoint': cell_data.iloc[i]['timepoint'],
                        'pre_death_state': states[i-1],
                        'death_score': cell_data.iloc[i]['death_score'],
                        'primary_cause': self._identify_primary_cause(cell_data.iloc[i])
                    })
        
        return pd.DataFrame(events)
    
    def _identify_primary_cause(self, cell_data):
        """Identify primary cause of death."""
        features = {
            'disappeared': cell_data['disappeared'],
            'blebbing': cell_data['blebbing'],
            'shrinkage': cell_data['shrinkage'],
            'shape_change': cell_data['shape_change'],
            'texture_change': cell_data['texture_change']
        }
        
        return max(features.items(), key=lambda x: x[1])[0]
    
    def _create_summary_plots(self, df, output_path):
        """Create summary visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Cell states over time
        ax = axes[0, 0]
        for state in ['alive', 'dying', 'dead']:
            counts = []
            times = []
            for t in df['timepoint'].unique():
                frame_data = df[df['timepoint'] == t]
                count = len(frame_data[frame_data['state'] == state])
                counts.append(count)
                times.append(t * self.metadata['time_interval'])
            ax.plot(times, counts, marker='o', label=state)
        
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Cell Count')
        ax.set_title('Cell States Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Death score distribution
        ax = axes[0, 1]
        death_scores = df.groupby('track_id')['death_score'].max()
        ax.hist(death_scores, bins=20, edgecolor='black')
        ax.set_xlabel('Maximum Death Score')
        ax.set_ylabel('Number of Cells')
        ax.set_title('Death Score Distribution')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Death features
        ax = axes[1, 0]
        features = ['disappeared', 'blebbing', 'shrinkage', 'shape_change', 'texture_change']
        feature_means = [df[df['state'] == 'dead'][f].mean() if len(df[df['state'] == 'dead']) > 0 else 0 
                        for f in features]
        ax.bar(features, feature_means)
        ax.set_ylabel('Average Score')
        ax.set_title('Death Features in Dead Cells')
        ax.set_xticklabels(features, rotation=45)
        
        # Plot 4: Survival curve
        ax = axes[1, 1]
        initial_cells = len(df[df['timepoint'] == 0])
        survival = []
        times = []
        
        for t in sorted(df['timepoint'].unique()):
            frame_data = df[df['timepoint'] == t]
            alive_count = len(frame_data[frame_data['state'] != 'dead'])
            survival.append(alive_count / initial_cells * 100)
            times.append(t * self.metadata['time_interval'])
        
        ax.plot(times, survival, 'b-', linewidth=2)
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Survival (%)')
        ax.set_title('Cell Survival Curve')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        plt.savefig(output_path / f"{self.file_path.stem}_summary_plots.png", dpi=300)
        plt.show()
    
    def create_movie(self, output_dir, fps=3):
        """Create annotated movie showing death detection."""
        output_path = Path(output_dir)
        movie_path = output_path / f"{self.file_path.stem}_death_detection.mp4"
        
        print(f"Creating movie: {movie_path}")
        
        # Setup figure
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, height_ratios=[2, 2, 1])
        
        # Main displays
        ax_bf = fig.add_subplot(gs[0, 0])
        ax_tritc = fig.add_subplot(gs[0, 1])
        ax_overlay = fig.add_subplot(gs[0, 2])
        
        # Feature displays
        ax_features = fig.add_subplot(gs[1, :])
        
        # Timeline
        ax_timeline = fig.add_subplot(gs[2, :])
        
        # Initialize with first frame
        bf_frame = self.get_frame(0, 'brightfield')
        tritc_frame = self.get_frame(0, 'TRITC')
        
        # Display images
        im_bf = ax_bf.imshow(bf_frame, cmap='gray')
        ax_bf.set_title('Brightfield')
        ax_bf.axis('off')
        
        im_tritc = ax_tritc.imshow(tritc_frame, cmap='hot')
        ax_tritc.set_title('TRITC (Nuclear Stain)')
        ax_tritc.axis('off')
        
        im_overlay = ax_overlay.imshow(bf_frame, cmap='gray')
        ax_overlay.set_title('Cell States')
        ax_overlay.axis('off')
        
        # Cell markers
        cell_markers = []
        
        # Timeline setup
        times = [r['time_min'] for r in self.results]
        alive_counts = []
        dead_counts = []
        
        for result in self.results:
            alive = sum(1 for c in result['cells'] if c['state'] == 'alive')
            dead = sum(1 for c in result['cells'] if c['state'] == 'dead')
            alive_counts.append(alive)
            dead_counts.append(dead)
        
        line_alive, = ax_timeline.plot([], [], 'g-', label='Alive')
        line_dead, = ax_timeline.plot([], [], 'r-', label='Dead')
        ax_timeline.set_xlim(0, max(times))
        ax_timeline.set_ylim(0, max(alive_counts) * 1.1)
        ax_timeline.set_xlabel('Time (minutes)')
        ax_timeline.set_ylabel('Cell Count')
        ax_timeline.legend()
        ax_timeline.grid(True, alpha=0.3)
        
        # Progress marker
        progress_line = ax_timeline.axvline(0, color='black', linestyle='--', alpha=0.5)
        
        def update_frame(frame_idx):
            """Update animation frame."""
            # Clear old markers
            for marker in cell_markers:
                marker.remove()
            cell_markers.clear()
            
            # Get frame data
            frame_result = self.results[frame_idx]
            t = frame_result['timepoint']
            
            # Update images
            bf_frame = self.get_frame(t, 'brightfield')
            tritc_frame = self.get_frame(t, 'TRITC')
            
            im_bf.set_data(bf_frame)
            im_tritc.set_data(tritc_frame)
            im_overlay.set_data(bf_frame)
            
            # Update title
            fig.suptitle(f"Time: {frame_result['time_min']:.0f} minutes", fontsize=16)
            
            # Draw cell states
            for cell in frame_result['cells']:
                y, x = cell['position']
                state = cell['state']
                
                # Color and marker based on state
                if state == 'alive':
                    color, marker = 'green', 'o'
                elif state == 'dying':
                    color, marker = 'yellow', 's'
                else:  # dead
                    color, marker = 'red', 'x'
                
                m = ax_overlay.plot(x, y, marker, color=color, markersize=12, 
                                   markeredgewidth=2, markeredgecolor='white')[0]
                cell_markers.append(m)
                
                # Add death score
                score = cell['death_score']
                if score > 0.3:
                    text = ax_overlay.text(x+10, y, f'{score:.2f}', color=color, 
                                         fontsize=8, weight='bold')
                    cell_markers.append(text)
            
            # Update feature display
            ax_features.clear()
            
            # Show top 5 cells by death score
            sorted_cells = sorted(frame_result['cells'], key=lambda c: c['death_score'], reverse=True)[:5]
            
            bar_width = 0.15
            x_positions = np.arange(len(sorted_cells))
            
            features = ['disappeared', 'blebbing', 'shrinkage', 'shape_change', 'texture_change']
            colors = ['black', 'red', 'orange', 'blue', 'green']
            
            for i, feature in enumerate(features):
                values = [c['features'][feature] for c in sorted_cells]
                ax_features.bar(x_positions + i * bar_width, values, bar_width, 
                              label=feature, color=colors[i])
            
            ax_features.set_xlabel('Cell (by death score)')
            ax_features.set_ylabel('Feature Score')
            ax_features.set_title('Death Features - Top 5 Cells')
            ax_features.set_xticks(x_positions + bar_width * 2)
            ax_features.set_xticklabels([f"Cell {c['track_id']}\n({c['state']})" for c in sorted_cells])
            ax_features.legend()
            ax_features.set_ylim(0, 1)
            
            # Update timeline
            current_time = frame_result['time_min']
            line_alive.set_data(times[:frame_idx+1], alive_counts[:frame_idx+1])
            line_dead.set_data(times[:frame_idx+1], dead_counts[:frame_idx+1])
            progress_line.set_xdata([current_time])
            
            return [im_bf, im_tritc, im_overlay] + cell_markers + [line_alive, line_dead, progress_line]
        
        # Create animation
        anim = animation.FuncAnimation(fig, update_frame, frames=len(self.results),
                                     interval=1000/fps, blit=False)
        
        # Save movie
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(str(movie_path), writer=writer)
        
        plt.close()
        print(f"Movie saved: {movie_path}")
    
    def close(self):
        """Close the ND2 reader."""
        if self.reader:
            self.reader.close()


class CellTracker:
    """Simple cell tracker to maintain cell identities across frames."""
    
    def __init__(self, max_distance=50):
        self.max_distance = max_distance
        self.tracks = {}
        self.next_id = 1
        self.frame_assignments = {}
    
    def update(self, cells, frame_num):
        """Update tracks with new cell detections."""
        # Store current frame assignments
        self.frame_assignments[frame_num] = {}
        
        if frame_num == 0:
            # First frame - assign new IDs
            for cell in cells:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    'positions': [cell['centroid']],
                    'frames': [frame_num]
                }
                self.frame_assignments[frame_num][cell['id']] = track_id
        else:
            # Match to previous frame
            prev_frame = frame_num - 1
            unmatched_cells = cells.copy()
            
            # Find matches
            for track_id, track in self.tracks.items():
                if track['frames'][-1] == prev_frame:
                    last_pos = track['positions'][-1]
                    
                    # Find nearest cell
                    best_dist = float('inf')
                    best_cell = None
                    
                    for cell in unmatched_cells:
                        dist = np.sqrt((cell['centroid'][0] - last_pos[0])**2 + 
                                     (cell['centroid'][1] - last_pos[1])**2)
                        
                        if dist < best_dist and dist < self.max_distance:
                            best_dist = dist
                            best_cell = cell
                    
                    if best_cell:
                        # Update track
                        track['positions'].append(best_cell['centroid'])
                        track['frames'].append(frame_num)
                        self.frame_assignments[frame_num][best_cell['id']] = track_id
                        unmatched_cells.remove(best_cell)
            
            # Create new tracks for unmatched cells
            for cell in unmatched_cells:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    'positions': [cell['centroid']],
                    'frames': [frame_num]
                }
                self.frame_assignments[frame_num][cell['id']] = track_id
    
    def get_track_id(self, cell_id, frame_num):
        """Get track ID for a cell in a specific frame."""
        return self.frame_assignments.get(frame_num, {}).get(cell_id, -1)


class MorphologyAnalyzer:
    """Analyze cell morphology in brightfield to detect death."""
    
    def __init__(self):
        self.cell_references = {}  # Store healthy cell appearances
    
    def analyze_cell(self, bf_image, tritc_image, cell_info, track_id, timepoint):
        """Analyze cell morphology to determine if it's dead."""
        # Extract ROI around cell
        y, x = int(cell_info['centroid'][0]), int(cell_info['centroid'][1])
        
        # ROI size based on cell size
        roi_size = int(np.sqrt(cell_info['area']) * 2)
        roi_size = max(30, min(roi_size, 60))  # Clamp size
        
        # Extract ROI
        y0 = max(0, y - roi_size)
        y1 = min(bf_image.shape[0], y + roi_size)
        x0 = max(0, x - roi_size)
        x1 = min(bf_image.shape[1], x + roi_size)
        
        bf_roi = bf_image[y0:y1, x0:x1].copy()
        tritc_roi = tritc_image[y0:y1, x0:x1].copy()
        
        # Get TRITC mask
        tritc_mask = tritc_roi > threshold_otsu(tritc_roi)
        
        # Analyze features
        features = {
            'disappeared': self._check_disappeared(bf_roi, tritc_mask),
            'blebbing': self._detect_blebbing(bf_roi, tritc_mask),
            'shrinkage': self._measure_shrinkage(tritc_mask, cell_info['area']),
            'shape_change': self._analyze_shape(bf_roi, tritc_mask),
            'texture_change': self._analyze_texture(bf_roi, tritc_mask, track_id)
        }
        
        # Calculate death score
        death_score = self._calculate_death_score(features)
        
        # Determine state
        if death_score >= 0.7:
            state = 'dead'
        elif death_score >= 0.4:
            state = 'dying'
        else:
            state = 'alive'
        
        # Store reference for healthy cells
        if state == 'alive' and timepoint < 5:
            self.cell_references[track_id] = {
                'texture': self._get_texture_stats(bf_roi, tritc_mask),
                'size': np.sum(tritc_mask)
            }
        
        return {
            'state': state,
            'death_score': death_score,
            'features': features
        }
    
    def _check_disappeared(self, bf_roi, mask):
        """Check if cell has disappeared (lysed)."""
        if np.sum(mask) == 0:
            return 1.0
        
        # Check contrast in masked region
        masked_pixels = bf_roi[mask]
        if len(masked_pixels) < 10:
            return 0.8
        
        # Low contrast indicates disappearance
        contrast = np.std(masked_pixels) / (np.mean(masked_pixels) + 1e-6)
        
        # Check edges
        edges = cv2.Canny(bf_roi.astype(np.uint8), 30, 100)
        edge_density = np.sum(edges[mask]) / np.sum(mask)
        
        # Combine metrics
        disappear_score = 1.0 - min(contrast * 5, 1.0) * min(edge_density * 10, 1.0)
        
        return disappear_score
    
    def _detect_blebbing(self, bf_roi, mask):
        """Detect membrane blebbing."""
        # Find cell boundary
        dilated = binary_dilation(mask, iterations=3)
        boundary = dilated & ~mask
        
        if np.sum(boundary) < 10:
            return 0.0
        
        # Look for irregularities at boundary
        boundary_pixels = bf_roi[boundary]
        
        # High variance at boundary suggests blebbing
        if len(boundary_pixels) > 0:
            variance = np.var(boundary_pixels)
            bleb_score = min(variance / 1000, 1.0)
        else:
            bleb_score = 0.0
        
        # Also check for small circular structures
        edges = cv2.Canny(bf_roi.astype(np.uint8), 50, 150)
        
        # Count edge pixels near boundary
        boundary_edges = edges & dilated
        edge_ratio = np.sum(boundary_edges) / np.sum(dilated)
        
        bleb_score = max(bleb_score, min(edge_ratio * 5, 1.0))
        
        return bleb_score
    
    def _measure_shrinkage(self, mask, original_area):
        """Measure cell shrinkage."""
        current_area = np.sum(mask)
        
        if original_area > 0:
            shrinkage = 1.0 - (current_area / original_area)
            return max(0, min(shrinkage, 1.0))
        
        return 0.0
    
    def _analyze_shape(self, bf_roi, mask):
        """Analyze shape irregularity."""
        if np.sum(mask) < 20:
            return 0.5
        
        # Find contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return 0.5
        
        # Use largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Calculate shape metrics
        area = cv2.contourArea(contour)
        if area < 10:
            return 0.7
        
        perimeter = cv2.arcLength(contour, True)
        
        # Circularity
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0
        
        # Convexity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Irregular shapes have low circularity and solidity
        irregularity = 1.0 - (circularity * 0.5 + solidity * 0.5)
        
        return min(irregularity, 1.0)
    
    def _analyze_texture(self, bf_roi, mask, track_id):
        """Analyze texture changes."""
        current_texture = self._get_texture_stats(bf_roi, mask)
        
        # Compare to reference if available
        if track_id in self.cell_references:
            ref_texture = self.cell_references[track_id]['texture']
            
            # Calculate change
            changes = []
            for key in current_texture:
                if key in ref_texture and ref_texture[key] > 0:
                    relative_change = abs(current_texture[key] - ref_texture[key]) / ref_texture[key]
                    changes.append(relative_change)
            
            if changes:
                texture_score = min(np.mean(changes) * 2, 1.0)
            else:
                texture_score = 0.5
        else:
            # No reference - use absolute values
            # Dead cells typically have higher contrast
            texture_score = min(current_texture.get('contrast', 0) / 50, 1.0)
        
        return texture_score
    
    def _get_texture_stats(self, roi, mask):
        """Calculate texture statistics."""
        if np.sum(mask) < 10:
            return {'contrast': 0, 'energy': 0, 'homogeneity': 1}
        
        # Get masked pixels
        masked = roi.copy()
        masked[~mask] = np.mean(roi[mask])
        
        # Normalize
        if masked.max() > masked.min():
            masked_norm = ((masked - masked.min()) / (masked.max() - masked.min()) * 255).astype(np.uint8)
        else:
            return {'contrast': 0, 'energy': 0, 'homogeneity': 1}
        
        try:
            # Calculate GLCM
            glcm = graycomatrix(masked_norm, distances=[1], angles=[0], 
                              levels=64, symmetric=True, normed=True)
            
            return {
                'contrast': graycoprops(glcm, 'contrast')[0, 0],
                'energy': graycoprops(glcm, 'energy')[0, 0],
                'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0]
            }
        except:
            return {'contrast': 0, 'energy': 0, 'homogeneity': 1}
    
    def _calculate_death_score(self, features):
        """Calculate overall death score."""
        weights = {
            'disappeared': 0.35,
            'blebbing': 0.25,
            'shrinkage': 0.20,
            'shape_change': 0.10,
            'texture_change': 0.10
        }
        
        score = sum(features.get(k, 0) * v for k, v in weights.items())
        return min(score, 1.0)


def main():
    """Main function to run the death detector."""
    parser = argparse.ArgumentParser(description='Detect cell death using brightfield morphology')
    parser.add_argument('--input', '-i', required=True, help='Input ND2 file path')
    parser.add_argument('--output', '-o', default='death_analysis', help='Output directory')
    parser.add_argument('--movie', '-m', action='store_true', help='Create annotated movie')
    parser.add_argument('--fps', type=int, default=3, help='Movie frame rate')
    
    args = parser.parse_args()
    
    # Create detector
    detector = CellDeathDetector(args.input)
    
    # Load file
    if not detector.load_file():
        print("Failed to load file!")
        return
    
    # Run analysis
    results = detector.analyze_all_frames()
    
    # Save results
    df = detector.save_results(args.output)
    
    # Create movie if requested
    if args.movie:
        detector.create_movie(args.output, args.fps)
    
    # Close
    detector.close()
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {args.output}")
    
    # Print summary
    print("\nSummary:")
    print(f"Total cells tracked: {df['track_id'].nunique()}")
    print(f"Cells that died: {len(df[df['state'] == 'dead']['track_id'].unique())}")
    
    # Death causes
    death_data = df[df['state'] == 'dead']
    if len(death_data) > 0:
        print("\nDeath signatures observed:")
        for feature in ['disappeared', 'blebbing', 'shrinkage', 'shape_change', 'texture_change']:
            count = len(death_data[death_data[feature] > 0.5])
            pct = count / len(death_data) * 100
            print(f"  {feature}: {count} cells ({pct:.1f}%)")


if __name__ == "__main__":
    # If running directly without command line args, use this test code:
    import sys
    
    if len(sys.argv) == 1:
        # Test mode - update these paths
        print("Running in test mode...")
        print("=" * 60)
        
        nd2_file = r"D:\New\BrainBites\Cell\2.nd2"  # UPDATE THIS PATH
        output_dir = "death_analysis_test"
        
        # Create detector
        detector = CellDeathDetector(nd2_file)
        
        # Load file
        if detector.load_file():
            # Run analysis
            results = detector.analyze_all_frames()
            
            # Save results
            df = detector.save_results(output_dir)
            
            # Create movie
            detector.create_movie(output_dir, fps=3)
            
            # Close
            detector.close()
            
            print("\nTest complete!")
            print(f"Check the {output_dir} folder for results")
        else:
            print("Failed to load test file!")
    else:
        # Run with command line arguments
        main()