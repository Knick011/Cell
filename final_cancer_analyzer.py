"""
Final Cancer Cell Analysis System V2
Uses aggregate size and shape for death detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches
from matplotlib.gridspec import GridSpec
import os
from datetime import datetime
import glob
from scipy.ndimage import label, binary_fill_holes
from skimage.measure import regionprops

from nk_cancer_analyzer import ND2Analyzer
from nk_cancer_analyzer_phase3 import DropletDetector, DropletCellAnalyzer
from nk_cancer_analyzer_phase4 import CellTracker

class AggregateCellTracker(CellTracker):
    """Tracker using aggregate size and shape for death detection."""
    
    def __init__(self, max_distance=30, max_frames_missing=3):
        super().__init__(max_distance, max_frames_missing)
        self.aggregate_history = {}  # Store aggregate properties over time
        self.shape_change_frames = {}  # Track non-spherical frames
        self.division_events = []  # Track cell divisions
        
    def update_with_aggregates(self, detections, frame_num, cell_aggregates):
        """Update tracks with aggregate information."""
        # First do normal tracking
        assignments = self.update(detections, frame_num, 'cancer')
        
        # Store aggregate data
        for i, det in enumerate(detections):
            if i in assignments and i < len(cell_aggregates):
                track_id = assignments[i]
                agg = cell_aggregates[i]
                
                if track_id not in self.aggregate_history:
                    self.aggregate_history[track_id] = []
                
                self.aggregate_history[track_id].append({
                    'frame': frame_num,
                    'area': agg['area'],
                    'eccentricity': agg['eccentricity'],
                    'solidity': agg['solidity'],
                    'major_axis': agg['major_axis_length'],
                    'minor_axis': agg['minor_axis_length']
                })
        
        # Check for cell divisions
        self._detect_divisions(assignments, frame_num)
        
        return assignments
    
    def _detect_divisions(self, assignments, frame_num):
        """Detect potential cell divisions."""
        # Look for new tracks appearing very close to existing ones
        new_tracks = []
        for i, track_id in assignments.items():
            track = self.tracks[track_id]
            if track['first_frame'] == frame_num:
                new_tracks.append((track_id, track['positions'][0]))
        
        # Check if new tracks are near existing tracks
        for new_id, (new_x, new_y) in new_tracks:
            for tid, track in self.tracks.items():
                if tid != new_id and track['status'] == 'active' and track['last_seen'] == frame_num:
                    last_x, last_y = track['positions'][-1]
                    dist = np.sqrt((new_x - last_x)**2 + (new_y - last_y)**2)
                    
                    # If very close, might be division
                    if dist < 50:  # pixels
                        self.division_events.append({
                            'parent_id': tid,
                            'child_id': new_id,
                            'frame': frame_num,
                            'distance': dist
                        })
                        # Mark in track
                        self.tracks[new_id]['parent'] = tid
                        if 'children' not in self.tracks[tid]:
                            self.tracks[tid]['children'] = []
                        self.tracks[tid]['children'].append(new_id)
    
    def mark_dead_cells_aggregate(self, frame_num, dying_threshold=0.7, dead_threshold=0.4):
        """
        Death detection based on aggregate changes:
        - Aggregate drops by 30% -> dying
        - Aggregate drops by 60% -> dead
        - Non-spherical shape for 4+ frames -> dead
        """
        for tid, track in self.tracks.items():
            if track['type'] == 'cancer':
                
                # Handle missing cells
                if frame_num - track['last_seen'] > self.max_frames_missing:
                    if track['status'] == 'active':
                        track['status'] = 'dying'
                        track['first_dying_frame'] = track['last_seen']
                    elif track['status'] == 'dying':
                        track['status'] = 'dead'
                        track['death_frame'] = track['last_seen'] + self.max_frames_missing
                    continue
                
                # Check aggregate changes
                if tid in self.aggregate_history and len(self.aggregate_history[tid]) >= 3:
                    agg_history = self.aggregate_history[tid]
                    
                    # Get baseline aggregate size (from first few frames)
                    baseline_area = np.mean([a['area'] for a in agg_history[:5]])
                    current_area = agg_history[-1]['area']
                    
                    # Check shape (eccentricity > 0.7 means non-spherical)
                    current_eccentricity = agg_history[-1]['eccentricity']
                    is_non_spherical = current_eccentricity > 0.7
                    
                    # Track shape changes
                    if is_non_spherical:
                        if tid not in self.shape_change_frames:
                            self.shape_change_frames[tid] = []
                        self.shape_change_frames[tid].append(frame_num)
                    else:
                        # Reset if becomes spherical again
                        self.shape_change_frames[tid] = []
                    
                    # Check for sustained non-spherical shape
                    if tid in self.shape_change_frames:
                        consecutive_frames = len([f for f in self.shape_change_frames[tid] 
                                                if frame_num - f < 4])
                        if consecutive_frames >= 4:
                            if track['status'] != 'dead':
                                track['status'] = 'dead'
                                track['death_frame'] = frame_num
                                track['death_reason'] = 'shape_change'
                            continue
                    
                    # Check aggregate size changes
                    if track['status'] == 'active':
                        # Check if aggregate dropped by 30%
                        if current_area < baseline_area * dying_threshold:
                            track['status'] = 'dying'
                            track['first_dying_frame'] = frame_num
                            track['baseline_area'] = baseline_area
                    
                    elif track['status'] == 'dying':
                        # Check if aggregate dropped by 60%
                        baseline = track.get('baseline_area', baseline_area)
                        if current_area < baseline * dead_threshold:
                            track['status'] = 'dead'
                            track['death_frame'] = frame_num
                            track['death_reason'] = 'aggregate_loss'


class EnhancedCellAnalyzer(DropletCellAnalyzer):
    """Enhanced analyzer that extracts aggregate properties."""
    
    def detect_nuclei_with_aggregates(self, tritc_image, brightfield_image, mask, droplet):
        """Detect nuclei and return aggregate properties."""
        # Get nuclei positions
        nuclei = self._detect_nuclei(tritc_image, brightfield_image, mask, droplet)
        
        # For each nucleus, extract aggregate properties
        aggregates = []
        
        for nx, ny, intensity, area in nuclei:
            # Extract region around nucleus
            x, y = int(nx), int(ny)
            
            # Define region of interest
            roi_size = 30  # pixels
            x0 = max(0, x - roi_size)
            x1 = min(tritc_image.shape[1], x + roi_size)
            y0 = max(0, y - roi_size)
            y1 = min(tritc_image.shape[0], y + roi_size)
            
            if x1 > x0 and y1 > y0:
                roi = tritc_image[y0:y1, x0:x1]
                
                # Threshold to get aggregate
                if roi.max() > 0:
                    threshold = np.percentile(roi[roi > 0], 50) if np.any(roi > 0) else 0
                    binary = roi > threshold
                    binary = binary_fill_holes(binary)
                    
                    # Label and get properties
                    labeled, num = label(binary)
                    
                    # Find the region containing the center
                    center_x, center_y = roi_size, roi_size
                    if x < roi_size:
                        center_x = x
                    if y < roi_size:
                        center_y = y
                    
                    # Get properties
                    props = regionprops(labeled)
                    
                    # Find the region closest to center
                    best_prop = None
                    min_dist = float('inf')
                    
                    for prop in props:
                        cy, cx = prop.centroid
                        dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                        if dist < min_dist:
                            min_dist = dist
                            best_prop = prop
                    
                    if best_prop:
                        aggregates.append({
                            'area': best_prop.area,
                            'eccentricity': best_prop.eccentricity,
                            'solidity': best_prop.solidity,
                            'major_axis_length': best_prop.major_axis_length,
                            'minor_axis_length': best_prop.minor_axis_length
                        })
                    else:
                        # Default values if no aggregate found
                        aggregates.append({
                            'area': area,
                            'eccentricity': 0,
                            'solidity': 1,
                            'major_axis_length': np.sqrt(area),
                            'minor_axis_length': np.sqrt(area)
                        })
        
        return nuclei, aggregates


class FinalAnalyzerV2:
    """Final analyzer with aggregate-based death detection."""
    
    def __init__(self, nd2_file, time_interval_min=15):
        self.nd2_file = nd2_file
        self.time_interval = time_interval_min
        self.analyzer = None
        self.droplets = None
        self.masks = None
        self.tracker = None
        self.frame_data = []
        self.results = {}
        
    def analyze(self):
        """Run complete analysis."""
        print(f"Loading {self.nd2_file}...")
        
        # Load ND2 file
        self.analyzer = ND2Analyzer(self.nd2_file)
        if not self.analyzer.load_file():
            return False
        
        # Detect droplets
        droplet_detector = DropletDetector()
        bf_frame = self.analyzer.get_frame(0, 'brightfield')
        self.droplets = droplet_detector.detect_droplets(bf_frame)
        self.masks, _ = droplet_detector.create_droplet_masks(bf_frame.shape, self.droplets)
        
        print(f"Found {len(self.droplets)} droplets")
        
        # Initialize tracker
        self.tracker = AggregateCellTracker(max_distance=30, max_frames_missing=3)
        
        # Initialize results
        for droplet in self.droplets:
            self.results[droplet['id']] = {
                'initial_cells': 0,
                'final_alive': 0,
                'total_dead': 0,
                'death_times': [],
                'divisions': 0
            }
        
        # Analyze all frames
        enhanced_analyzer = EnhancedCellAnalyzer(droplet_detector, None)
        
        for t in range(self.analyzer.metadata['frames']):
            if t % 10 == 0:
                print(f"Processing frame {t}/{self.analyzer.metadata['frames']-1}")
            
            frame_result = self._process_frame(t, enhanced_analyzer)
            self.frame_data.append(frame_result)
        
        # Finalize results
        self._finalize_results()
        
        print("Analysis complete!")
        return True
    
    def _process_frame(self, timepoint, enhanced_analyzer):
        """Process single frame with aggregate analysis."""
        bf_frame = self.analyzer.get_frame(timepoint, 'brightfield')
        tritc_frame = self.analyzer.get_frame(timepoint, 'TRITC')
        
        frame_result = {
            'timepoint': timepoint,
            'time_min': timepoint * self.time_interval,
            'brightfield': bf_frame,
            'tritc': tritc_frame,
            'droplet_data': {}
        }
        
        all_cancer_cells = []
        all_aggregates = []
        
        # Process each droplet
        for droplet in self.droplets:
            did = droplet['id']
            mask = self.masks[did]
            
            # Extract masked regions
            masked_tritc = tritc_frame.copy()
            masked_tritc[~mask] = 0
            masked_bf = bf_frame.copy()
            masked_bf[~mask] = 0
            
            # Detect cells with aggregate properties
            nuclei, aggregates = enhanced_analyzer.detect_nuclei_with_aggregates(
                masked_tritc, masked_bf, mask, droplet
            )
            
            # Convert to tracking format
            cancer_cells = []
            for i, (nx, ny, intensity, area) in enumerate(nuclei):
                cell = {
                    'centroid_x': nx,
                    'centroid_y': ny,
                    'mean_intensity': intensity,
                    'area': area,
                    'droplet_id': did
                }
                cancer_cells.append(cell)
                all_cancer_cells.append(cell)
                
                if i < len(aggregates):
                    all_aggregates.append(aggregates[i])
            
            # Store initial count
            if timepoint == 0:
                self.results[did]['initial_cells'] = len(cancer_cells)
            
            frame_result['droplet_data'][did] = {
                'cells': cancer_cells,
                'mask': mask
            }
        
        # Update tracker with aggregates
        assignments = self.tracker.update_with_aggregates(all_cancer_cells, timepoint, all_aggregates)
        self.tracker.mark_dead_cells_aggregate(timepoint)
        
        # Assign track info back
        for i, cell in enumerate(all_cancer_cells):
            did = cell['droplet_id']
            if i in assignments:
                track_id = assignments[i]
                track = self.tracker.tracks[track_id]
                
                for j, dcell in enumerate(frame_result['droplet_data'][did]['cells']):
                    if (dcell['centroid_x'] == cell['centroid_x'] and 
                        dcell['centroid_y'] == cell['centroid_y']):
                        dcell['track_id'] = track_id
                        dcell['status'] = track['status']
                        
                        # Add division marker
                        if 'parent' in track:
                            dcell['is_division'] = True
                            dcell['parent_id'] = track['parent']
                        break
        
        # Calculate statistics
        for did in frame_result['droplet_data']:
            cells = frame_result['droplet_data'][did]['cells']
            
            alive = sum(1 for c in cells if c.get('status') == 'active')
            dying = sum(1 for c in cells if c.get('status') == 'dying')
            dead = 0
            
            # Count dead cells
            for tid, track in self.tracker.tracks.items():
                if track['status'] == 'dead' and track['positions']:
                    last_x, last_y = track['positions'][-1]
                    if self.masks[did][int(last_y), int(last_x)]:
                        dead += 1
            
            frame_result['droplet_data'][did]['stats'] = {
                'alive': alive,
                'dying': dying,
                'dead': dead,
                'total': alive + dying
            }
            
            # Update final count
            if timepoint == self.analyzer.metadata['frames'] - 1:
                self.results[did]['final_alive'] = alive
        
        return frame_result
    
    def _finalize_results(self):
        """Finalize results with death times and divisions."""
        # Process death events
        for tid, track in self.tracker.tracks.items():
            if track['status'] == 'dead' and track['death_frame'] is not None:
                # Find droplet
                last_x, last_y = track['positions'][-1]
                for did, mask in self.masks.items():
                    if mask[int(last_y), int(last_x)]:
                        death_time = track['death_frame'] * self.time_interval
                        self.results[did]['death_times'].append(death_time)
                        self.results[did]['total_dead'] += 1
                        break
        
        # Count divisions per droplet
        for event in self.tracker.division_events:
            parent_track = self.tracker.tracks[event['parent_id']]
            if parent_track['positions']:
                x, y = parent_track['positions'][0]
                for did, mask in self.masks.items():
                    if mask[int(y), int(x)]:
                        self.results[did]['divisions'] += 1
                        break
        
        # Sort death times
        for did in self.results:
            self.results[did]['death_times'].sort()
    
    def create_movie(self, output_path=None, fps=3):
        """Create movie with aggregate-based visualization."""
        if not self.frame_data:
            return
        
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(self.nd2_file))[0]
            output_path = f"{base_name}_analysis.mp4"
        
        print(f"Creating movie: {output_path}")
        
        # Calculate grid layout
        n_droplets = len(self.droplets)
        cols = int(np.ceil(np.sqrt(n_droplets * 1.5)))
        rows = int(np.ceil(n_droplets / cols))
        
        # Create figure
        fig = plt.figure(figsize=(cols * 4, rows * 4 + 1.5))
        gs = GridSpec(rows + 1, cols, figure=fig, height_ratios=[1]*rows + [0.2])
        
        # Setup droplet displays
        droplet_data = {}
        
        for i, droplet in enumerate(self.droplets):
            did = droplet['id']
            row = i // cols
            col = i % cols
            
            ax = fig.add_subplot(gs[row, col])
            
            # Get crop region
            mask = self.masks[did]
            y_coords, x_coords = np.where(mask)
            
            if len(y_coords) > 0:
                margin = 20
                y_min = max(0, y_coords.min() - margin)
                y_max = min(self.frame_data[0]['tritc'].shape[0], y_coords.max() + margin)
                x_min = max(0, x_coords.min() - margin)
                x_max = min(self.frame_data[0]['tritc'].shape[1], x_coords.max() + margin)
                
                droplet_data[did] = {
                    'ax': ax,
                    'crop': (x_min, y_min, x_max, y_max),
                    'center': (droplet['center_x'] - x_min, droplet['center_y'] - y_min),
                    'radius': droplet['radius_px'],
                    'markers': []
                }
                
                # Initial display
                ax.set_title(f"Droplet {did}", fontsize=10)
                ax.axis('off')
                
                # Initialize image and info
                frame0 = self.frame_data[0]
                tritc_crop = frame0['tritc'][y_min:y_max, x_min:x_max]
                
                img = ax.imshow(tritc_crop, cmap='hot')
                droplet_data[did]['img'] = img
                
                # Draw circle
                circle = plt.Circle(droplet_data[did]['center'], 
                                  droplet_data[did]['radius'],
                                  color='lime', fill=False, linewidth=1.5)
                ax.add_patch(circle)
                
                # Info text
                info = ax.text(0.5, -0.1, '', transform=ax.transAxes,
                             ha='center', va='top', fontsize=8)
                droplet_data[did]['info'] = info
        
        # Global info
        ax_info = fig.add_subplot(gs[-1, :])
        ax_info.axis('off')
        global_info = ax_info.text(0.5, 0.5, '', transform=ax_info.transAxes,
                                  ha='center', va='center', fontsize=12,
                                  bbox=dict(boxstyle="round,pad=0.5", 
                                          facecolor='lightgray', alpha=0.8))
        
        def update_frame(frame_idx):
            frame = self.frame_data[frame_idx]
            time_min = frame['time_min']
            
            total_alive = 0
            total_dying = 0
            total_dead = 0
            total_divisions = sum(r['divisions'] for r in self.results.values())
            
            for did in droplet_data:
                if did not in frame['droplet_data']:
                    continue
                
                # Update image
                x_min, y_min, x_max, y_max = droplet_data[did]['crop']
                tritc_crop = frame['tritc'][y_min:y_max, x_min:x_max]
                
                # Enhance
                if tritc_crop.max() > tritc_crop.min():
                    vmin = np.percentile(tritc_crop[tritc_crop > 0], 5) if np.any(tritc_crop > 0) else 0
                    vmax = np.percentile(tritc_crop, 99)
                    enhanced = np.clip((tritc_crop - vmin) / (vmax - vmin + 1e-8), 0, 1)
                else:
                    enhanced = tritc_crop
                
                droplet_data[did]['img'].set_data(enhanced)
                
                # Clear old markers
                for m in droplet_data[did]['markers']:
                    m.remove()
                droplet_data[did]['markers'].clear()
                
                # Add cell markers
                cells = frame['droplet_data'][did]['cells']
                stats = frame['droplet_data'][did]['stats']
                
                for cell in cells:
                    if 'status' in cell:
                        cx = cell['centroid_x'] - x_min
                        cy = cell['centroid_y'] - y_min
                        
                        if cell['status'] == 'active':
                            color = 'lime' if cell.get('is_division') else 'white'
                            m = droplet_data[did]['ax'].plot(cx, cy, '+', 
                                                            color=color, 
                                                            markersize=8 if cell.get('is_division') else 6,
                                                            markeredgewidth=2 if cell.get('is_division') else 1.5)[0]
                            droplet_data[did]['markers'].append(m)
                            
                            # Mark divisions
                            if cell.get('is_division'):
                                t = droplet_data[did]['ax'].text(cx + 10, cy, 'D', 
                                                                color='lime', 
                                                                fontsize=8,
                                                                fontweight='bold')
                                droplet_data[did]['markers'].append(t)
                                
                        elif cell['status'] == 'dying':
                            m = droplet_data[did]['ax'].plot(cx, cy, 'x', 
                                                            color='yellow', 
                                                            markersize=7,
                                                            markeredgewidth=2)[0]
                            droplet_data[did]['markers'].append(m)
                            
                            # Show aggregate loss
                            if 'track_id' in cell:
                                track = self.tracker.tracks[cell['track_id']]
                                if cell['track_id'] in self.tracker.aggregate_history:
                                    agg_hist = self.tracker.aggregate_history[cell['track_id']]
                                    if len(agg_hist) > 0:
                                        baseline = agg_hist[0]['area']
                                        current = agg_hist[-1]['area']
                                        loss_pct = (1 - current/baseline) * 100
                                        t = droplet_data[did]['ax'].text(cx, cy - 10, 
                                                                        f'-{loss_pct:.0f}%',
                                                                        color='yellow', 
                                                                        fontsize=6,
                                                                        ha='center')
                                        droplet_data[did]['markers'].append(t)
                
                # Update info
                info_text = f"A:{stats['alive']} D:{stats['dying']} †:{stats['dead']}"
                droplet_data[did]['info'].set_text(info_text)
                
                # Color coding
                if stats['alive'] == 0 and stats['total'] == 0:
                    droplet_data[did]['info'].set_color('red')
                elif stats['dying'] > 0:
                    droplet_data[did]['info'].set_color('orange')
                else:
                    droplet_data[did]['info'].set_color('green')
                
                total_alive += stats['alive']
                total_dying += stats['dying']
                total_dead += stats['dead']
            
            # Update global info
            initial_total = sum(self.results[d]['initial_cells'] for d in self.results)
            survival_rate = (total_alive / initial_total * 100) if initial_total > 0 else 0
            
            global_text = f"Time: {time_min:.0f} min | "
            global_text += f"Alive: {total_alive} | "
            global_text += f"Dying: {total_dying} | "
            global_text += f"Dead: {total_dead} | "
            global_text += f"Divisions: {total_divisions} | "
            global_text += f"Survival: {survival_rate:.1f}%"
            
            global_info.set_text(global_text)
            
            fig.suptitle(f'Cancer Cell Analysis - {os.path.basename(self.nd2_file)} - T = {time_min:.0f} min', 
                        fontsize=14, y=0.98)
            
            return list(droplet_data[d]['img'] for d in droplet_data) + \
                   list(droplet_data[d]['info'] for d in droplet_data) + \
                   [m for d in droplet_data for m in droplet_data[d]['markers']] + \
                   [global_info]
        
        # Create animation
        anim = animation.FuncAnimation(fig, update_frame, frames=len(self.frame_data),
                                     interval=1000/fps, blit=False)
        
        # Save - MP4 only
        ffmpeg_path = r"D:\ffmpeg-2025-06-28-git-cfd1f81e7d-full_build\bin\ffmpeg.exe"
        if not os.path.exists(ffmpeg_path):
            raise FileNotFoundError(f"FFmpeg not found at {ffmpeg_path}")
        
        # Temporarily add FFmpeg to PATH
        original_path = os.environ.get('PATH', '')
        os.environ['PATH'] = os.path.dirname(ffmpeg_path) + os.pathsep + original_path
        
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
            anim.save(output_path, writer=writer, dpi=150)
            print(f"MP4 movie saved: {output_path}")
        except Exception as e:
            print(f"FFMpegWriter failed: {e}")
            print("Please ensure FFmpeg is properly installed and accessible.")
            raise
        finally:
            # Restore original PATH
            os.environ['PATH'] = original_path
        plt.close()
        
        print(f"Movie saved: {output_path}")
    
    def export_results(self):
        """Export results in requested format."""
        base_name = os.path.splitext(os.path.basename(self.nd2_file))[0]
        
        excel_data = []
        
        for droplet in self.droplets:
            did = droplet['id']
            result = self.results[did]
            
            row = {
                'ND2 Name': base_name,
                'Droplet Number': did,
                'Number of cancer cells at the start': result['initial_cells'],
                'Number of cancer cells dead': result['total_dead'],
                'Number of cancer cells alive at the end': result['final_alive'],
                'Number of cell divisions': result['divisions']
            }
            
            # Add death times
            for i, death_time in enumerate(result['death_times']):
                col_name = f'Time of death for cell {i+1}'
                row[col_name] = death_time
            
            excel_data.append(row)
        
        return pd.DataFrame(excel_data)


def analyze_single_file(nd2_file, output_dir=None):
    """Analyze a single ND2 file."""
    if output_dir is None:
        output_dir = os.path.dirname(nd2_file)
    
    analyzer = FinalAnalyzerV2(nd2_file)
    
    if analyzer.analyze():
        # Create movie
        movie_path = os.path.join(output_dir, 
                                os.path.splitext(os.path.basename(nd2_file))[0] + "_analysis.mp4")
        analyzer.create_movie(movie_path)
        
        # Get results
        results_df = analyzer.export_results()
        
        # Close
        analyzer.analyzer.close()
        
        return results_df
    
    return None


def batch_analyze(directory_path, output_file='cancer_analysis_results.xlsx'):
    """Analyze all ND2 files and create combined Excel."""
    nd2_files = glob.glob(os.path.join(directory_path, "*.nd2"))
    
    if not nd2_files:
        print(f"No ND2 files found in {directory_path}")
        return
    
    print(f"Found {len(nd2_files)} ND2 files to process")
    
    all_results = []
    
    for i, nd2_file in enumerate(nd2_files):
        print(f"\n{'='*60}")
        print(f"Processing file {i+1}/{len(nd2_files)}: {os.path.basename(nd2_file)}")
        print(f"{'='*60}")
        
        try:
            results = analyze_single_file(nd2_file, directory_path)
            if results is not None:
                all_results.append(results)
                print(f"✓ Successfully processed {os.path.basename(nd2_file)}")
        except Exception as e:
            print(f"✗ Error processing {os.path.basename(nd2_file)}: {str(e)}")
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save to Excel
        excel_path = os.path.join(directory_path, output_file)
        combined_df.to_excel(excel_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"All results saved to: {excel_path}")
        print(f"Processed {len(all_results)} files successfully")
    
    return combined_df


if __name__ == "__main__":
    # Process all files in directory
    directory = r"D:\New\BrainBites\Cell"
    batch_analyze(directory)