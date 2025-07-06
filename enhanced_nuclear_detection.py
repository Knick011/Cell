"""
Enhanced Cancer Cell Analysis with Peak-Based Nuclear Detection and Persistent Tracking
Improves cell separation and maintains consistent cell counts
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
from scipy.ndimage import maximum_filter, label, center_of_mass
from skimage import morphology, measure, filters
from skimage.feature import peak_local_max, graycomatrix, graycoprops
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment

from nk_cancer_analyzer import ND2Analyzer
from nk_cancer_analyzer_phase3 import DropletDetector, DropletCellAnalyzer


class EnhancedNuclearDetector:
    """Detect individual nuclei using peak detection instead of blob detection."""
    
    def __init__(self, min_peak_distance=15, min_peak_intensity_percentile=85):
        """
        Args:
            min_peak_distance: Minimum distance between nuclear centers (pixels)
            min_peak_intensity_percentile: Minimum intensity percentile for valid peaks
        """
        self.min_peak_distance = min_peak_distance
        self.min_peak_intensity_percentile = min_peak_intensity_percentile
    
    def detect_nuclear_centers(self, tritc_image, mask=None):
        """
        Detect individual nuclear centers using peak detection.
        
        Args:
            tritc_image: TRITC channel image
            mask: Optional mask to restrict detection area
            
        Returns:
            List of (x, y, intensity) tuples for each detected nucleus
        """
        # Apply mask if provided
        if mask is not None:
            tritc_masked = tritc_image.copy()
            tritc_masked[~mask] = 0
        else:
            tritc_masked = tritc_image
        
        # Skip if no signal
        if np.max(tritc_masked) == 0:
            return []
        
        # Step 1: Simple denoising with Gaussian blur (faster than median)
        denoised = cv2.GaussianBlur(tritc_masked.astype(np.float32), (3, 3), 1)
        
        # Step 2: Background subtraction with smaller kernel
        background = cv2.GaussianBlur(denoised, (21, 21), 10)
        signal = np.maximum(0, denoised - background * 0.8)
        
        # Step 3: Find intensity threshold
        # Only consider pixels within mask for threshold calculation
        if mask is not None:
            valid_pixels = signal[mask]
        else:
            valid_pixels = signal[signal > 0]
        
        if len(valid_pixels) == 0:
            return []
        
        # Use percentile-based threshold
        intensity_threshold = np.percentile(valid_pixels, self.min_peak_intensity_percentile)
        
        # Step 4: Find local maxima (peaks) - use scipy's maximum_filter for efficiency
        # Create a local maximum filter
        local_max = maximum_filter(signal, size=self.min_peak_distance)
        
        # Find peaks where signal equals local maximum and above threshold
        peak_mask = (signal == local_max) & (signal >= intensity_threshold)
        
        # Get coordinates of peaks
        coordinates = np.where(peak_mask)
        y_coords, x_coords = coordinates
        
        # Step 5: Extract peak properties
        nuclei = []
        for y, x in zip(y_coords, x_coords):
            # Get peak intensity
            intensity = signal[y, x]
            
            # Calculate local region properties
            # Create small region around peak
            region_size = 8  # Reduced from 10
            y0 = max(0, y - region_size)
            y1 = min(signal.shape[0], y + region_size)
            x0 = max(0, x - region_size)
            x1 = min(signal.shape[1], x + region_size)
            
            local_region = signal[y0:y1, x0:x1]
            
            # Calculate integrated intensity (more robust than single pixel)
            if local_region.size > 0:
                integrated_intensity = np.sum(local_region)
            else:
                integrated_intensity = intensity
            
            nuclei.append({
                'x': x,
                'y': y,
                'peak_intensity': intensity,
                'integrated_intensity': integrated_intensity,
                'confidence': self._calculate_peak_confidence(signal, x, y)
            })
        
        # Sort by intensity (strongest peaks first)
        nuclei.sort(key=lambda n: n['integrated_intensity'], reverse=True)
        
        return nuclei
    
    def _calculate_peak_confidence(self, image, x, y):
        """Calculate confidence that this is a real nuclear peak."""
        # Check if peak is prominent compared to surroundings
        region_size = 5
        y0 = max(0, y - region_size)
        y1 = min(image.shape[0], y + region_size)
        x0 = max(0, x - region_size)
        x1 = min(image.shape[1], x + region_size)
        
        local_region = image[y0:y1, x0:x1]
        if local_region.size == 0:
            return 0
        
        peak_val = image[y, x]
        mean_val = np.mean(local_region)
        
        # Confidence based on peak prominence
        if mean_val > 0:
            confidence = peak_val / mean_val
        else:
            confidence = 1.0 if peak_val > 0 else 0
        
        return min(confidence, 2.0)  # Cap at 2.0


class PersistentCellTracker:
    """Enhanced tracker that maintains cell identities persistently."""
    
    def __init__(self, max_distance=30, max_frames_missing=5):
        """
        Args:
            max_distance: Maximum distance to link cells between frames
            max_frames_missing: Frames before considering a cell truly gone
        """
        self.max_distance = max_distance
        self.max_frames_missing = max_frames_missing
        self.tracks = {}
        self.next_id = 1
        self.death_states = {}  # Permanent death registry
        
    def update(self, detections, frame_num, brightfield_states=None):
        """
        Update tracks with new detections, maintaining persistent IDs.
        
        Args:
            detections: List of detected nuclei
            frame_num: Current frame number
            brightfield_states: Dict of position -> bf_state
            
        Returns:
            Dict of detection_idx -> (track_id, status)
        """
        # Find candidate tracks (including missing ones)
        candidate_tracks = {}
        for tid, track in self.tracks.items():
            # Include tracks that have been missing for a few frames
            if frame_num - track['last_seen'] <= self.max_frames_missing:
                # Predict position based on history
                if len(track['positions']) >= 2:
                    # Simple linear prediction
                    dx = track['positions'][-1][0] - track['positions'][-2][0]
                    dy = track['positions'][-1][1] - track['positions'][-2][1]
                    predicted_x = track['positions'][-1][0] + dx * (frame_num - track['last_seen'])
                    predicted_y = track['positions'][-1][1] + dy * (frame_num - track['last_seen'])
                else:
                    predicted_x = track['positions'][-1][0]
                    predicted_y = track['positions'][-1][1]
                
                candidate_tracks[tid] = {
                    'track': track,
                    'predicted_pos': (predicted_x, predicted_y),
                    'frames_missing': frame_num - track['last_seen']
                }
        
        # Build cost matrix
        assignments = {}
        used_tracks = set()
        used_detections = set()
        
        if candidate_tracks and detections:
            # Compute all distances
            distances = []
            for i, det in enumerate(detections):
                for tid, cand in candidate_tracks.items():
                    pred_x, pred_y = cand['predicted_pos']
                    dist = np.sqrt((det['x'] - pred_x)**2 + (det['y'] - pred_y)**2)
                    
                    # Increase cost for tracks that have been missing
                    cost = dist * (1 + 0.1 * cand['frames_missing'])
                    
                    distances.append((cost, i, tid))
            
            # Sort by cost and assign greedily
            distances.sort()
            
            for cost, det_idx, track_id in distances:
                if cost < self.max_distance and det_idx not in used_detections and track_id not in used_tracks:
                    assignments[det_idx] = track_id
                    used_tracks.add(track_id)
                    used_detections.add(det_idx)
        
        # Update matched tracks
        for det_idx, track_id in assignments.items():
            det = detections[det_idx]
            track = self.tracks[track_id]
            
            # Update position
            track['positions'].append((det['x'], det['y']))
            track['intensities'].append(det['integrated_intensity'])
            track['last_seen'] = frame_num
            track['consecutively_missing'] = 0
            
            # Update state based on brightfield if available
            if brightfield_states and (det['x'], det['y']) in brightfield_states:
                bf_state = brightfield_states[(det['x'], det['y'])]
                
                # Once dead, always dead
                if track_id in self.death_states:
                    track['state'] = 'dead'
                elif bf_state == 'dead':
                    track['state'] = 'dead'
                    self.death_states[track_id] = frame_num  # Record death
                else:
                    track['state'] = bf_state
            elif track_id in self.death_states:
                track['state'] = 'dead'
        
        # Create new tracks for unassigned detections
        for i, det in enumerate(detections):
            if i not in used_detections:
                track_id = self.next_id
                self.next_id += 1
                
                # Determine initial state
                state = 'alive'
                if brightfield_states and (det['x'], det['y']) in brightfield_states:
                    state = brightfield_states[(det['x'], det['y'])]
                    if state == 'dead':
                        self.death_states[track_id] = frame_num
                
                self.tracks[track_id] = {
                    'id': track_id,
                    'first_frame': frame_num,
                    'last_seen': frame_num,
                    'positions': [(det['x'], det['y'])],
                    'intensities': [det['integrated_intensity']],
                    'state': state,
                    'consecutively_missing': 0
                }
                
                assignments[i] = track_id
        
        # Update missing tracks
        for tid, track in self.tracks.items():
            if tid not in used_tracks and track['last_seen'] < frame_num:
                track['consecutively_missing'] += 1
        
        # Return assignments with states
        result = {}
        for det_idx, track_id in assignments.items():
            track = self.tracks[track_id]
            result[det_idx] = (track_id, track['state'])
        
        return result
    
    def get_active_tracks(self, frame_num):
        """Get all tracks that should be displayed (including missing and dead)."""
        active = []
        
        for tid, track in self.tracks.items():
            # Include if:
            # 1. Recently seen (could be temporarily missing)
            # 2. Dead but was present in this droplet
            if (frame_num - track['last_seen'] <= self.max_frames_missing or 
                track['state'] == 'dead'):
                
                # Get last known position
                if track['positions']:
                    x, y = track['positions'][-1]
                    active.append({
                        'track_id': tid,
                        'x': x,
                        'y': y,
                        'state': track['state'],
                        'last_seen': track['last_seen'],
                        'missing_frames': frame_num - track['last_seen']
                    })
        
        return active


class ImprovedCancerAnalyzer:
    """Complete analyzer with improved nuclear detection and persistent tracking."""
    
    def __init__(self, nd2_file, time_interval_min=15):
        self.nd2_file = nd2_file
        self.time_interval = time_interval_min
        self.analyzer = None
        self.nuclear_detector = EnhancedNuclearDetector(
            min_peak_distance=15,  # Adjust based on your cell size
            min_peak_intensity_percentile=85
        )
        self.death_detector = None  # Will use the BrightfieldDeathDetector
        self.droplet_trackers = {}  # Separate tracker per droplet
        self.droplets = None
        self.masks = None
        self.frame_cache = []
        self.results = []
    
    def analyze(self, output_dir=None):
        """Run complete analysis."""
        print(f"Loading {self.nd2_file}...")
        
        # Import brightfield death detector
        from brightfield_death_analyzer import BrightfieldDeathDetector
        self.death_detector = BrightfieldDeathDetector()
        
        # Load ND2 file
        self.analyzer = ND2Analyzer(self.nd2_file)
        if not self.analyzer.load_file():
            return False
        
        # Create output directory
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(self.nd2_file), 
                                    "enhanced_peak_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        # Detect droplets
        print("Detecting droplets...")
        droplet_detector = DropletDetector()
        bf_frame = self.analyzer.get_frame(0, 'brightfield')
        if bf_frame is None:
            print("Error: Could not load brightfield frame")
            return False
        self.droplets = droplet_detector.detect_droplets(bf_frame)
        self.masks, _ = droplet_detector.create_droplet_masks(bf_frame.shape, self.droplets)
        
        print(f"Found {len(self.droplets)} droplets")
        
        # Initialize trackers for each droplet
        for droplet in self.droplets:
            self.droplet_trackers[droplet['id']] = PersistentCellTracker(
                max_distance=30,
                max_frames_missing=5
            )
        
        # Process all frames
        previous_bf = None
        
        for t in range(self.analyzer.metadata['frames']):
            if t % 10 == 0:
                print(f"Processing frame {t}/{self.analyzer.metadata['frames']-1}")
            
            frame_result = self._process_frame(t, previous_bf)
            self.results.append(frame_result)
            
            # Store previous brightfield
            previous_bf = frame_result['brightfield']
        
        # Generate outputs
        print("Generating outputs...")
        self._save_results(output_dir)
        self._create_movie(output_dir)
        self._create_validation_plots(output_dir)
        
        # Close analyzer
        self.analyzer.close()
        
        print(f"Analysis complete! Results saved to: {output_dir}")
        return True
    
    def _process_frame(self, timepoint, previous_bf):
        """Process a single frame."""
        # Get frames
        if self.analyzer is None:
            return None
            
        bf_frame = self.analyzer.get_frame(timepoint, 'brightfield')
        tritc_frame = self.analyzer.get_frame(timepoint, 'TRITC')
        
        if bf_frame is None or tritc_frame is None:
            return None
        
        # Store for visualization
        self.frame_cache.append({
            'timepoint': timepoint,
            'brightfield': bf_frame,
            'tritc': tritc_frame
        })
        
        frame_result = {
            'timepoint': timepoint,
            'time_min': timepoint * self.time_interval,
            'brightfield': bf_frame,
            'tritc': tritc_frame,
            'droplet_results': {}
        }
        
        # Process each droplet
        if self.droplets is None or self.masks is None:
            return frame_result
            
        for droplet in self.droplets:
            did = droplet['id']
            mask = self.masks[did]
            
            # Step 1: Detect nuclear centers using peak detection
            nuclei = self.nuclear_detector.detect_nuclear_centers(tritc_frame, mask)
            
            # Step 2: Analyze brightfield morphology at each nuclear position
            brightfield_states = {}
            if self.death_detector is not None:
                for nucleus in nuclei:
                    death_result = self.death_detector.analyze_cell_morphology(
                        bf_frame,
                        tritc_frame,
                        (nucleus['x'], nucleus['y']),
                        cell_id=f"d{did}_t{timepoint}_n{nucleus['x']}_{nucleus['y']}",
                        previous_bf=previous_bf,
                        store_reference=(timepoint < 5)
                    )
                    
                    brightfield_states[(nucleus['x'], nucleus['y'])] = death_result['state']
                    nucleus['death_score'] = death_result['death_score']
                    nucleus['death_features'] = death_result['features']
            
            # Step 3: Update tracking with brightfield states
            tracker = self.droplet_trackers[did]
            assignments = tracker.update(nuclei, timepoint, brightfield_states)
            
            # Step 4: Get all tracks to display (including missing/dead)
            all_tracks = tracker.get_active_tracks(timepoint)
            
            # Step 5: Calculate statistics
            alive = sum(1 for t in all_tracks if t['state'] == 'alive' and t['missing_frames'] == 0)
            dying = sum(1 for t in all_tracks if t['state'] == 'dying' and t['missing_frames'] == 0)
            dead = sum(1 for t in all_tracks if t['state'] == 'dead')
            missing = sum(1 for t in all_tracks if t['state'] == 'alive' and t['missing_frames'] > 0)
            
            frame_result['droplet_results'][did] = {
                'detected_nuclei': nuclei,
                'assignments': assignments,
                'all_tracks': all_tracks,
                'alive': alive,
                'dying': dying,
                'dead': dead,
                'missing': missing,
                'total_tracked': len(all_tracks)
            }
        
        return frame_result
    
    def _create_movie(self, output_dir, fps=3):
        """Create movie showing peak detection and death tracking."""
        base_name = os.path.splitext(os.path.basename(self.nd2_file))[0]
        movie_path = os.path.join(output_dir, f"{base_name}_peak_tracking_movie.mp4")
        
        print(f"Creating movie: {movie_path}")
        
        # Setup figure
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, height_ratios=[2, 2, 1])
        
        # Main display
        ax_main = fig.add_subplot(gs[:2, :2])
        ax_zoom = fig.add_subplot(gs[0, 2])
        ax_tritc_zoom = fig.add_subplot(gs[1, 2])
        ax_counts = fig.add_subplot(gs[2, :])
        
        # Initialize
        frame0 = self.results[0]
        
        # Main view - brightfield with overlay
        ax_main.imshow(frame0['brightfield'], cmap='gray')
        ax_main.set_title('Cell Tracking Overview', fontsize=14)
        ax_main.axis('off')
        
        # Draw droplets
        if self.droplets is not None:
            for droplet in self.droplets:
                circle = patches.Circle((droplet['center_x'], droplet['center_y']), 
                                      droplet['radius_px'], 
                                      color='lime', fill=False, linewidth=2)
                ax_main.add_patch(circle)
        
        # Zoom windows
        ax_zoom.set_title('Brightfield Zoom', fontsize=12)
        ax_zoom.axis('off')
        ax_tritc_zoom.set_title('TRITC Peaks Zoom', fontsize=12)
        ax_tritc_zoom.axis('off')
        
        # Cell count plot
        ax_counts.set_xlabel('Time (minutes)')
        ax_counts.set_ylabel('Cell Count')
        ax_counts.set_title('Cell States Over Time')
        ax_counts.grid(True, alpha=0.3)
        
        # Storage for plot elements
        cell_markers = []
        count_lines = {
            'alive': None,
            'dying': None,
            'dead': None,
            'total': None
        }
        
        # Time series data
        times = []
        counts = {'alive': [], 'dying': [], 'dead': [], 'total': []}
        
        def update_frame(frame_idx):
            frame = self.results[frame_idx]
            
            # Clear old markers
            for marker in cell_markers:
                marker.remove()
            cell_markers.clear()
            
            # Update main view
            ax_main.imshow(frame['brightfield'], cmap='gray')
            ax_main.set_title(f'Cell Tracking - T = {frame["time_min"]:.0f} min', fontsize=14)
            
            # Collect all cells for zoom
            all_cells = []
            
            # Add cell markers
            for did, data in frame['droplet_results'].items():
                # Draw all tracked cells
                for track in data['all_tracks']:
                    x, y = track['x'], track['y']
                    state = track['state']
                    track_id = track['track_id']
                    
                    # Choose marker based on state
                    if state == 'alive':
                        if track['missing_frames'] == 0:
                            marker = ax_main.plot(x, y, 'o', color='lime', 
                                                markersize=8, markeredgecolor='white',
                                                markeredgewidth=1)[0]
                        else:
                            # Missing cell - show predicted position
                            marker = ax_main.plot(x, y, 'o', color='lime', 
                                                markersize=8, markeredgecolor='yellow',
                                                markeredgewidth=2, alpha=0.5)[0]
                    elif state == 'dying':
                        marker = ax_main.plot(x, y, 's', color='yellow', 
                                            markersize=10, markeredgecolor='black',
                                            markeredgewidth=1)[0]
                    else:  # dead
                        marker = ax_main.plot(x, y, 'x', color='red', 
                                            markersize=12, markeredgewidth=3)[0]
                    
                    cell_markers.append(marker)
                    
                    # Add track ID
                    text = ax_main.text(x+5, y-5, str(track_id), 
                                       color='white', fontsize=8,
                                       bbox=dict(boxstyle="round,pad=0.2", 
                                               facecolor='black', alpha=0.5))
                    cell_markers.append(text)
                    
                    all_cells.append(track)
            
            # Update zoom on first droplet with cells
            if self.droplets is not None:
                for did, data in frame['droplet_results'].items():
                    if data['detected_nuclei']:
                        droplet = next(d for d in self.droplets if d['id'] == did)
                        
                        # Define zoom region
                        margin = 50
                        x0 = max(0, droplet['center_x'] - margin)
                        x1 = min(frame['brightfield'].shape[1], droplet['center_x'] + margin)
                        y0 = max(0, droplet['center_y'] - margin)
                        y1 = min(frame['brightfield'].shape[0], droplet['center_y'] + margin)
                        
                        # Show brightfield zoom
                        bf_zoom = frame['brightfield'][y0:y1, x0:x1]
                        ax_zoom.clear()
                        ax_zoom.imshow(bf_zoom, cmap='gray')
                        ax_zoom.set_title(f'Droplet {did} - Brightfield', fontsize=12)
                        ax_zoom.axis('off')
                        
                        # Show TRITC with peaks
                        tritc_zoom = frame['tritc'][y0:y1, x0:x1]
                        ax_tritc_zoom.clear()
                        ax_tritc_zoom.imshow(tritc_zoom, cmap='hot')
                        
                        # Mark detected peaks
                        for nucleus in data['detected_nuclei']:
                            nx = nucleus['x'] - x0
                            ny = nucleus['y'] - y0
                            ax_tritc_zoom.plot(nx, ny, '+', color='white', 
                                             markersize=12, markeredgewidth=2)
                        
                        ax_tritc_zoom.set_title(f'Droplet {did} - Nuclear Peaks', fontsize=12)
                        ax_tritc_zoom.axis('off')
                        
                        break
            
            # Update counts
            times.append(frame['time_min'])
            total_alive = sum(d['alive'] for d in frame['droplet_results'].values())
            total_dying = sum(d['dying'] for d in frame['droplet_results'].values())
            total_dead = sum(d['dead'] for d in frame['droplet_results'].values())
            total_all = sum(d['total_tracked'] for d in frame['droplet_results'].values())
            
            counts['alive'].append(total_alive)
            counts['dying'].append(total_dying)
            counts['dead'].append(total_dead)
            counts['total'].append(total_all)
            
            # Update count plot
            if count_lines['alive'] is None:
                count_lines['alive'], = ax_counts.plot(times, counts['alive'], 
                                                      'g-', label='Alive', linewidth=2)
                count_lines['dying'], = ax_counts.plot(times, counts['dying'], 
                                                      'y-', label='Dying', linewidth=2)
                count_lines['dead'], = ax_counts.plot(times, counts['dead'], 
                                                     'r-', label='Dead', linewidth=2)
                count_lines['total'], = ax_counts.plot(times, counts['total'], 
                                                      'b--', label='Total Tracked', linewidth=2)
                ax_counts.legend()
            else:
                if count_lines['alive'] is not None:
                    count_lines['alive'].set_data(times, counts['alive'])
                if count_lines['dying'] is not None:
                    count_lines['dying'].set_data(times, counts['dying'])
                if count_lines['dead'] is not None:
                    count_lines['dead'].set_data(times, counts['dead'])
                if count_lines['total'] is not None:
                    count_lines['total'].set_data(times, counts['total'])
            
            # Add info text
            info_text = f"Total Tracked: {total_all} | "
            info_text += f"Alive: {total_alive} | "
            info_text += f"Dying: {total_dying} | "
            info_text += f"Dead: {total_dead}"
            
            # Count missing
            total_missing = sum(d['missing'] for d in frame['droplet_results'].values())
            if total_missing > 0:
                info_text += f" | Missing: {total_missing}"
            
            fig.suptitle(info_text, fontsize=16)
            
            return cell_markers + list(count_lines.values())
        
        # Create animation
        anim = animation.FuncAnimation(fig, update_frame, frames=len(self.results),
                                     interval=1000/fps, blit=False)
        
        # Save with FFmpeg
        self._save_animation(anim, movie_path, fps)
        plt.close()
        
        print(f"Movie saved: {movie_path}")
    
    def _save_animation(self, anim, output_path, fps):
        """Save animation handling FFmpeg path."""
        ffmpeg_path = r"D:\ffmpeg-2025-06-28-git-cfd1f81e7d-full_build\bin\ffmpeg.exe"
        
        if os.path.exists(ffmpeg_path):
            original_path = os.environ.get('PATH', '')
            os.environ['PATH'] = os.path.dirname(ffmpeg_path) + os.pathsep + original_path
            
            try:
                writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
                anim.save(output_path, writer=writer)
            finally:
                os.environ['PATH'] = original_path
        else:
            try:
                writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
                anim.save(output_path, writer=writer)
            except:
                print("Warning: Could not save movie. FFmpeg not found.")
    
    def _save_results(self, output_dir):
        """Save analysis results to Excel."""
        base_name = os.path.splitext(os.path.basename(self.nd2_file))[0]
        excel_path = os.path.join(output_dir, f"{base_name}_peak_tracking_results.xlsx")
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Time series data
            rows = []
            for frame in self.results:
                for did, data in frame['droplet_results'].items():
                    rows.append({
                        'timepoint': frame['timepoint'],
                        'time_min': frame['time_min'],
                        'droplet_id': did,
                        'detected_peaks': len(data['detected_nuclei']),
                        'alive': data['alive'],
                        'dying': data['dying'],
                        'dead': data['dead'],
                        'missing': data['missing'],
                        'total_tracked': data['total_tracked']
                    })
            
            df_timeseries = pd.DataFrame(rows)
            df_timeseries.to_excel(writer, sheet_name='Time_Series', index=False)
            
            # Track summary
            track_rows = []
            for did, tracker in self.droplet_trackers.items():
                for tid, track in tracker.tracks.items():
                    track_rows.append({
                        'track_id': tid,
                        'droplet_id': did,
                        'first_frame': track['first_frame'],
                        'last_seen': track['last_seen'],
                        'final_state': track['state'],
                        'death_frame': tracker.death_states.get(tid, 'N/A'),
                        'lifespan_frames': track['last_seen'] - track['first_frame'] + 1,
                        'max_intensity': max(track['intensities']) if track['intensities'] else 0
                    })
            
            if track_rows:
                df_tracks = pd.DataFrame(track_rows)
                df_tracks.to_excel(writer, sheet_name='Cell_Tracks', index=False)
            
            # Summary by droplet
            summary_rows = []
            if self.masks is not None:
                for did in self.masks.keys():
                    # Get initial and final counts
                    initial_data = next((r for r in rows if r['droplet_id'] == did and r['timepoint'] == 0), {})
                    final_data = next((r for r in rows if r['droplet_id'] == did and 
                                     r['timepoint'] == self.results[-1]['timepoint']), {})
                    
                    # Count total deaths
                    tracker = self.droplet_trackers[did]
                    total_deaths = len(tracker.death_states)
                    
                    summary_rows.append({
                        'droplet_id': did,
                        'initial_cells': initial_data.get('total_tracked', 0),
                        'final_alive': final_data.get('alive', 0),
                        'total_deaths': total_deaths,
                        'max_cells_tracked': max((r['total_tracked'] for r in rows if r['droplet_id'] == did), default=0),
                        'survival_rate_%': (final_data.get('alive', 0) / initial_data.get('total_tracked', 1) * 100) 
                                         if initial_data.get('total_tracked', 0) > 0 else 0
                    })
            
            df_summary = pd.DataFrame(summary_rows)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"Results saved to: {excel_path}")
    
    def _create_validation_plots(self, output_dir):
        """Create plots to validate peak detection and tracking."""
        base_name = os.path.splitext(os.path.basename(self.nd2_file))[0]
        
        # Plot 1: Peak detection validation
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Nuclear Peak Detection Validation', fontsize=16)
        
        # Show examples from different time points
        time_indices = [0, len(self.results)//4, len(self.results)//2, 
                       3*len(self.results)//4, len(self.results)-1]
        
        for idx, t_idx in enumerate(time_indices[:6]):
            if t_idx >= len(self.results):
                continue
                
            ax = axes[idx//3, idx%3]
            frame = self.results[t_idx]
            
            # Find a droplet with cells
            for did, data in frame['droplet_results'].items():
                if data['detected_nuclei']:
                    droplet = next(d for d in self.droplets if d['id'] == did)
                    
                    # Extract region
                    margin = 60
                    x0 = max(0, droplet['center_x'] - margin)
                    x1 = min(frame['tritc'].shape[1], droplet['center_x'] + margin)
                    y0 = max(0, droplet['center_y'] - margin)
                    y1 = min(frame['tritc'].shape[0], droplet['center_y'] + margin)
                    
                    tritc_crop = frame['tritc'][y0:y1, x0:x1]
                    
                    # Enhance for visualization
                    if np.any(tritc_crop > 0):
                        vmin, vmax = np.percentile(tritc_crop[tritc_crop > 0], [5, 99])
                        tritc_enhanced = np.clip((tritc_crop - vmin) / (vmax - vmin + 1e-8), 0, 1)
                    else:
                        tritc_enhanced = tritc_crop
                    
                    ax.imshow(tritc_enhanced, cmap='hot')
                    
                    # Mark detected peaks
                    for nucleus in data['detected_nuclei']:
                        nx = nucleus['x'] - x0
                        ny = nucleus['y'] - y0
                        ax.plot(nx, ny, '+', color='white', markersize=12, markeredgewidth=2)
                        
                        # Show confidence
                        conf = nucleus.get('confidence', 1.0)
                        ax.text(nx+5, ny-5, f"{conf:.1f}", color='white', fontsize=8,
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.5))
                    
                    ax.set_title(f'T={frame["time_min"]:.0f}min, Droplet {did}\n{len(data["detected_nuclei"])} peaks',
                               fontsize=10)
                    ax.axis('off')
                    break
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_peak_detection_validation.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Tracking consistency
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Cell count consistency per droplet
        for did in self.masks.keys():
            times = []
            total_tracked = []
            detected_peaks = []
            
            for frame in self.results:
                if did in frame['droplet_results']:
                    times.append(frame['time_min'])
                    total_tracked.append(frame['droplet_results'][did]['total_tracked'])
                    detected_peaks.append(len(frame['droplet_results'][did]['detected_nuclei']))
            
            if times and max(total_tracked) > 0:
                ax1.plot(times, total_tracked, '-', label=f'Droplet {did} (tracked)', linewidth=2)
                ax1.plot(times, detected_peaks, '--', label=f'Droplet {did} (detected)', alpha=0.7)
        
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Cell Count')
        ax1.set_title('Tracking Consistency: Total Tracked vs Detected Peaks')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Missing cells over time
        times = []
        total_missing = []
        
        for frame in self.results:
            times.append(frame['time_min'])
            missing = sum(d['missing'] for d in frame['droplet_results'].values())
            total_missing.append(missing)
        
        ax2.plot(times, total_missing, 'r-', linewidth=2)
        ax2.fill_between(times, total_missing, alpha=0.3, color='red')
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Number of Missing Cells')
        ax2.set_title('Temporarily Missing Cells (Detection Gaps)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_tracking_consistency.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Validation plots saved to output directory")


def run_enhanced_analysis(nd2_file, output_dir=None):
    """
    Run the enhanced analysis with peak detection and persistent tracking.
    
    Args:
        nd2_file: Path to ND2 file
        output_dir: Output directory (optional)
    
    Returns:
        analyzer object if successful
    """
    analyzer = ImprovedCancerAnalyzer(nd2_file)
    
    if analyzer.analyze(output_dir):
        return analyzer
    else:
        return None


# Integration function to use with existing code
def integrate_enhanced_detection(existing_analyzer_class):
    """
    Decorator to add enhanced detection to existing analyzer classes.
    
    Usage:
        @integrate_enhanced_detection
        class YourAnalyzer:
            ...
    """
    class EnhancedAnalyzer(existing_analyzer_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.nuclear_detector = EnhancedNuclearDetector()
            self.persistent_trackers = {}
        
        def detect_cells_enhanced(self, tritc_image, mask=None, droplet_id=None):
            """Replace existing cell detection with peak-based detection."""
            # Use enhanced nuclear detection
            nuclei = self.nuclear_detector.detect_nuclear_centers(tritc_image, mask)
            
            # Convert to your existing format
            cells = []
            for nucleus in nuclei:
                cells.append({
                    'centroid_x': nucleus['x'],
                    'centroid_y': nucleus['y'],
                    'mean_intensity': nucleus['integrated_intensity'],
                    'peak_intensity': nucleus['peak_intensity'],
                    'confidence': nucleus['confidence']
                })
            
            return cells
        
        def update_tracking_enhanced(self, cells, droplet_id, frame_num, bf_states=None):
            """Use persistent tracking."""
            if droplet_id not in self.persistent_trackers:
                self.persistent_trackers[droplet_id] = PersistentCellTracker()
            
            return self.persistent_trackers[droplet_id].update(cells, frame_num, bf_states)
    
    return EnhancedAnalyzer


if __name__ == "__main__":
    # Test the enhanced analyzer
    nd2_file = r"D:\New\BrainBites\Cell\2.nd2"
    
    print("Running Enhanced Analysis with Peak Detection...")
    print("=" * 60)
    print("Features:")
    print("- Peak-based nuclear detection (no blob merging)")
    print("- Persistent cell tracking (handles temporary detection failures)")
    print("- Dead cells remain marked as dead")
    print("- Consistent cell counts across frames")
    print("=" * 60)
    
    analyzer = run_enhanced_analysis(nd2_file)
    
    if analyzer:
        print("\nAnalysis completed successfully!")
        print("\nKey improvements:")
        print("1. Individual nuclei detected as peaks, not blobs")
        print("2. Cell IDs remain consistent even if detection temporarily fails")
        print("3. Once dead, cells stay dead (no resurrection)")
        print("4. Missing cells are tracked and can reappear")
    else:
        print("\nAnalysis failed. Please check your ND2 file path.")