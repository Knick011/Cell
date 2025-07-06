"""
Enhanced Cancer Cell Analysis System - Improved Tracking and Death Detection
Optimized for robust cell tracking and precise death event detection
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
from scipy.ndimage import label, gaussian_filter, binary_erosion
from scipy.spatial.distance import cdist
from skimage.measure import regionprops
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings('ignore')

# Import your existing modules
from nk_cancer_analyzer import ND2Analyzer
from nk_cancer_analyzer_phase3 import DropletDetector, DropletCellAnalyzer


class EnhancedTracker:
    """Enhanced cell tracker with improved assignment and gap handling."""
    
    def __init__(self, max_distance=30, max_gap_frames=5):
        self.max_distance = max_distance
        self.max_gap_frames = max_gap_frames
        self.tracks = {}
        self.next_id = 1
        self.assignment_history = []  # For debugging
        
    def build_tracks(self, all_detections):
        """Build complete tracks using Hungarian algorithm for optimal assignment."""
        print("Building enhanced cell tracks...")
        
        if not all_detections:
            return
        
        # Initialize tracks with first frame
        frame0, detections0 = all_detections[0]
        for det in detections0:
            track_id = self.next_id
            self.next_id += 1
            
            self.tracks[track_id] = {
                'positions': [(det['x'], det['y'])],
                'frames': [frame0],
                'properties': [det],
                'droplet_id': det['droplet_id'],
                'status': 'active',
                'death_frame': None,
                'death_reason': None,
                'gaps': [],
                'confidence_scores': [1.0],  # Track assignment confidence
                'velocity_history': []  # For motion prediction
            }
        
        print(f"Initialized {len(self.tracks)} tracks from frame 0")
        
        # Process subsequent frames
        for frame_idx in range(1, len(all_detections)):
            frame_num, detections = all_detections[frame_idx]
            
            if frame_idx % 20 == 0:
                print(f"  Processing frame {frame_num}/{len(all_detections)-1}")
            
            self._assign_frame(frame_num, detections)
        
        print(f"Built {len(self.tracks)} total tracks")
        self._compute_track_statistics()
        
    def _assign_frame(self, frame_num, detections):
        """Assign detections to tracks for current frame."""
        if not detections:
            return
        
        # Get candidate tracks (active or recently seen)
        candidate_tracks = []
        for tid, track in self.tracks.items():
            if track['frames']:
                frames_since_last = frame_num - track['frames'][-1]
                if frames_since_last <= self.max_gap_frames:
                    candidate_tracks.append(tid)
        
        if not candidate_tracks:
            # Create new tracks for all detections
            for det in detections:
                self._create_new_track(frame_num, det)
            return
        
        # Build cost matrix with motion prediction
        cost_matrix = self._build_cost_matrix(candidate_tracks, detections, frame_num)
        
        # Hungarian assignment
        if cost_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Apply assignments
            assigned_tracks = set()
            assigned_detections = set()
            
            for row, col in zip(row_indices, col_indices):
                cost = cost_matrix[row, col]
                
                if cost < self.max_distance:
                    tid = candidate_tracks[row]
                    det = detections[col]
                    
                    self._update_track(tid, frame_num, det, cost)
                    assigned_tracks.add(tid)
                    assigned_detections.add(col)
            
            # Create new tracks for unassigned detections
            for det_idx, det in enumerate(detections):
                if det_idx not in assigned_detections:
                    self._create_new_track(frame_num, det)
    
    def _build_cost_matrix(self, candidate_tracks, detections, frame_num):
        """Build cost matrix with motion prediction."""
        if not candidate_tracks or not detections:
            return np.array([])
        
        cost_matrix = np.full((len(candidate_tracks), len(detections)), np.inf)
        
        for i, tid in enumerate(candidate_tracks):
            track = self.tracks[tid]
            
            # Predict position based on velocity
            predicted_pos = self._predict_position(track, frame_num)
            
            for j, det in enumerate(detections):
                det_pos = (det['x'], det['y'])
                
                # Base distance cost
                distance = np.sqrt((predicted_pos[0] - det_pos[0])**2 + 
                                 (predicted_pos[1] - det_pos[1])**2)
                
                # Add penalty for droplet mismatch
                if det['droplet_id'] != track['droplet_id']:
                    distance += 100  # Heavy penalty for cross-droplet assignment
                
                # Add penalty for large gaps
                gap_frames = frame_num - track['frames'][-1]
                if gap_frames > 1:
                    distance += gap_frames * 5  # Penalty for gap size
                
                cost_matrix[i, j] = distance
        
        return cost_matrix
    
    def _predict_position(self, track, frame_num):
        """Predict cell position based on recent motion."""
        if len(track['positions']) < 2:
            return track['positions'][-1]
        
        # Use last position if gap is too large
        gap_frames = frame_num - track['frames'][-1]
        if gap_frames > 3:
            return track['positions'][-1]
        
        # Simple linear prediction based on last 2-3 positions
        recent_positions = track['positions'][-3:]
        recent_frames = track['frames'][-3:]
        
        if len(recent_positions) >= 2:
            # Calculate average velocity
            vx = vy = 0
            for i in range(1, len(recent_positions)):
                dt = recent_frames[i] - recent_frames[i-1]
                if dt > 0:
                    vx += (recent_positions[i][0] - recent_positions[i-1][0]) / dt
                    vy += (recent_positions[i][1] - recent_positions[i-1][1]) / dt
            
            vx /= (len(recent_positions) - 1)
            vy /= (len(recent_positions) - 1)
            
            # Predict position
            dt = gap_frames
            pred_x = track['positions'][-1][0] + vx * dt
            pred_y = track['positions'][-1][1] + vy * dt
            
            return (pred_x, pred_y)
        
        return track['positions'][-1]
    
    def _create_new_track(self, frame_num, det):
        """Create a new track."""
        track_id = self.next_id
        self.next_id += 1
        
        self.tracks[track_id] = {
            'positions': [(det['x'], det['y'])],
            'frames': [frame_num],
            'properties': [det],
            'droplet_id': det['droplet_id'],
            'status': 'active',
            'death_frame': None,
            'death_reason': None,
            'gaps': [],
            'confidence_scores': [1.0],
            'velocity_history': []
        }
    
    def _update_track(self, track_id, frame_num, det, cost):
        """Update existing track."""
        track = self.tracks[track_id]
        
        # Check for gap
        last_frame = track['frames'][-1]
        if frame_num - last_frame > 1:
            track['gaps'].append((last_frame, frame_num))
        
        # Update track
        track['positions'].append((det['x'], det['y']))
        track['frames'].append(frame_num)
        track['properties'].append(det)
        
        # Update confidence (lower cost = higher confidence)
        confidence = max(0.1, 1.0 - cost / self.max_distance)
        track['confidence_scores'].append(confidence)
        
        # Update velocity history
        if len(track['positions']) >= 2:
            dt = frame_num - track['frames'][-2]
            if dt > 0:
                vx = (det['x'] - track['positions'][-2][0]) / dt
                vy = (det['y'] - track['positions'][-2][1]) / dt
                track['velocity_history'].append((vx, vy))
    
    def _compute_track_statistics(self):
        """Compute statistics for all tracks."""
        print("Computing track statistics...")
        
        for tid, track in self.tracks.items():
            # Track length and gaps
            track['length'] = len(track['frames'])
            track['total_gap_frames'] = sum(end - start - 1 for start, end in track['gaps'])
            track['avg_confidence'] = np.mean(track['confidence_scores'])
            
            # Motion statistics
            if track['velocity_history']:
                velocities = np.array(track['velocity_history'])
                track['avg_speed'] = np.mean(np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2))
                track['motion_consistency'] = 1.0 - np.std(np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2))
            else:
                track['avg_speed'] = 0
                track['motion_consistency'] = 1.0
    
    def analyze_death_patterns(self, total_frames):
        """Enhanced death pattern analysis."""
        print("Analyzing death patterns with enhanced detection...")
        
        for tid, track in self.tracks.items():
            if track['length'] < 3:
                continue
            
            death_frame = self._detect_death_frame(track, total_frames)
            
            if death_frame:
                track['death_frame'] = death_frame
                track['status'] = 'dead'
                
                # Mark dying phase (3 frames before death)
                if death_frame in track['frames']:
                    death_idx = track['frames'].index(death_frame)
                    dying_start_idx = max(0, death_idx - 3)
                    track['dying_start_frame'] = track['frames'][dying_start_idx]
                else:
                    track['dying_start_frame'] = track['frames'][-1]
        
        # Report statistics
        dead_tracks = [t for t in self.tracks.values() if t['status'] == 'dead']
        print(f"Detected {len(dead_tracks)} dead cells out of {len(self.tracks)} total tracks")
        
        if dead_tracks:
            death_reasons = {}
            for track in dead_tracks:
                reason = track.get('death_reason', 'unknown')
                death_reasons[reason] = death_reasons.get(reason, 0) + 1
            
            print("Death reasons:")
            for reason, count in death_reasons.items():
                print(f"  {reason}: {count}")
    
    def _detect_death_frame(self, track, total_frames):
        """Detect death frame using multiple criteria."""
        properties = track['properties']
        frames = track['frames']
        
        # Method 1: Nuclear dispersion (aggregate expansion + intensity drop)
        death_frame = self._detect_nuclear_dispersion(properties, frames)
        if death_frame:
            track['death_reason'] = 'nuclear_dispersion'
            return death_frame
        
        # Method 2: Signal diffusion (low density)
        death_frame = self._detect_signal_diffusion(properties, frames)
        if death_frame:
            track['death_reason'] = 'signal_diffusion'
            return death_frame
        
        # Method 3: Disappearance
        death_frame = self._detect_disappearance(track, total_frames)
        if death_frame:
            track['death_reason'] = 'disappeared'
            return death_frame
        
        # Method 4: Sustained low intensity
        death_frame = self._detect_intensity_drop(properties, frames)
        if death_frame:
            track['death_reason'] = 'intensity_drop'
            return death_frame
        
        return None
    
    def _detect_nuclear_dispersion(self, properties, frames):
        """Detect death by nuclear aggregate dispersion."""
        if len(properties) < 5:
            return None
        
        areas = [p['area'] for p in properties]
        aggregate_sizes = [p['aggregate_pixels'] for p in properties]
        intensities = [p['intensity'] for p in properties]
        
        # Calculate integrated intensity
        integrated_intensities = [p['intensity'] * p['aggregate_pixels'] for p in properties]
        
        # Baseline from first 3 frames
        baseline_aggregate = np.mean(aggregate_sizes[:3])
        baseline_integrated = np.mean(integrated_intensities[:3])
        
        for i in range(3, len(aggregate_sizes)):
            recent_aggregate = np.mean(aggregate_sizes[max(0, i-2):i+1])
            recent_integrated = integrated_intensities[i]
            
            # Check for expansion + intensity drop
            if (recent_aggregate > baseline_aggregate * 1.4 and
                recent_integrated < baseline_integrated * 0.6):
                return frames[i]
        
        return None
    
    def _detect_signal_diffusion(self, properties, frames):
        """Detect death by signal diffusion."""
        if len(properties) < 5:
            return None
        
        for i in range(3, len(properties)):
            intensity = properties[i]['intensity']
            aggregate_size = properties[i]['aggregate_pixels']
            
            if aggregate_size > 0:
                density = intensity / np.sqrt(aggregate_size)
                baseline_density = properties[0]['intensity'] / np.sqrt(properties[0]['aggregate_pixels'])
                
                if density < baseline_density * 0.3:
                    return frames[i]
        
        return None
    
    def _detect_disappearance(self, track, total_frames):
        """Detect death by disappearance."""
        if not track['frames']:
            return None
        
        last_seen = track['frames'][-1]
        
        # Consider dead if missing for last 5+ frames
        if last_seen < total_frames - 5:
            return last_seen + 1
        
        return None
    
    def _detect_intensity_drop(self, properties, frames):
        """Detect death by sustained intensity drop."""
        if len(properties) < 5:
            return None
        
        intensities = [p['intensity'] for p in properties]
        baseline_intensity = np.mean(intensities[:3])
        
        # Look for sustained drop
        for i in range(3, len(intensities)):
            recent_intensity = np.mean(intensities[max(0, i-2):i+1])
            
            if recent_intensity < baseline_intensity * 0.4:
                # Confirm it stays low
                if i < len(intensities) - 1:
                    next_intensity = intensities[i+1]
                    if next_intensity < baseline_intensity * 0.5:
                        return frames[i]
                else:
                    return frames[i]
        
        return None


class RobustPropertyExtractor:
    """Robust nuclear property extractor with improved signal analysis."""
    
    def __init__(self, roi_radius=25):
        self.roi_radius = roi_radius
    
    def extract_properties(self, image, x, y):
        """Extract comprehensive nuclear properties."""
        # Define ROI with bounds checking
        h, w = image.shape
        y_min = max(0, int(y - self.roi_radius))
        y_max = min(h, int(y + self.roi_radius))
        x_min = max(0, int(x - self.roi_radius))
        x_max = min(w, int(x + self.roi_radius))
        
        if y_max <= y_min or x_max <= x_min:
            return self._empty_properties()
        
        roi = image[y_min:y_max, x_min:x_max]
        
        if roi.max() == 0:
            return self._empty_properties()
        
        # Multi-threshold analysis
        properties = {}
        
        # Threshold at different levels
        thresholds = [0.2, 0.3, 0.4, 0.5]
        max_val = roi.max()
        
        for thresh_frac in thresholds:
            threshold = max_val * thresh_frac
            mask = roi > threshold
            
            if np.sum(mask) > 0:
                properties[f'area_{int(thresh_frac*100)}'] = np.sum(mask)
                properties[f'intensity_{int(thresh_frac*100)}'] = np.mean(roi[mask])
                properties[f'integrated_{int(thresh_frac*100)}'] = np.sum(roi[mask])
        
        # Use 30% threshold as primary
        primary_threshold = max_val * 0.3
        primary_mask = roi > primary_threshold
        
        if np.sum(primary_mask) > 0:
            # Basic properties
            aggregate_pixels = np.sum(primary_mask)
            mean_intensity = np.mean(roi[primary_mask])
            
            # Morphological properties
            labeled, n_regions = label(primary_mask)
            
            if n_regions > 0:
                # Get largest region
                region_props = regionprops(labeled, intensity_image=roi)
                largest_region = max(region_props, key=lambda r: r.area)
                
                area = largest_region.area
                eccentricity = largest_region.eccentricity
                solidity = largest_region.solidity
                
                # Intensity distribution
                intensities = roi[primary_mask]
                intensity_std = np.std(intensities)
                intensity_skew = self._calculate_skewness(intensities)
                
            else:
                area = aggregate_pixels
                eccentricity = 0
                solidity = 1
                intensity_std = 0
                intensity_skew = 0
        else:
            aggregate_pixels = 0
            mean_intensity = 0
            area = 0
            eccentricity = 0
            solidity = 1
            intensity_std = 0
            intensity_skew = 0
        
        return {
            'intensity': mean_intensity,
            'area': area,
            'aggregate_pixels': aggregate_pixels,
            'eccentricity': eccentricity,
            'solidity': solidity,
            'intensity_std': intensity_std,
            'intensity_skew': intensity_skew,
            **properties
        }
    
    def _empty_properties(self):
        """Return empty properties dict."""
        return {
            'intensity': 0,
            'area': 0,
            'aggregate_pixels': 0,
            'eccentricity': 0,
            'solidity': 1,
            'intensity_std': 0,
            'intensity_skew': 0
        }
    
    def _calculate_skewness(self, data):
        """Calculate skewness of intensity distribution."""
        if len(data) < 3:
            return 0
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            return 0
        
        return np.mean(((data - mean_val) / std_val) ** 3)


class EnhancedAnalyzer:
    """Enhanced analyzer with improved tracking and visualization."""
    
    def __init__(self, nd2_file, time_interval_min=15):
        self.nd2_file = nd2_file
        self.time_interval = time_interval_min
        self.analyzer = None
        self.droplets = None
        self.masks = None
        self.tracker = None
        self.frame_cache = []
        self.results = {}
        self.analysis_metadata = {}
        
    def analyze(self):
        """Run complete enhanced analysis."""
        print(f"Starting enhanced analysis of {os.path.basename(self.nd2_file)}")
        
        # Load ND2 file
        self.analyzer = ND2Analyzer(self.nd2_file)
        if not self.analyzer.load_file():
            print("Failed to load ND2 file")
            return False
        
        # Store metadata
        self.analysis_metadata = {
            'filename': os.path.basename(self.nd2_file),
            'total_frames': self.analyzer.metadata['frames'],
            'time_interval': self.time_interval,
            'analysis_time': datetime.now().isoformat()
        }
        
        # Detect droplets
        print("Detecting droplets...")
        droplet_detector = DropletDetector()
        bf_frame = self.analyzer.get_frame(0, 'brightfield')
        self.droplets = droplet_detector.detect_droplets(bf_frame)
        self.masks, _ = droplet_detector.create_droplet_masks(bf_frame.shape, self.droplets)
        
        print(f"Found {len(self.droplets)} droplets")
        
        # Initialize results
        for droplet in self.droplets:
            self.results[droplet['id']] = {
                'initial_cells': 0,
                'final_alive': 0,
                'total_dead': 0,
                'death_times': [],
                'death_reasons': {}
            }
        
        # Detect all cells
        print("Detecting cells in all frames...")
        all_detections = self._detect_all_cells()
        
        # Build enhanced tracks
        print("Building enhanced tracks...")
        self.tracker = EnhancedTracker(max_distance=35, max_gap_frames=5)
        self.tracker.build_tracks(all_detections)
        
        # Analyze death patterns
        self.tracker.analyze_death_patterns(self.analyzer.metadata['frames'])
        
        # Build frame data
        print("Building frame data...")
        self._build_frame_data()
        
        # Finalize results
        self._finalize_results()
        
        print("Enhanced analysis complete!")
        return True
    
    def _detect_all_cells(self):
        """Detect cells with enhanced property extraction."""
        cell_analyzer = DropletCellAnalyzer(DropletDetector(), None)
        property_extractor = RobustPropertyExtractor(roi_radius=25)
        all_detections = []
        
        total_frames = self.analyzer.metadata['frames']
        
        for t in range(total_frames):
            if t % 10 == 0:
                print(f"  Frame {t}/{total_frames-1} ({t/total_frames*100:.1f}%)")
            
            bf_frame = self.analyzer.get_frame(t, 'brightfield')
            tritc_frame = self.analyzer.get_frame(t, 'TRITC')
            
            # Cache frames
            self.frame_cache.append({
                'timepoint': t,
                'brightfield': bf_frame,
                'tritc': tritc_frame
            })
            
            frame_detections = []
            
            for droplet in self.droplets:
                did = droplet['id']
                mask = self.masks[did]
                
                # Extract masked regions
                masked_tritc = tritc_frame.copy()
                masked_tritc[~mask] = 0
                masked_bf = bf_frame.copy()
                masked_bf[~mask] = 0
                
                # Detect nuclei
                try:
                    nuclei = cell_analyzer._detect_nuclei(masked_tritc, masked_bf, mask, droplet)
                    
                    # Extract enhanced properties
                    for nx, ny, intensity, area in nuclei:
                        properties = property_extractor.extract_properties(tritc_frame, nx, ny)
                        
                        detection = {
                            'x': nx,
                            'y': ny,
                            'droplet_id': did,
                            **properties
                        }
                        frame_detections.append(detection)
                        
                except Exception as e:
                    print(f"Warning: Error detecting nuclei in droplet {did}, frame {t}: {e}")
                    continue
            
            all_detections.append((t, frame_detections))
        
        return all_detections
    
    def _build_frame_data(self):
        """Build frame data with enhanced track information."""
        self.frame_data = []
        
        for t, frame_cache in enumerate(self.frame_cache):
            frame_result = {
                'timepoint': t,
                'time_min': t * self.time_interval,
                'brightfield': frame_cache['brightfield'],
                'tritc': frame_cache['tritc'],
                'droplet_data': {}
            }
            
            # Initialize droplet data
            for droplet in self.droplets:
                did = droplet['id']
                frame_result['droplet_data'][did] = {
                    'cells': [],
                    'stats': {'alive': 0, 'dying': 0, 'dead': 0}
                }
            
            # Add cells from tracks
            for tid, track in self.tracker.tracks.items():
                if t in track['frames']:
                    idx = track['frames'].index(t)
                    pos = track['positions'][idx]
                    props = track['properties'][idx]
                    did = track['droplet_id']
                    
                    # Determine status
                    status = 'alive'  # Changed from 'active' to 'alive' to match stats dict
                    if track['status'] == 'dead':
                        if track['death_frame'] and t >= track['death_frame']:
                            continue  # Skip dead cells
                        elif hasattr(track, 'dying_start_frame') and t >= track['dying_start_frame']:
                            status = 'dying'
                    
                    cell_info = {
                        'centroid_x': pos[0],
                        'centroid_y': pos[1],
                        'track_id': tid,
                        'status': status,
                        'confidence': track['confidence_scores'][idx],
                        **props
                    }
                    
                    frame_result['droplet_data'][did]['cells'].append(cell_info)
                    frame_result['droplet_data'][did]['stats'][status] += 1
            
            # Count dead cells
            for tid, track in self.tracker.tracks.items():
                if (track['status'] == 'dead' and 
                    track['death_frame'] and 
                    t >= track['death_frame']):
                    did = track['droplet_id']
                    frame_result['droplet_data'][did]['stats']['dead'] += 1
            
            self.frame_data.append(frame_result)
    
    def _finalize_results(self):
        """Finalize results with enhanced statistics."""
        # Count initial cells
        if self.frame_data:
            frame0 = self.frame_data[0]
            for did in self.results:
                self.results[did]['initial_cells'] = frame0['droplet_data'][did]['stats']['alive']
        
        # Count final alive cells
        if self.frame_data:
            final_frame = self.frame_data[-1]
            for did in self.results:
                self.results[did]['final_alive'] = final_frame['droplet_data'][did]['stats']['alive']
        
        # Process death events
        for tid, track in self.tracker.tracks.items():
            if track['status'] == 'dead' and track['death_frame']:
                did = track['droplet_id']
                death_time = track['death_frame'] * self.time_interval
                
                self.results[did]['death_times'].append(death_time)
                self.results[did]['total_dead'] += 1
                
                # Count death reasons
                reason = track.get('death_reason', 'unknown')
                self.results[did]['death_reasons'][reason] = self.results[did]['death_reasons'].get(reason, 0) + 1
        
        # Sort death times
        for did in self.results:
            self.results[did]['death_times'].sort()
    
    def export_results(self):
        """Export enhanced results."""
        base_name = os.path.splitext(os.path.basename(self.nd2_file))[0]
        excel_data = []
        
        for droplet in self.droplets:
            did = droplet['id']
            result = self.results[did]
            
            row = {
                'ND2 Name': base_name,
                'Droplet Number': did,
                'Droplet Type': droplet['type'],
                'Number of cancer cells at the start': result['initial_cells'],
                'Number of cancer cells dead': result['total_dead'],
                'Number of cancer cells alive at the end': result['final_alive'],
                'Survival Rate (%)': (result['final_alive'] / result['initial_cells'] * 100) if result['initial_cells'] > 0 else 0
            }
            
            # Add death times
            for i, death_time in enumerate(result['death_times']):
                row[f'Time of death for cell {i+1}'] = death_time
            
            # Add death reasons
            for reason, count in result['death_reasons'].items():
                row[f'Deaths by {reason}'] = count
            
            excel_data.append(row)
        
        return pd.DataFrame(excel_data)

    def create_enhanced_movie(self, output_path=None, fps=3):
        """Create enhanced movie with better visualization."""
        if not self.frame_data:
            return
        
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(self.nd2_file))[0]
            output_path = f"{base_name}_enhanced_analysis.mp4"
        
        print(f"Creating enhanced movie: {output_path}")
        
        # Calculate grid layout
        n_droplets = len(self.droplets)
        cols = int(np.ceil(np.sqrt(n_droplets * 1.5)))
        rows = int(np.ceil(n_droplets / cols))
        
        # Create figure with enhanced layout
        fig = plt.figure(figsize=(cols * 4.5, rows * 4.5 + 2))
        gs = GridSpec(rows + 1, cols, figure=fig, height_ratios=[1]*rows + [0.3])
        
        # Setup enhanced droplet displays
        droplet_displays = self._setup_droplet_displays(fig, gs, cols)
        
        # Global info panel
        ax_info = fig.add_subplot(gs[-1, :])
        ax_info.axis('off')
        
        # Create multiple info elements
        global_info = ax_info.text(0.5, 0.8, '', transform=ax_info.transAxes,
                                  ha='center', va='center', fontsize=14, weight='bold',
                                  bbox=dict(boxstyle="round,pad=0.5", 
                                          facecolor='lightblue', alpha=0.8))
        
        survival_info = ax_info.text(0.25, 0.3, '', transform=ax_info.transAxes,
                                   ha='center', va='center', fontsize=12,
                                   bbox=dict(boxstyle="round,pad=0.3", 
                                           facecolor='lightgreen', alpha=0.8))
        
        death_info = ax_info.text(0.75, 0.3, '', transform=ax_info.transAxes,
                                ha='center', va='center', fontsize=12,
                                bbox=dict(boxstyle="round,pad=0.3", 
                                        facecolor='lightcoral', alpha=0.8))
        
        def update_frame(frame_idx):
            """Enhanced frame update with better visualization."""
            frame = self.frame_data[frame_idx]
            time_min = frame['time_min']
            
            # Update each droplet display
            total_stats = {'alive': 0, 'dying': 0, 'dead': 0}
            droplet_survival_rates = []
            
            for did, display in droplet_displays.items():
                if did not in frame['droplet_data']:
                    continue
                
                # Update image with enhanced contrast
                self._update_droplet_image(display, frame, did)
                
                # Update cell markers with confidence indicators
                self._update_cell_markers(display, frame['droplet_data'][did], frame_idx)
                
                # Update statistics display
                stats = frame['droplet_data'][did]['stats']
                initial = self.results[did]['initial_cells']
                
                # Calculate survival rate
                if initial > 0:
                    survival_rate = (stats['alive'] / initial) * 100
                    droplet_survival_rates.append(survival_rate)
                else:
                    survival_rate = 0
                
                # Enhanced info text with color coding
                info_text = f"A:{stats['alive']} D:{stats['dying']} †:{stats['dead']}\n"
                info_text += f"Survival: {survival_rate:.1f}%"
                
                display['info'].set_text(info_text)
                
                # Color code based on status
                if stats['alive'] == 0 and initial > 0:
                    display['info'].set_color('red')
                    display['info'].set_weight('bold')
                elif stats['dying'] > 0:
                    display['info'].set_color('orange')
                    display['info'].set_weight('normal')
                elif survival_rate < 50:
                    display['info'].set_color('darkorange')
                    display['info'].set_weight('normal')
                else:
                    display['info'].set_color('green')
                    display['info'].set_weight('normal')
                
                # Accumulate totals
                for key in total_stats:
                    total_stats[key] += stats[key]
            
            # Update global information
            total_initial = sum(r['initial_cells'] for r in self.results.values())
            overall_survival = (total_stats['alive'] / total_initial * 100) if total_initial > 0 else 0
            
            global_text = f"Time: {time_min:.0f} min | Frame: {frame_idx+1}/{len(self.frame_data)}"
            global_info.set_text(global_text)
            
            # Survival statistics
            if droplet_survival_rates:
                avg_survival = np.mean(droplet_survival_rates)
                survival_text = f"Overall Survival: {overall_survival:.1f}%\n"
                survival_text += f"Average per Droplet: {avg_survival:.1f}%"
            else:
                survival_text = "No survival data"
            survival_info.set_text(survival_text)
            
            # Death statistics
            total_current = sum(total_stats.values())
            death_text = f"Alive: {total_stats['alive']} | Dying: {total_stats['dying']}\n"
            death_text += f"Dead: {total_stats['dead']} | Total: {total_current}/{total_initial}"
            death_info.set_text(death_text)
            
            # Main title with progress
            progress = (frame_idx + 1) / len(self.frame_data) * 100
            fig.suptitle(f'Enhanced Cancer Cell Analysis - T = {time_min:.0f} min ({progress:.1f}%)', 
                        fontsize=16, y=0.98, weight='bold')
            
            return self._collect_artists(droplet_displays, global_info, survival_info, death_info)
        
        # Create and save animation
        anim = animation.FuncAnimation(fig, update_frame, frames=len(self.frame_data),
                                     interval=1000/fps, blit=False, repeat=True)
        
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=3000, extra_args=['-vcodec', 'libx264'])
            anim.save(output_path, writer=writer, dpi=120)
            print(f"Enhanced movie saved: {output_path}")
        except Exception as e:
            print(f"Error saving movie: {e}")
            # Try alternative format
            try:
                anim.save(output_path.replace('.mp4', '.gif'), writer='pillow', fps=fps//2)
                print(f"Saved as GIF instead: {output_path.replace('.mp4', '.gif')}")
            except Exception as e2:
                print(f"Failed to save animation: {e2}")
        
        plt.close()
    
    def _setup_droplet_displays(self, fig, gs, cols):
        """Setup droplet display panels."""
        droplet_displays = {}
        
        for i, droplet in enumerate(self.droplets):
            did = droplet['id']
            row = i // cols
            col = i % cols
            
            ax = fig.add_subplot(gs[row, col])
            
            # Get crop region with better margins
            mask = self.masks[did]
            y_coords, x_coords = np.where(mask)
            
            if len(y_coords) > 0:
                margin = 25  # Increased margin
                y_min = max(0, y_coords.min() - margin)
                y_max = min(self.frame_data[0]['tritc'].shape[0], y_coords.max() + margin)
                x_min = max(0, x_coords.min() - margin)
                x_max = min(self.frame_data[0]['tritc'].shape[1], x_coords.max() + margin)
                
                # Setup display
                ax.set_title(f"Droplet {did} ({droplet['type']})", fontsize=11, pad=3, weight='bold')
                ax.axis('off')
                
                # Enhanced droplet circle
                cx = droplet['center_x'] - x_min
                cy = droplet['center_y'] - y_min
                circle = plt.Circle((cx, cy), droplet['radius_px'], 
                                  color='cyan', fill=False, linewidth=2, alpha=0.8)
                ax.add_patch(circle)
                
                # Initialize image
                frame0 = self.frame_data[0]
                tritc_crop = frame0['tritc'][y_min:y_max, x_min:x_max]
                img = ax.imshow(np.zeros_like(tritc_crop), cmap='hot', vmin=0, vmax=1)
                
                # Info text with better positioning
                info = ax.text(0.5, -0.12, '', transform=ax.transAxes,
                             ha='center', va='top', fontsize=9, weight='bold',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
                
                droplet_displays[did] = {
                    'ax': ax,
                    'crop': (x_min, y_min, x_max, y_max),
                    'img': img,
                    'info': info,
                    'markers': [],
                    'confidence_bars': []
                }
        
        return droplet_displays
    
    def _update_droplet_image(self, display, frame, did):
        """Update droplet image with enhanced contrast."""
        x_min, y_min, x_max, y_max = display['crop']
        tritc_crop = frame['tritc'][y_min:y_max, x_min:x_max]
        mask_crop = self.masks[did][y_min:y_max, x_min:x_max]
        
        # Apply mask
        masked = tritc_crop.copy().astype(float)
        masked[~mask_crop] = 0
        
        # Enhanced contrast adjustment
        if np.any(masked > 0):
            # Use adaptive percentiles
            nonzero_vals = masked[masked > 0]
            vmin = np.percentile(nonzero_vals, 2)
            vmax = np.percentile(nonzero_vals, 98)
            
            # Apply gamma correction for better visibility
            enhanced = np.clip((masked - vmin) / (vmax - vmin + 1e-8), 0, 1)
            enhanced = np.power(enhanced, 0.8)  # Gamma correction
        else:
            enhanced = masked
        
        display['img'].set_data(enhanced)
    
    def _update_cell_markers(self, display, droplet_data, frame_idx):
        """Update cell markers with confidence visualization."""
        # Clear old markers
        for marker in display['markers']:
            marker.remove()
        display['markers'].clear()
        
        x_min, y_min, x_max, y_max = display['crop']
        
        for cell in droplet_data['cells']:
            cx = cell['centroid_x'] - x_min
            cy = cell['centroid_y'] - y_min
            
            # Get confidence if available
            confidence = cell.get('confidence', 1.0)
            
            if cell['status'] == 'alive':
                # Size marker based on confidence
                size = 6 + confidence * 4
                alpha = 0.7 + confidence * 0.3
                
                marker = display['ax'].plot(cx, cy, '+', color='white', 
                                          markersize=size, markeredgewidth=2, alpha=alpha)[0]
                display['markers'].append(marker)
                
                # Add confidence indicator for low confidence
                if confidence < 0.7:
                    conf_marker = display['ax'].plot(cx, cy, 'o', color='yellow', 
                                                   markersize=size//2, fillstyle='none', 
                                                   markeredgewidth=1, alpha=0.6)[0]
                    display['markers'].append(conf_marker)
                    
            elif cell['status'] == 'dying':
                # Animated dying marker
                phase = (frame_idx % 6) / 6.0 * 2 * np.pi
                size = 8 + 2 * np.sin(phase)
                alpha = 0.6 + 0.4 * np.abs(np.sin(phase))
                
                marker = display['ax'].plot(cx, cy, 'x', color='orange', 
                                          markersize=size, markeredgewidth=3, alpha=alpha)[0]
                display['markers'].append(marker)
                
                # Show track ID for dying cells
                if 'track_id' in cell:
                    text = display['ax'].text(cx + 8, cy - 8, f"T{cell['track_id']}", 
                                            color='orange', fontsize=7, weight='bold',
                                            bbox=dict(boxstyle="round,pad=0.2", 
                                                    facecolor='black', alpha=0.7))
                    display['markers'].append(text)
    
    def _collect_artists(self, droplet_displays, global_info, survival_info, death_info):
        """Collect all artists for animation."""
        artists = []
        
        for display in droplet_displays.values():
            artists.append(display['img'])
            artists.append(display['info'])
            artists.extend(display['markers'])
        
        artists.extend([global_info, survival_info, death_info])
        return artists


def analyze_single_file_enhanced(nd2_file, output_dir=None):
    """Analyze single file with enhanced system."""
    if output_dir is None:
        output_dir = os.path.dirname(nd2_file)
    
    analyzer = EnhancedAnalyzer(nd2_file)
    
    if analyzer.analyze():
        # Export results
        results_df = analyzer.export_results()
        
        # Save individual results
        base_name = os.path.splitext(os.path.basename(nd2_file))[0]
        results_path = os.path.join(output_dir, f"{base_name}_enhanced_results.xlsx")
        results_df.to_excel(results_path, index=False)
        
        print(f"Results saved to: {results_path}")
        
        # Create enhanced movie
        movie_path = os.path.join(output_dir, f"{base_name}_enhanced_analysis.mp4")
        analyzer.create_enhanced_movie(movie_path)
        
        # Close analyzer
        analyzer.analyzer.close()
        
        return results_df
    
    return None


def batch_analyze_enhanced(directory_path, output_file='enhanced_cancer_analysis_results.xlsx'):
    """Enhanced batch analysis with better error handling and reporting."""
    nd2_files = glob.glob(os.path.join(directory_path, "*.nd2"))
    
    if not nd2_files:
        print(f"No ND2 files found in {directory_path}")
        return None
    
    print(f"Found {len(nd2_files)} ND2 files for enhanced analysis")
    
    all_results = []
    processing_log = []
    
    for i, nd2_file in enumerate(nd2_files):
        filename = os.path.basename(nd2_file)
        print(f"\n{'='*80}")
        print(f"Processing file {i+1}/{len(nd2_files)}: {filename}")
        print(f"{'='*80}")
        
        start_time = datetime.now()
        
        try:
            # Analyze file
            analyzer = EnhancedAnalyzer(nd2_file)
            
            if analyzer.analyze():
                # Create movie
                try:
                    movie_path = os.path.join(directory_path, 
                                            os.path.splitext(filename)[0] + "_enhanced_analysis.mp4")
                    analyzer.create_enhanced_movie(movie_path)
                    movie_status = "Success"
                except Exception as e:
                    movie_status = f"Failed: {str(e)[:50]}"
                
                # Get results
                results = analyzer.export_results()
                all_results.append(results)
                
                # Log success
                processing_time = (datetime.now() - start_time).total_seconds()
                log_entry = {
                    'filename': filename,
                    'status': 'Success',
                    'processing_time_sec': processing_time,
                    'droplets_found': len(analyzer.droplets),
                    'total_tracks': len(analyzer.tracker.tracks),
                    'movie_creation': movie_status,
                    'error': None
                }
                
                print(f"✓ Successfully processed {filename}")
                print(f"  - Processing time: {processing_time:.1f} seconds")
                print(f"  - Droplets found: {len(analyzer.droplets)}")
                print(f"  - Cell tracks: {len(analyzer.tracker.tracks)}")
                print(f"  - Movie creation: {movie_status}")
                
                # Clean up
                analyzer.analyzer.close()
                
            else:
                raise Exception("Analysis failed - could not load or process file")
                
        except Exception as e:
            # Log error
            processing_time = (datetime.now() - start_time).total_seconds()
            log_entry = {
                'filename': filename,
                'status': 'Failed',
                'processing_time_sec': processing_time,
                'droplets_found': 0,
                'total_tracks': 0,
                'movie_creation': 'Not attempted',
                'error': str(e)[:100]
            }
            
            print(f"✗ Error processing {filename}: {str(e)[:100]}")
        
        processing_log.append(log_entry)
    
    # Combine and save results
    if all_results:
        print(f"\n{'='*80}")
        print("Combining results and saving...")
        
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save main results
        excel_path = os.path.join(directory_path, output_file)
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main results
            combined_df.to_excel(writer, sheet_name='Cell Analysis Results', index=False)
            
            # Processing log
            log_df = pd.DataFrame(processing_log)
            log_df.to_excel(writer, sheet_name='Processing Log', index=False)
            
            # Summary statistics
            summary_stats = _generate_summary_stats(combined_df, log_df)
            summary_stats.to_excel(writer, sheet_name='Summary Statistics', index=False)
        
        print(f"✓ All results saved to: {excel_path}")
        print(f"✓ Successfully processed {len(all_results)}/{len(nd2_files)} files")
        
        # Print summary
        _print_analysis_summary(log_df, combined_df)
        
        return combined_df, log_df
    
    else:
        print(f"\n{'='*80}")
        print("No files were successfully processed")
        
        # Save error log
        if processing_log:
            log_df = pd.DataFrame(processing_log)
            error_log_path = os.path.join(directory_path, 'processing_errors.xlsx')
            log_df.to_excel(error_log_path, index=False)
            print(f"Error log saved to: {error_log_path}")
        
        return None, None


def _generate_summary_stats(results_df, log_df):
    """Generate summary statistics."""
    summary_data = []
    
    # Overall statistics
    total_files = len(log_df)
    successful_files = len(log_df[log_df['status'] == 'Success'])
    total_droplets = results_df['Droplet Number'].nunique() if not results_df.empty else 0
    
    summary_data.append({
        'Metric': 'Total Files Processed',
        'Value': total_files,
        'Description': 'Number of ND2 files attempted'
    })
    
    summary_data.append({
        'Metric': 'Successfully Processed',
        'Value': successful_files,
        'Description': 'Number of files processed without errors'
    })
    
    summary_data.append({
        'Metric': 'Success Rate (%)',
        'Value': (successful_files / total_files * 100) if total_files > 0 else 0,
        'Description': 'Percentage of files successfully processed'
    })
    
    if not results_df.empty:
        # Cell statistics
        total_initial_cells = results_df['Number of cancer cells at the start'].sum()
        total_final_alive = results_df['Number of cancer cells alive at the end'].sum()
        total_dead = results_df['Number of cancer cells dead'].sum()
        
        summary_data.extend([
            {
                'Metric': 'Total Droplets Analyzed',
                'Value': total_droplets,
                'Description': 'Number of droplets across all files'
            },
            {
                'Metric': 'Total Initial Cells',
                'Value': total_initial_cells,
                'Description': 'Total cancer cells at experiment start'
            },
            {
                'Metric': 'Total Final Alive',
                'Value': total_final_alive,
                'Description': 'Total cancer cells alive at experiment end'
            },
            {
                'Metric': 'Total Dead Cells',
                'Value': total_dead,
                'Description': 'Total cancer cells that died during experiment'
            },
            {
                'Metric': 'Overall Survival Rate (%)',
                'Value': (total_final_alive / total_initial_cells * 100) if total_initial_cells > 0 else 0,
                'Description': 'Overall percentage of cells surviving'
            }
        ])
        
        # Processing time statistics
        if 'processing_time_sec' in log_df.columns:
            successful_times = log_df[log_df['status'] == 'Success']['processing_time_sec']
            if not successful_times.empty:
                summary_data.extend([
                    {
                        'Metric': 'Average Processing Time (sec)',
                        'Value': successful_times.mean(),
                        'Description': 'Average time to process one file'
                    },
                    {
                        'Metric': 'Total Processing Time (min)',
                        'Value': successful_times.sum() / 60,
                        'Description': 'Total time spent processing all files'
                    }
                ])
    
    return pd.DataFrame(summary_data)


def _print_analysis_summary(log_df, results_df):
    """Print analysis summary to console."""
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    total_files = len(log_df)
    successful = len(log_df[log_df['status'] == 'Success'])
    
    print(f"Files processed: {successful}/{total_files} ({successful/total_files*100:.1f}% success rate)")
    
    if not results_df.empty:
        total_initial = results_df['Number of cancer cells at the start'].sum()
        total_alive = results_df['Number of cancer cells alive at the end'].sum()
        total_dead = results_df['Number of cancer cells dead'].sum()
        
        print(f"Total droplets analyzed: {results_df['Droplet Number'].nunique()}")
        print(f"Cancer cells tracked: {total_initial} initial → {total_alive} alive + {total_dead} dead")
        print(f"Overall survival rate: {total_alive/total_initial*100:.1f}%")
    
    # Show any errors
    failed_files = log_df[log_df['status'] == 'Failed']
    if not failed_files.empty:
        print(f"\nFailed files ({len(failed_files)}):")
        for _, row in failed_files.iterrows():
            print(f"  - {row['filename']}: {row['error']}")


if __name__ == "__main__":
    # Test with single file
    nd2_file = r"D:\New\BrainBites\Cell\03.nd2"
    
    print("Running enhanced cancer cell analysis...")
    results = analyze_single_file_enhanced(nd2_file)
    
    if results is not None:
        print("\nAnalysis completed successfully!")
        print(f"Results shape: {results.shape}")
        print("\nResults preview:")
        print(results.head())
        
        # Print summary statistics
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        
        total_droplets = len(results)
        total_initial = results['Number of cancer cells at the start'].sum()
        total_alive = results['Number of cancer cells alive at the end'].sum()
        total_dead = results['Number of cancer cells dead'].sum()
        
        print(f"Droplets analyzed: {total_droplets}")
        print(f"Initial cancer cells: {total_initial}")
        print(f"Final alive cells: {total_alive}")
        print(f"Total dead cells: {total_dead}")
        print(f"Overall survival rate: {total_alive/total_initial*100:.1f}%")
        
        # Show survival by droplet type if available
        if 'Droplet Type' in results.columns:
            print(f"\nSurvival by droplet type:")
            type_summary = results.groupby('Droplet Type').agg({
                'Number of cancer cells at the start': 'sum',
                'Number of cancer cells alive at the end': 'sum',
                'Survival Rate (%)': 'mean'
            })
            print(type_summary)
    else:
        print("Analysis failed!")
    
    # Uncomment to run batch analysis on a directory
    # directory = r"D:\New\BrainBites\Cell"
    # print("\nRunning batch analysis...")
    # batch_results, batch_log = batch_analyze_enhanced(directory)