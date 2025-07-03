"""
Droplet-Focused Cancer Cell Analysis
Creates individual movies for each droplet with enhanced death detection
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
from scipy.ndimage import label, binary_erosion, binary_dilation

from nk_cancer_analyzer import ND2Analyzer
from nk_cancer_analyzer_phase3 import DropletDetector, DropletCellAnalyzer
from nk_cancer_analyzer_phase4 import CellTracker

class EnhancedCellTracker(CellTracker):
    """Enhanced tracker with morphological death detection."""
    
    def __init__(self, max_distance=30, max_frames_missing=3):
        super().__init__(max_distance, max_frames_missing)
        self.morphology_history = {}  # Store cell morphology over time
        
    def update_with_morphology(self, detections, frame_num, cell_images=None):
        """Update tracks with morphology information."""
        # First do normal tracking
        assignments = self.update(detections, frame_num, 'cancer')
        
        # Store morphology data
        if cell_images is not None:
            for i, det in enumerate(detections):
                if i in assignments:
                    track_id = assignments[i]
                    if track_id not in self.morphology_history:
                        self.morphology_history[track_id] = []
                    
                    # Calculate morphological features
                    morph_features = self._calculate_morphology(cell_images[i])
                    self.morphology_history[track_id].append({
                        'frame': frame_num,
                        'area': morph_features['area'],
                        'compactness': morph_features['compactness'],
                        'fragmentation': morph_features['fragmentation']
                    })
        
        return assignments
    
    def _calculate_morphology(self, cell_image):
        """Calculate morphological features of a cell."""
        # Binary threshold
        threshold = np.percentile(cell_image[cell_image > 0], 50) if np.any(cell_image > 0) else 0
        binary = cell_image > threshold
        
        # Calculate area
        area = np.sum(binary)
        
        # Calculate compactness (how round/compact vs spread out)
        if area > 0:
            # Find contour
            labeled, num = label(binary)
            
            # Erode to find core
            core = binary_erosion(binary, iterations=2)
            core_area = np.sum(core)
            
            # Fragmentation score (how many pieces)
            dilated = binary_dilation(binary, iterations=1)
            labeled_dilated, num_fragments = label(dilated)
            
            compactness = core_area / area if area > 0 else 0
            fragmentation = num_fragments
        else:
            compactness = 0
            fragmentation = 0
        
        return {
            'area': area,
            'compactness': compactness,
            'fragmentation': fragmentation
        }
    
    def mark_dead_cells_enhanced(self, frame_num, intensity_threshold=0.3):
        """Enhanced death detection using intensity and morphology."""
        for tid, track in self.tracks.items():
            if track['type'] == 'cancer' and track['status'] == 'active':
                # Check if cell hasn't been seen recently
                if frame_num - track['last_seen'] > self.max_frames_missing:
                    # Don't immediately mark as dead - check if it reappears
                    track['status'] = 'missing'
                    track['missing_since'] = track['last_seen']
                
                # Check intensity drop
                elif len(track['intensities']) > 2:
                    current_intensity = track['intensities'][-1]
                    max_intensity = max(track['intensities'][:5])  # Use early frames as reference
                    
                    # Check morphology if available
                    dying_score = 0
                    
                    # Intensity check
                    if current_intensity < intensity_threshold * max_intensity:
                        dying_score += 1
                    
                    # Morphology check
                    if tid in self.morphology_history and len(self.morphology_history[tid]) > 2:
                        recent_morph = self.morphology_history[tid][-3:]
                        
                        # Check for area reduction
                        area_trend = [m['area'] for m in recent_morph]
                        if len(area_trend) >= 3 and area_trend[-1] < area_trend[0] * 0.5:
                            dying_score += 1
                        
                        # Check for fragmentation
                        if recent_morph[-1]['fragmentation'] > 2:
                            dying_score += 1
                        
                        # Check for loss of compactness
                        if recent_morph[-1]['compactness'] < 0.3:
                            dying_score += 1
                    
                    # Mark as dying if multiple indicators
                    if dying_score >= 2:
                        track['status'] = 'dying'
                        if 'first_dying_frame' not in track:
                            track['first_dying_frame'] = frame_num
                    
                    # Confirm death after sustained dying state
                    if track['status'] == 'dying' and 'first_dying_frame' in track:
                        if frame_num - track['first_dying_frame'] >= 3:
                            track['status'] = 'dead'
                            track['death_frame'] = frame_num
            
            # Handle missing cells that reappear
            elif track['status'] == 'missing':
                if frame_num == track['last_seen']:
                    # Cell reappeared!
                    track['status'] = 'active'
                    del track['missing_since']
                elif frame_num - track['missing_since'] > self.max_frames_missing + 3:
                    # Really gone
                    track['status'] = 'dead'
                    track['death_frame'] = track['missing_since']


class DropletFocusedAnalyzer:
    """Analyzer that creates individual movies for each droplet."""
    
    def __init__(self, nd2_file, time_interval_min=15):
        self.nd2_file = nd2_file
        self.time_interval = time_interval_min
        self.analyzer = None
        self.droplets = None
        self.droplet_trackers = {}  # Separate tracker for each droplet
        self.droplet_data = {}  # Store all data by droplet
        
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
        
        # Initialize separate tracker for each droplet
        for droplet in self.droplets:
            did = droplet['id']
            self.droplet_trackers[did] = EnhancedCellTracker(max_distance=30, max_frames_missing=3)
            self.droplet_data[did] = {
                'droplet_info': droplet,
                'frames': [],
                'summary': {}
            }
        
        # Process all frames
        cell_analyzer = DropletCellAnalyzer(droplet_detector, None)
        
        for t in range(self.analyzer.metadata['frames']):
            if t % 10 == 0:
                print(f"Processing frame {t}/{self.analyzer.metadata['frames']-1}")
            
            self._process_frame(t, cell_analyzer)
        
        # Calculate summaries
        self._calculate_summaries()
        
        print("Analysis complete!")
        return True
    
    def _process_frame(self, timepoint, cell_analyzer):
        """Process a single frame for all droplets."""
        # Get frames
        bf_frame = self.analyzer.get_frame(timepoint, 'brightfield')
        tritc_frame = self.analyzer.get_frame(timepoint, 'TRITC')
        
        for droplet in self.droplets:
            did = droplet['id']
            mask = self.masks[did]
            
            # Extract droplet region with margin
            margin = int(droplet['radius_px'] * 0.2)
            x0 = max(0, droplet['center_x'] - droplet['radius_px'] - margin)
            x1 = min(tritc_frame.shape[1], droplet['center_x'] + droplet['radius_px'] + margin)
            y0 = max(0, droplet['center_y'] - droplet['radius_px'] - margin)
            y1 = min(tritc_frame.shape[0], droplet['center_y'] + droplet['radius_px'] + margin)
            
            # Crop frames
            tritc_crop = tritc_frame[y0:y1, x0:x1].copy()
            mask_crop = mask[y0:y1, x0:x1]
            
            # Apply mask
            masked_tritc = tritc_crop.copy()
            masked_tritc[~mask_crop] = 0
            
            # Detect cells
            masked_bf = bf_frame[y0:y1, x0:x1].copy()
            masked_bf[~mask_crop] = 0
            
            # Adjust detection to cropped coordinates
            nuclei = cell_analyzer._detect_nuclei(masked_tritc, masked_bf, mask_crop, 
                                                {'center_x': droplet['center_x'] - x0,
                                                 'center_y': droplet['center_y'] - y0,
                                                 'radius_px': droplet['radius_px']})
            
            # Convert back to full image coordinates
            cancer_cells = []
            cell_images = []
            
            for nx, ny, intensity, area in nuclei:
                # Extract cell region for morphology
                cell_x = int(nx)
                cell_y = int(ny)
                cell_size = 15  # pixels
                
                cx0 = max(0, cell_x - cell_size)
                cx1 = min(tritc_crop.shape[1], cell_x + cell_size)
                cy0 = max(0, cell_y - cell_size)
                cy1 = min(tritc_crop.shape[0], cell_y + cell_size)
                
                if cx1 > cx0 and cy1 > cy0:
                    cell_img = tritc_crop[cy0:cy1, cx0:cx1]
                    cell_images.append(cell_img)
                    
                    cancer_cells.append({
                        'centroid_x': nx + x0,  # Convert to full image coords
                        'centroid_y': ny + y0,
                        'mean_intensity': intensity,
                        'area': area
                    })
            
            # Update tracker with morphology
            tracker = self.droplet_trackers[did]
            assignments = tracker.update_with_morphology(cancer_cells, timepoint, cell_images)
            
            # Enhanced death marking
            tracker.mark_dead_cells_enhanced(timepoint)
            
            # Count states
            alive = dying = dead = 0
            cell_states = {}
            
            for i, cell in enumerate(cancer_cells):
                if i in assignments:
                    track_id = assignments[i]
                    track = tracker.tracks[track_id]
                    cell_states[i] = (track_id, track['status'])
                    
                    if track['status'] == 'active':
                        alive += 1
                    elif track['status'] in ['dying', 'missing']:
                        dying += 1
            
            # Count dead from all tracks
            for tid, track in tracker.tracks.items():
                if track['status'] == 'dead':
                    dead += 1
            
            # Store frame data
            self.droplet_data[did]['frames'].append({
                'timepoint': timepoint,
                'time_min': timepoint * self.time_interval,
                'tritc_crop': tritc_crop,
                'mask_crop': mask_crop,
                'cells': cancer_cells,
                'cell_states': cell_states,
                'crop_coords': (x0, y0, x1, y1),
                'alive': alive,
                'dying': dying,
                'dead': dead
            })
    
    def _calculate_summaries(self):
        """Calculate summary statistics for each droplet."""
        for did, data in self.droplet_data.items():
            tracker = self.droplet_trackers[did]
            frames = data['frames']
            
            if not frames:
                continue
            
            # Initial and final counts
            initial_alive = frames[0]['alive']
            final_alive = frames[-1]['alive']
            
            # Death events
            death_times = []
            for tid, track in tracker.tracks.items():
                if track['death_frame'] is not None:
                    death_times.append(track['death_frame'] * self.time_interval)
            
            # Calculate summary
            data['summary'] = {
                'initial_cells': initial_alive,
                'final_cells': final_alive,
                'total_deaths': len(death_times),
                'survival_rate': (final_alive / initial_alive * 100) if initial_alive > 0 else 0,
                'avg_death_time': np.mean(death_times) if death_times else 'N/A',
                'first_death': min(death_times) if death_times else 'N/A',
                'last_death': max(death_times) if death_times else 'N/A'
            }
    
    def create_droplet_movies(self, output_dir, fps=3):
        """Create individual movie for each droplet."""
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.nd2_file))[0]
        
        for did, data in self.droplet_data.items():
            if not data['frames']:
                continue
            
            print(f"Creating movie for droplet {did}...")
            
            # Setup figure with custom layout
            fig = plt.figure(figsize=(12, 8))
            gs = GridSpec(3, 2, figure=fig, height_ratios=[3, 1, 0.5])
            
            # Main image
            ax_main = fig.add_subplot(gs[0, :])
            
            # Survival curve
            ax_survival = fig.add_subplot(gs[1, 0])
            
            # Cell count bars
            ax_counts = fig.add_subplot(gs[1, 1])
            
            # Info text area
            ax_info = fig.add_subplot(gs[2, :])
            ax_info.axis('off')
            
            # Initialize main image
            frame0 = data['frames'][0]
            tritc = frame0['tritc_crop']
            vmin, vmax = np.percentile(tritc[tritc > 0], [5, 99]) if np.any(tritc > 0) else (0, 1)
            tritc_enhanced = np.clip((tritc - vmin) / (vmax - vmin + 1e-8), 0, 1)
            
            # Apply red colormap
            tritc_colored = plt.cm.hot(tritc_enhanced)
            tritc_colored[:, :, 3] = 1  # Full opacity
            
            img_main = ax_main.imshow(tritc_colored)
            ax_main.set_title(f'Droplet {did} - T=0 min', fontsize=14)
            ax_main.axis('off')
            
            # Draw droplet circle
            x0, y0, x1, y1 = frame0['crop_coords']
            droplet = data['droplet_info']
            circle = plt.Circle((droplet['center_x'] - x0, droplet['center_y'] - y0), 
                              droplet['radius_px'], 
                              color='lime', fill=False, linewidth=2)
            ax_main.add_patch(circle)
            
            # Scale bar (50 µm)
            pixel_size = 0.65  # µm/pixel
            scale_length = 50 / pixel_size  # 50 µm in pixels
            scale_y = tritc_enhanced.shape[0] - 20
            ax_main.plot([10, 10 + scale_length], [scale_y, scale_y], 
                       'white', linewidth=3)
            ax_main.text(10 + scale_length/2, scale_y - 10, '50 µm', 
                       color='white', ha='center', fontsize=10)
            
            # Initialize survival curve
            times = [f['time_min'] for f in data['frames']]
            survivals = [f['alive'] for f in data['frames']]
            survival_line, = ax_survival.plot(times[0:1], survivals[0:1], 'b-', linewidth=2)
            ax_survival.set_xlabel('Time (min)')
            ax_survival.set_ylabel('Live Cells')
            ax_survival.set_xlim(0, max(times))
            ax_survival.set_ylim(0, max(survivals) * 1.1 if survivals else 1)
            ax_survival.grid(True, alpha=0.3)
            
            # Initialize count bars
            categories = ['Alive', 'Dying', 'Dead']
            counts = [frame0['alive'], frame0['dying'], frame0['dead']]
            colors = ['green', 'yellow', 'red']
            bars = ax_counts.bar(categories, counts, color=colors)
            ax_counts.set_ylabel('Cell Count')
            ax_counts.set_ylim(0, sum(counts) * 1.2 if sum(counts) > 0 else 5)
            
            # Cell markers
            cell_markers = []
            
            # Summary text
            summary = data['summary']
            summary_text = f"Initial: {summary['initial_cells']} | "
            summary_text += f"Deaths: {summary['total_deaths']} | "
            summary_text += f"Survival: {summary['survival_rate']:.1f}%"
            if summary['avg_death_time'] != 'N/A':
                summary_text += f" | Avg Death Time: {summary['avg_death_time']:.0f} min"
            
            info_text = ax_info.text(0.5, 0.5, summary_text, 
                                   transform=ax_info.transAxes,
                                   ha='center', va='center', fontsize=11,
                                   bbox=dict(boxstyle="round,pad=0.5", 
                                           facecolor='lightgray', alpha=0.8))
            
            def update_frame(frame_idx):
                """Update function for animation."""
                frame = data['frames'][frame_idx]
                
                # Update main image
                tritc = frame['tritc_crop']
                vmin, vmax = np.percentile(tritc[tritc > 0], [5, 99]) if np.any(tritc > 0) else (0, 1)
                tritc_enhanced = np.clip((tritc - vmin) / (vmax - vmin + 1e-8), 0, 1)
                
                tritc_colored = plt.cm.hot(tritc_enhanced)
                tritc_colored[:, :, 3] = 1
                
                img_main.set_data(tritc_colored)
                ax_main.set_title(f"Droplet {did} - T={frame['time_min']:.0f} min", fontsize=14)
                
                # Clear old markers
                for marker in cell_markers:
                    marker.remove()
                cell_markers.clear()
                
                # Add cell markers
                x0, y0, x1, y1 = frame['crop_coords']
                for i, cell in enumerate(frame['cells']):
                    if i in frame['cell_states']:
                        track_id, status = frame['cell_states'][i]
                        
                        # Adjust coordinates to crop
                        cx = cell['centroid_x'] - x0
                        cy = cell['centroid_y'] - y0
                        
                        if status == 'active':
                            marker = ax_main.plot(cx, cy, '+', color='white', 
                                                markersize=10, markeredgewidth=2)[0]
                        elif status in ['dying', 'missing']:
                            marker = ax_main.plot(cx, cy, 'x', color='yellow', 
                                                markersize=12, markeredgewidth=3)[0]
                            
                            # Add dying indicator
                            track = self.droplet_trackers[did].tracks[track_id]
                            if 'first_dying_frame' in track:
                                frames_dying = frame['timepoint'] - track['first_dying_frame']
                                text = ax_main.text(cx, cy - 15, f'd{frames_dying}', 
                                                  color='yellow', fontsize=8,
                                                  ha='center', va='center',
                                                  bbox=dict(boxstyle="round,pad=0.2", 
                                                          facecolor='black', alpha=0.7))
                                cell_markers.append(text)
                        
                        if marker:
                            cell_markers.append(marker)
                
                # Update survival curve
                survival_line.set_data(times[:frame_idx+1], survivals[:frame_idx+1])
                
                # Update count bars
                for bar, count in zip(bars, [frame['alive'], frame['dying'], frame['dead']]):
                    bar.set_height(count)
                
                # Update current stats
                current_text = f"Time: {frame['time_min']:.0f} min | "
                current_text += f"Alive: {frame['alive']} | "
                current_text += f"Dying: {frame['dying']} | "
                current_text += f"Dead: {frame['dead']}"
                
                fig.suptitle(current_text, fontsize=12)
                
                return [img_main] + cell_markers + [survival_line] + list(bars)
            
            # Create animation
            anim = animation.FuncAnimation(fig, update_frame, frames=len(data['frames']),
                                         interval=1000/fps, blit=False)
            
            # Save movie - MP4 only
            output_path = os.path.join(output_dir, f"{base_name}_droplet_{did}.mp4")
            ffmpeg_path = r"D:\ffmpeg-2025-06-28-git-cfd1f81e7d-full_build\bin\ffmpeg.exe"
            if not os.path.exists(ffmpeg_path):
                raise FileNotFoundError(f"FFmpeg not found at {ffmpeg_path}")
            
            # Temporarily add FFmpeg to PATH
            original_path = os.environ.get('PATH', '')
            os.environ['PATH'] = os.path.dirname(ffmpeg_path) + os.pathsep + original_path
            
            try:
                writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
                anim.save(output_path, writer=writer)
                print(f"MP4 movie saved: {output_path}")
            except Exception as e:
                print(f"FFMpegWriter failed: {e}")
                print("Please ensure FFmpeg is properly installed and accessible.")
                raise
            finally:
                # Restore original PATH
                os.environ['PATH'] = original_path
            
            plt.close()
            print(f"Saved: {output_path}")
    
    def export_results(self, output_path=None):
        """Export detailed results to Excel."""
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(self.nd2_file))[0]
            output_path = f"{base_name}_droplet_analysis.xlsx"
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_rows = []
            for did, data in self.droplet_data.items():
                summary = data['summary']
                summary['droplet_id'] = did
                summary['droplet_type'] = data['droplet_info']['type']
                summary_rows.append(summary)
            
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Individual droplet sheets
            for did, data in self.droplet_data.items():
                rows = []
                for frame in data['frames']:
                    rows.append({
                        'timepoint': frame['timepoint'],
                        'time_min': frame['time_min'],
                        'alive': frame['alive'],
                        'dying': frame['dying'],
                        'dead': frame['dead'],
                        'total': frame['alive'] + frame['dying']
                    })
                
                df = pd.DataFrame(rows)
                df.to_excel(writer, sheet_name=f'Droplet_{did}', index=False)
        
        print(f"Results exported to: {output_path}")


def analyze_droplets_focused(nd2_file, output_dir=None):
    """Run droplet-focused analysis."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(nd2_file), "droplet_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = DropletFocusedAnalyzer(nd2_file)
    
    if analyzer.analyze():
        # Create individual droplet movies
        analyzer.create_droplet_movies(output_dir, fps=3)
        
        # Export results
        excel_path = os.path.join(output_dir, 
                                os.path.splitext(os.path.basename(nd2_file))[0] + "_results.xlsx")
        analyzer.export_results(excel_path)
        
        # Close
        analyzer.analyzer.close()
        
        print(f"\nAnalysis complete! Results in: {output_dir}")
        return analyzer
    
    return None


if __name__ == "__main__":
    # Test
    nd2_file = r"D:\New\BrainBites\Cell\2.nd2"
    analyze_droplets_focused(nd2_file)