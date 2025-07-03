"""
All Droplets Cancer Analysis with Optimized Death Detection
Shows all droplets in one video with individual statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches
from matplotlib.gridspec import GridSpec
import os
from datetime import datetime
from scipy.ndimage import gaussian_filter

from nk_cancer_analyzer import ND2Analyzer
from nk_cancer_analyzer_phase3 import DropletDetector, DropletCellAnalyzer
from nk_cancer_analyzer_phase4 import CellTracker

class OptimizedCellTracker(CellTracker):
    """Optimized tracker with better death detection."""
    
    def __init__(self, max_distance=30, max_frames_missing=3):
        super().__init__(max_distance, max_frames_missing)
        self.intensity_history_window = 5  # Frames to average for baseline
        
    def mark_dead_cells_optimized(self, frame_num, intensity_threshold=0.4):
        """
        Optimized death detection based on:
        1. Sustained intensity drop (not just temporary)
        2. Comparison to cell's own baseline (not global)
        3. Confirmation over multiple frames
        """
        for tid, track in self.tracks.items():
            if track['type'] == 'cancer':
                # Handle missing cells
                if frame_num - track['last_seen'] > self.max_frames_missing:
                    if track['status'] == 'active':
                        track['status'] = 'dead'
                        track['death_frame'] = track['last_seen'] + 1
                    continue
                
                # For active cells, check intensity patterns
                if track['status'] == 'active' and len(track['intensities']) >= 3:
                    intensities = track['intensities']
                    
                    # Calculate baseline from early frames (when cell was healthy)
                    baseline_end = min(self.intensity_history_window, len(intensities) // 2)
                    baseline = np.mean(intensities[:baseline_end])
                    
                    # Get recent intensities
                    recent = intensities[-3:]
                    current = recent[-1]
                    
                    # Check for sustained drop
                    if all(i < baseline * intensity_threshold for i in recent):
                        track['status'] = 'dying'
                        if 'first_dying_frame' not in track:
                            track['first_dying_frame'] = frame_num - 2
                
                # Confirm death after sustained dying
                elif track['status'] == 'dying':
                    # Check if intensity continues to drop or stays very low
                    if len(track['intensities']) > 0:
                        current = track['intensities'][-1]
                        baseline = np.mean(track['intensities'][:self.intensity_history_window])
                        
                        # Death confirmed if:
                        # 1. Very low intensity for 3+ frames
                        # 2. Or continued decline
                        frames_dying = frame_num - track.get('first_dying_frame', frame_num)
                        
                        if frames_dying >= 3 and current < baseline * 0.2:
                            track['status'] = 'dead'
                            track['death_frame'] = frame_num


class AllDropletsAnalyzer:
    """Analyzer showing all droplets in one comprehensive view."""
    
    def __init__(self, nd2_file, time_interval_min=15):
        self.nd2_file = nd2_file
        self.time_interval = time_interval_min
        self.analyzer = None
        self.droplets = None
        self.masks = None
        self.tracker = None
        self.frame_data = []
        self.droplet_positions = {}
        
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
        
        # Initialize unified tracker
        self.tracker = OptimizedCellTracker(max_distance=30, max_frames_missing=3)
        
        # Store droplet layout info
        self._calculate_droplet_layout()
        
        # Analyze all frames
        cell_analyzer = DropletCellAnalyzer(droplet_detector, None)
        
        for t in range(self.analyzer.metadata['frames']):
            if t % 10 == 0:
                print(f"Processing frame {t}/{self.analyzer.metadata['frames']-1}")
            
            frame_result = self._process_frame(t, cell_analyzer)
            self.frame_data.append(frame_result)
        
        print("Analysis complete!")
        return True
    
    def _calculate_droplet_layout(self):
        """Calculate optimal layout for displaying all droplets."""
        # Sort droplets by position for consistent layout
        sorted_droplets = sorted(self.droplets, key=lambda d: (d['center_y'], d['center_x']))
        
        for i, droplet in enumerate(sorted_droplets):
            self.droplet_positions[droplet['id']] = i
    
    def _process_frame(self, timepoint, cell_analyzer):
        """Process single frame for all droplets."""
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
        
        # Process each droplet
        for droplet in self.droplets:
            did = droplet['id']
            mask = self.masks[did]
            
            # Extract masked regions
            masked_tritc = tritc_frame.copy()
            masked_tritc[~mask] = 0
            masked_bf = bf_frame.copy()
            masked_bf[~mask] = 0
            
            # Detect cells
            nuclei = cell_analyzer._detect_nuclei(masked_tritc, masked_bf, mask, droplet)
            
            # Convert to tracking format with droplet ID
            cancer_cells = []
            for nx, ny, intensity, area in nuclei:
                cell = {
                    'centroid_x': nx,
                    'centroid_y': ny,
                    'mean_intensity': intensity,
                    'area': area,
                    'droplet_id': did
                }
                cancer_cells.append(cell)
                all_cancer_cells.append(cell)
            
            # Store droplet-specific data
            frame_result['droplet_data'][did] = {
                'cells': cancer_cells,
                'mask': mask
            }
        
        # Update global tracker
        assignments = self.tracker.update(all_cancer_cells, timepoint, 'cancer')
        self.tracker.mark_dead_cells_optimized(timepoint)
        
        # Assign track IDs back to droplets
        for i, cell in enumerate(all_cancer_cells):
            did = cell['droplet_id']
            if i in assignments:
                track_id = assignments[i]
                track = self.tracker.tracks[track_id]
                
                # Find this cell in the droplet data
                for j, dcell in enumerate(frame_result['droplet_data'][did]['cells']):
                    if (dcell['centroid_x'] == cell['centroid_x'] and 
                        dcell['centroid_y'] == cell['centroid_y']):
                        dcell['track_id'] = track_id
                        dcell['status'] = track['status']
                        break
        
        # Calculate statistics per droplet
        for did in frame_result['droplet_data']:
            cells = frame_result['droplet_data'][did]['cells']
            
            alive = sum(1 for c in cells if c.get('status') == 'active')
            dying = sum(1 for c in cells if c.get('status') == 'dying')
            
            # Count dead cells that belonged to this droplet
            dead = 0
            for tid, track in self.tracker.tracks.items():
                if track['status'] == 'dead' and track['positions']:
                    last_x, last_y = track['positions'][-1]
                    # Check if last position was in this droplet
                    if self.masks[did][int(last_y), int(last_x)]:
                        dead += 1
            
            frame_result['droplet_data'][did]['stats'] = {
                'alive': alive,
                'dying': dying,
                'dead': dead,
                'total': alive + dying
            }
        
        return frame_result
    
    def create_comprehensive_movie(self, output_path=None, fps=3):
        """Create movie showing all droplets with individual statistics."""
        if not self.frame_data:
            return
        
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(self.nd2_file))[0]
            output_path = f"{base_name}_all_droplets.mp4"
        
        print(f"Creating comprehensive movie: {output_path}")
        
        # Calculate grid layout
        n_droplets = len(self.droplets)
        cols = int(np.ceil(np.sqrt(n_droplets * 1.5)))  # Wider aspect ratio
        rows = int(np.ceil(n_droplets / cols))
        
        # Create figure
        fig = plt.figure(figsize=(cols * 4, rows * 4 + 2))
        gs = GridSpec(rows + 1, cols, figure=fig, height_ratios=[1]*rows + [0.3])
        
        # Create subplot for each droplet
        droplet_axes = {}
        droplet_images = {}
        droplet_info_texts = {}
        cell_markers = {}
        
        for droplet in self.droplets:
            did = droplet['id']
            pos = self.droplet_positions[did]
            row = pos // cols
            col = pos % cols
            
            ax = fig.add_subplot(gs[row, col])
            droplet_axes[did] = ax
            
            # Initialize with first frame
            frame0 = self.frame_data[0]
            
            # Get droplet region
            mask = frame0['droplet_data'][did]['mask']
            y_coords, x_coords = np.where(mask)
            
            if len(y_coords) > 0:
                margin = 20
                y_min = max(0, y_coords.min() - margin)
                y_max = min(frame0['tritc'].shape[0], y_coords.max() + margin)
                x_min = max(0, x_coords.min() - margin)
                x_max = min(frame0['tritc'].shape[1], x_coords.max() + margin)
                
                # Crop and enhance
                tritc_crop = frame0['tritc'][y_min:y_max, x_min:x_max]
                mask_crop = mask[y_min:y_max, x_min:x_max]
                
                # Apply mask and enhance
                masked_tritc = tritc_crop.copy()
                masked_tritc[~mask_crop] = 0
                
                # Enhance contrast
                if np.any(masked_tritc > 0):
                    vmin = np.percentile(masked_tritc[masked_tritc > 0], 5)
                    vmax = np.percentile(masked_tritc[masked_tritc > 0], 99)
                    enhanced = np.clip((masked_tritc - vmin) / (vmax - vmin + 1e-8), 0, 1)
                else:
                    enhanced = masked_tritc
                
                # Store crop info
                droplet_axes[did].crop_info = (x_min, y_min, x_max, y_max)
                
                # Display
                img = ax.imshow(enhanced, cmap='hot', vmin=0, vmax=1)
                droplet_images[did] = img
                
                # Draw droplet circle
                cx = droplet['center_x'] - x_min
                cy = droplet['center_y'] - y_min
                circle = plt.Circle((cx, cy), droplet['radius_px'], 
                                  color='lime', fill=False, linewidth=1.5)
                ax.add_patch(circle)
                
                # Title
                ax.set_title(f"Droplet {did} ({droplet['type']})", fontsize=10, pad=2)
                ax.axis('off')
                
                # Info text below image
                info_text = ax.text(0.5, -0.1, '', transform=ax.transAxes,
                                  ha='center', va='top', fontsize=8)
                droplet_info_texts[did] = info_text
                
                # Initialize cell markers list
                cell_markers[did] = []
        
        # Global info at bottom
        ax_info = fig.add_subplot(gs[-1, :])
        ax_info.axis('off')
        global_info = ax_info.text(0.5, 0.5, '', transform=ax_info.transAxes,
                                  ha='center', va='center', fontsize=12,
                                  bbox=dict(boxstyle="round,pad=0.5", 
                                          facecolor='lightgray', alpha=0.8))
        
        # Animation function
        def update_frame(frame_idx):
            frame = self.frame_data[frame_idx]
            time_min = frame['time_min']
            
            # Global statistics
            total_alive = 0
            total_dying = 0
            total_dead = 0
            
            # Update each droplet
            for droplet in self.droplets:
                did = droplet['id']  # Get the droplet ID
                
                if did not in frame['droplet_data']:
                    continue
                
                # Get crop info
                if hasattr(droplet_axes[did], 'crop_info'):
                    x_min, y_min, x_max, y_max = droplet_axes[did].crop_info
                    
                    # Update image
                    tritc_crop = frame['tritc'][y_min:y_max, x_min:x_max]
                    mask_crop = self.masks[did][y_min:y_max, x_min:x_max]
                    
                    masked_tritc = tritc_crop.copy()
                    masked_tritc[~mask_crop] = 0
                    
                    # Enhance
                    if np.any(masked_tritc > 0):
                        vmin = np.percentile(masked_tritc[masked_tritc > 0], 5)
                        vmax = np.percentile(masked_tritc[masked_tritc > 0], 99)
                        enhanced = np.clip((masked_tritc - vmin) / (vmax - vmin + 1e-8), 0, 1)
                    else:
                        enhanced = masked_tritc
                    
                    droplet_images[did].set_data(enhanced)
                    
                    # Clear old markers
                    for marker in cell_markers[did]:
                        marker.remove()
                    cell_markers[did].clear()
                    
                    # Add cell markers
                    cells = frame['droplet_data'][did]['cells']
                    for cell in cells:
                        if 'status' in cell:
                            cx = cell['centroid_x'] - x_min
                            cy = cell['centroid_y'] - y_min
                            
                            if cell['status'] == 'active':
                                marker = droplet_axes[did].plot(cx, cy, '+', 
                                                               color='white', 
                                                               markersize=6,
                                                               markeredgewidth=1.5)[0]
                                cell_markers[did].append(marker)
                            elif cell['status'] == 'dying':
                                marker = droplet_axes[did].plot(cx, cy, 'x', 
                                                               color='yellow', 
                                                               markersize=7,
                                                               markeredgewidth=2)[0]
                                cell_markers[did].append(marker)
                                
                                # Show dying duration
                                if 'track_id' in cell:
                                    track = self.tracker.tracks[cell['track_id']]
                                    if 'first_dying_frame' in track:
                                        dying_frames = frame['timepoint'] - track['first_dying_frame']
                                        text = droplet_axes[did].text(cx, cy - 8, 
                                                                     f'{dying_frames}',
                                                                     color='yellow', 
                                                                     fontsize=6,
                                                                     ha='center')
                                        cell_markers[did].append(text)
                    
                    # Update droplet info
                    stats = frame['droplet_data'][did]['stats']
                    info = f"A:{stats['alive']} D:{stats['dying']} †:{stats['dead']}"
                    droplet_info_texts[did].set_text(info)
                    
                    # Color code based on status
                    if stats['alive'] == 0 and stats['total'] == 0:
                        droplet_info_texts[did].set_color('red')
                    elif stats['dying'] > 0:
                        droplet_info_texts[did].set_color('orange')
                    else:
                        droplet_info_texts[did].set_color('green')
                    
                    # Accumulate totals
                    total_alive += stats['alive']
                    total_dying += stats['dying']
                    total_dead += stats['dead']
            
            # Update global info
            initial_total = sum(self.frame_data[0]['droplet_data'][d]['stats']['total'] 
                              for d in self.frame_data[0]['droplet_data'])
            
            survival_rate = (total_alive / initial_total * 100) if initial_total > 0 else 0
            
            global_text = f"Time: {time_min:.0f} min | "
            global_text += f"Total Alive: {total_alive} | "
            global_text += f"Dying: {total_dying} | "
            global_text += f"Dead: {total_dead} | "
            global_text += f"Survival: {survival_rate:.1f}%"
            
            global_info.set_text(global_text)
            
            # Update main title
            fig.suptitle(f'All Droplets Analysis - T = {time_min:.0f} minutes', 
                        fontsize=14, y=0.98)
            
            return [img for img in droplet_images.values()] + \
                   [t for t in droplet_info_texts.values()] + \
                   [m for markers in cell_markers.values() for m in markers] + \
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
    
    def export_optimized_results(self, output_path=None):
        """Export comprehensive results with death timing."""
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(self.nd2_file))[0]
            output_path = f"{base_name}_comprehensive_analysis.xlsx"
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Time series data
            rows = []
            for frame in self.frame_data:
                for did, data in frame['droplet_data'].items():
                    if 'stats' in data:
                        rows.append({
                            'timepoint': frame['timepoint'],
                            'time_min': frame['time_min'],
                            'droplet_id': did,
                            'alive': data['stats']['alive'],
                            'dying': data['stats']['dying'],
                            'dead': data['stats']['dead'],
                            'total': data['stats']['total']
                        })
            
            df_timeseries = pd.DataFrame(rows)
            df_timeseries.to_excel(writer, sheet_name='Time_Series', index=False)
            
            # Death events
            death_rows = []
            for tid, track in self.tracker.tracks.items():
                if track['death_frame'] is not None:
                    # Find droplet
                    last_x, last_y = track['positions'][-1]
                    droplet_id = None
                    for did, mask in self.masks.items():
                        if mask[int(last_y), int(last_x)]:
                            droplet_id = did
                            break
                    
                    death_rows.append({
                        'cell_id': tid,
                        'droplet_id': droplet_id,
                        'death_frame': track['death_frame'],
                        'death_time_min': track['death_frame'] * self.time_interval,
                        'first_detected': track['first_frame'],
                        'lifespan_min': (track['death_frame'] - track['first_frame']) * self.time_interval,
                        'baseline_intensity': np.mean(track['intensities'][:5]) if len(track['intensities']) >= 5 else track['intensities'][0],
                        'final_intensity': track['intensities'][-1] if track['intensities'] else 0
                    })
            
            if death_rows:
                df_deaths = pd.DataFrame(death_rows)
                df_deaths.to_excel(writer, sheet_name='Death_Events', index=False)
            
            # Summary by droplet
            summary_rows = []
            for did in self.masks.keys():
                # Get initial and final stats
                initial = self.frame_data[0]['droplet_data'].get(did, {}).get('stats', {})
                final = self.frame_data[-1]['droplet_data'].get(did, {}).get('stats', {})
                
                # Count deaths in this droplet
                droplet_deaths = [d for d in death_rows if d.get('droplet_id') == did]
                
                summary_rows.append({
                    'droplet_id': did,
                    'initial_cells': initial.get('total', 0),
                    'final_alive': final.get('alive', 0),
                    'total_deaths': len(droplet_deaths),
                    'survival_rate_%': (final.get('alive', 0) / initial.get('total', 1) * 100) if initial.get('total', 0) > 0 else 0,
                    'avg_death_time_min': np.mean([d['death_time_min'] for d in droplet_deaths]) if droplet_deaths else 'N/A',
                    'first_death_min': min([d['death_time_min'] for d in droplet_deaths]) if droplet_deaths else 'N/A',
                    'last_death_min': max([d['death_time_min'] for d in droplet_deaths]) if droplet_deaths else 'N/A'
                })
            
            df_summary = pd.DataFrame(summary_rows)
            df_summary.to_excel(writer, sheet_name='Droplet_Summary', index=False)
        
        print(f"Results exported to: {output_path}")


def analyze_all_droplets(nd2_file, output_dir=None):
    """Run comprehensive all-droplets analysis."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(nd2_file), "all_droplets_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = AllDropletsAnalyzer(nd2_file)
    
    if analyzer.analyze():
        # Create comprehensive movie
        movie_path = os.path.join(output_dir, 
                                os.path.splitext(os.path.basename(nd2_file))[0] + "_all_droplets.mp4")
        analyzer.create_comprehensive_movie(movie_path, fps=3)
        
        # Export results
        excel_path = os.path.join(output_dir,
                                os.path.splitext(os.path.basename(nd2_file))[0] + "_results.xlsx")
        analyzer.export_optimized_results(excel_path)
        
        # Close
        analyzer.analyzer.close()
        
        print(f"\nAnalysis complete! Results in: {output_dir}")
        return analyzer
    
    return None


if __name__ == "__main__":
    # Test
    nd2_file = r"D:\New\BrainBites\Cell\2.nd2"
    analyze_all_droplets(nd2_file)