"""
Enhanced Cancer Cell Survival Analysis
Combines Phase 4's death detection with movie generation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches
import os
from datetime import datetime
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment

# Import our modules
from nk_cancer_analyzer import ND2Analyzer
from nk_cancer_analyzer_phase3 import DropletDetector, DropletCellAnalyzer
from nk_cancer_analyzer_phase4 import CellTracker, TimeSeriesAnalyzer

class EnhancedSurvivalAnalyzer(TimeSeriesAnalyzer):
    """Enhanced analyzer combining Phase 4 detection with movies."""
    
    def __init__(self, nd2_analyzer, time_interval_min=15):
        super().__init__(nd2_analyzer, time_interval_min)
        self.frame_cache = []  # Store frames for movie generation
        
    def analyze_full_time_series(self, output_dir=None):
        """Enhanced analysis that stores frames for movies."""
        print(f"Analyzing {self.analyzer.metadata['frames']} timepoints...")
        
        # Initialize detectors
        droplet_detector = DropletDetector()
        cell_analyzer = DropletCellAnalyzer(droplet_detector, None)
        
        # Initialize tracker (cancer cells only)
        cancer_tracker = CellTracker(max_distance=30, max_frames_missing=3)
        
        # Detect droplets in first frame
        bf_frame = self.analyzer.get_frame(0, 'brightfield')
        droplets = droplet_detector.detect_droplets(bf_frame)
        masks, combined_mask = droplet_detector.create_droplet_masks(bf_frame.shape, droplets)
        
        print(f"Tracking cells in {len(droplets)} droplets...")
        
        # Process each timepoint
        frame_results = []
        
        for t in range(self.analyzer.metadata['frames']):
            if t % 10 == 0:
                print(f"Processing frame {t}/{self.analyzer.metadata['frames']-1}")
            
            # Get frames
            bf_frame = self.analyzer.get_frame(t, 'brightfield')
            tritc_frame = self.analyzer.get_frame(t, 'TRITC')
            
            if bf_frame is None or tritc_frame is None:
                continue
            
            # Store frames for movie
            self.frame_cache.append({
                'timepoint': t,
                'brightfield': bf_frame,
                'tritc': tritc_frame
            })
            
            # Analyze each droplet
            droplet_frame_data = []
            
            for droplet in droplets:
                droplet_id = droplet['id']
                mask = masks[droplet_id]
                
                # Extract regions
                masked_tritc = tritc_frame.copy()
                masked_tritc[~mask] = 0
                masked_bf = bf_frame.copy()
                masked_bf[~mask] = 0
                
                # Detect cancer cells using nuclear detection
                nuclei = cell_analyzer._detect_nuclei(masked_tritc, masked_bf, mask, droplet)
                
                # Convert to detection format
                cancer_cells = []
                for nx, ny, intensity, area in nuclei:
                    cancer_cells.append({
                        'centroid_x': nx,
                        'centroid_y': ny,
                        'mean_intensity': intensity,
                        'area': area
                    })
                
                # Update cancer cell tracks
                cancer_assignments = cancer_tracker.update(cancer_cells, t, 'cancer')
                
                # Mark dead cells (Phase 4's sophisticated method)
                cancer_tracker.mark_dead_cells(t, intensity_threshold=0.3)
                
                # Get cancer cell states
                alive_cancer = 0
                dying_cancer = 0
                dead_cancer = 0
                
                # Track assignments for this frame
                cell_track_map = {}
                
                for i, cell in enumerate(cancer_cells):
                    if i in cancer_assignments:
                        track_id = cancer_assignments[i]
                        track = cancer_tracker.tracks[track_id]
                        cell_track_map[i] = (track_id, track['status'])
                        
                        if track['status'] == 'active':
                            alive_cancer += 1
                        elif track['status'] == 'dying':
                            dying_cancer += 1
                
                # Count total dead from tracks
                for tid, track in cancer_tracker.tracks.items():
                    if track['type'] == 'cancer' and track['status'] == 'dead':
                        # Check if death occurred in this droplet
                        if track['positions']:
                            last_x, last_y = track['positions'][-1]
                            dist = np.sqrt((last_x - droplet['center_x'])**2 + 
                                         (last_y - droplet['center_y'])**2)
                            if dist < droplet['radius_px']:
                                dead_cancer += 1
                
                droplet_frame_data.append({
                    'timepoint': t,
                    'time_min': t * self.time_interval,
                    'droplet_id': droplet_id,
                    'droplet_type': droplet['type'],
                    'cancer_cells_alive': alive_cancer,
                    'cancer_cells_dying': dying_cancer,
                    'cancer_cells_dead': dead_cancer,
                    'total_cancer_cells': alive_cancer + dying_cancer,
                    'detected_cells': cancer_cells,
                    'cell_tracks': cell_track_map
                })
            
            frame_results.extend(droplet_frame_data)
        
        # Store results
        self.droplets = droplets
        self.masks = masks
        self.cancer_tracker = cancer_tracker
        self.cell_analyzer = cell_analyzer
        
        # Create DataFrame
        df = pd.DataFrame(frame_results)
        
        # Add death timing information (Phase 4 method)
        death_events = []
        for tid, track in cancer_tracker.tracks.items():
            if track['type'] == 'cancer' and track['death_frame'] is not None:
                # Find droplet
                last_x, last_y = track['positions'][-1]
                droplet_id = None
                for droplet in droplets:
                    dist = np.sqrt((last_x - droplet['center_x'])**2 + 
                                 (last_y - droplet['center_y'])**2)
                    if dist < droplet['radius_px']:
                        droplet_id = droplet['id']
                        break
                
                death_events.append({
                    'cell_id': tid,
                    'droplet_id': droplet_id,
                    'death_frame': track['death_frame'],
                    'death_time_min': track['death_frame'] * self.time_interval,
                    'lifespan_frames': track['death_frame'] - track['first_frame'],
                    'lifespan_min': (track['death_frame'] - track['first_frame']) * self.time_interval,
                    'max_intensity': max(track['intensities']),
                    'final_intensity': track['intensities'][-1] if track['intensities'] else 0
                })
        
        death_df = pd.DataFrame(death_events)
        
        # Save results
        if output_dir:
            self._save_results(df, death_df, output_dir)
        
        return df, death_df, cancer_tracker
    
    def create_enhanced_movie(self, time_series_df, output_path=None, fps=3):
        """Create enhanced time-lapse movie with death tracking."""
        if not self.frame_cache:
            print("No frames cached for movie!")
            return
        
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(self.analyzer.file_path.name))[0]
            output_path = f"{base_name}_enhanced_timelapse.mp4"
        
        print(f"Creating enhanced time-lapse movie: {output_path}")
        
        # Setup figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Initialize plots
        frame0 = self.frame_cache[0]
        
        # Brightfield
        bf_img = axes[0].imshow(frame0['brightfield'], cmap='gray')
        axes[0].set_title('Brightfield - T=0 min')
        axes[0].axis('off')
        
        # TRITC
        tritc = frame0['tritc']
        vmin, vmax = np.percentile(tritc[tritc > 0], [5, 99.5]) if np.any(tritc > 0) else (0, 1)
        tritc_enhanced = np.clip((tritc - vmin) / (vmax - vmin + 1e-8), 0, 1)
        tritc_img = axes[1].imshow(tritc_enhanced, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title('Cancer Cells (mKate2) - T=0 min')
        axes[1].axis('off')
        
        # Draw droplet circles
        for droplet in self.droplets:
            for ax in axes:
                circle = plt.Circle((droplet['center_x'], droplet['center_y']), 
                                  droplet['radius_px'], 
                                  color='lime', fill=False, linewidth=2)
                ax.add_patch(circle)
        
        # Track specific death events
        death_annotations = []
        
        # Info text
        info_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12, 
                            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        def update_frame(frame_idx):
            """Update function for animation."""
            frame = self.frame_cache[frame_idx]
            timepoint = frame['timepoint']
            
            # Update images
            bf_img.set_data(frame['brightfield'])
            
            # Enhance TRITC
            tritc = frame['tritc']
            vmin, vmax = np.percentile(tritc[tritc > 0], [5, 99.5]) if np.any(tritc > 0) else (0, 1)
            tritc_enhanced = np.clip((tritc - vmin) / (vmax - vmin + 1e-8), 0, 1)
            tritc_img.set_data(tritc_enhanced)
            
            # Update titles
            time_min = timepoint * self.time_interval
            axes[0].set_title(f'Brightfield - T={time_min:.0f} min')
            axes[1].set_title(f'Cancer Cells (mKate2) - T={time_min:.0f} min')
            
            # Clear old annotations
            for ann in death_annotations:
                ann.remove()
            death_annotations.clear()
            
            # Get frame data
            frame_data = time_series_df[time_series_df['timepoint'] == timepoint]
            
            total_alive = 0
            total_dying = 0
            total_dead = 0
            
            # Mark cells
            for _, row in frame_data.iterrows():
                if 'detected_cells' in row and row['detected_cells']:
                    for i, cell in enumerate(row['detected_cells']):
                        if 'cell_tracks' in row and i in row['cell_tracks']:
                            track_id, status = row['cell_tracks'][i]
                            
                            if status == 'active':
                                marker = axes[1].plot(cell['centroid_x'], cell['centroid_y'], 
                                                    '+', color='white', markersize=8, 
                                                    markeredgewidth=2)[0]
                                death_annotations.append(marker)
                            elif status == 'dying':
                                marker = axes[1].plot(cell['centroid_x'], cell['centroid_y'], 
                                                    'x', color='yellow', markersize=10, 
                                                    markeredgewidth=3)[0]
                                death_annotations.append(marker)
                                
                                # Add death timer
                                track = self.cancer_tracker.tracks[track_id]
                                frames_dying = timepoint - track.get('first_dying_frame', timepoint)
                                text = axes[1].text(cell['centroid_x'], cell['centroid_y'] - 10, 
                                                  f'{frames_dying}', color='yellow', 
                                                  fontsize=8, ha='center',
                                                  bbox=dict(boxstyle="round,pad=0.2", 
                                                          facecolor='black', alpha=0.7))
                                death_annotations.append(text)
                
                total_alive += row['cancer_cells_alive']
                total_dying += row['cancer_cells_dying']
                total_dead += row['cancer_cells_dead']
            
            # Update info
            info_str = f"Time: {time_min:.0f} min | "
            info_str += f"Alive: {total_alive} | "
            info_str += f"Dying: {total_dying} | "
            info_str += f"Dead: {total_dead} | "
            if total_alive + total_dead > 0:
                survival_pct = (total_alive / (total_alive + total_dead)) * 100
                info_str += f"Survival: {survival_pct:.1f}%"
            
            info_text.set_text(info_str)
            
            return [bf_img, tritc_img, info_text] + death_annotations
        
        # Create animation
        anim = animation.FuncAnimation(fig, update_frame, frames=len(self.frame_cache),
                                     interval=1000/fps, blit=False)
        
        # Save movie - MP4 only
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
        print(f"Movie saved: {output_path}")
    
    def _save_results(self, df, death_df, output_dir):
        """Save analysis results to files."""
        base_name = os.path.splitext(os.path.basename(self.analyzer.file_path.name))[0]
        
        # Save time series data
        excel_path = os.path.join(output_dir, f"{base_name}_enhanced_survival_data.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Time_Series', index=False)
            if len(death_df) > 0:
                death_df.to_excel(writer, sheet_name='Death_Events', index=False)
            
            # Add summary
            summary = {
                'total_droplets': len(self.droplets),
                'initial_cells': df[df['timepoint'] == 0]['cancer_cells_alive'].sum(),
                'final_cells': df[df['timepoint'] == df['timepoint'].max()]['cancer_cells_alive'].sum(),
                'total_deaths': len(death_df),
                'avg_death_time_min': death_df['death_time_min'].mean() if len(death_df) > 0 else 'N/A',
                'experiment_duration_min': df['time_min'].max()
            }
            pd.DataFrame([summary]).to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"Data exported to: {excel_path}")
    
    def create_killing_visualization(self, time_series_df, output_path=None):
        """Create survival curves visualization."""
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(self.analyzer.file_path.name))[0]
            output_path = f"{base_name}_survival_analysis.png"
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Collect survival data
        time_points = []
        survival_by_droplet = {d['id']: [] for d in self.droplets}
        total_survival = []
        
        for timepoint in sorted(time_series_df['timepoint'].unique()):
            time_points.append(timepoint * self.time_interval)
            frame_data = time_series_df[time_series_df['timepoint'] == timepoint]
            total_alive = 0
            
            for droplet in self.droplets:
                did = droplet['id']
                droplet_data = frame_data[frame_data['droplet_id'] == did]
                if len(droplet_data) > 0:
                    alive = droplet_data.iloc[0]['cancer_cells_alive']
                    survival_by_droplet[did].append(alive)
                    total_alive += alive
                else:
                    survival_by_droplet[did].append(0)
            
            total_survival.append(total_alive)
        
        # Plot individual droplets
        for did, survival in survival_by_droplet.items():
            if max(survival) > 0:  # Only plot droplets that had cells
                ax1.plot(time_points, survival, marker='o', markersize=4, 
                        label=f'Droplet {did}')
        
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Number of Live Cells')
        ax1.set_title('Cancer Cell Survival by Droplet')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot total survival
        ax2.plot(time_points, total_survival, 'b-', linewidth=3, marker='o')
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Total Live Cells')
        ax2.set_title('Total Cancer Cell Survival')
        ax2.grid(True, alpha=0.3)
        
        # Add percentage survival on right y-axis
        ax2_pct = ax2.twinx()
        if total_survival[0] > 0:
            pct_survival = [n/total_survival[0] * 100 for n in total_survival]
            ax2_pct.plot(time_points, pct_survival, 'r--', alpha=0.7)
            ax2_pct.set_ylabel('Survival (%)', color='r')
            ax2_pct.tick_params(axis='y', labelcolor='r')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()


def analyze_cancer_survival_enhanced(nd2_file, output_dir=None):
    """Run enhanced cancer survival analysis."""
    print(f"Starting enhanced survival analysis for: {nd2_file}")
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(nd2_file), "survival_analysis_enhanced")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ND2 file
    analyzer = ND2Analyzer(nd2_file)
    if not analyzer.load_file():
        print("Failed to load ND2 file!")
        return None
    
    # Run enhanced analysis
    enhanced_analyzer = EnhancedSurvivalAnalyzer(analyzer)
    time_series_df, death_events_df, cancer_tracker = enhanced_analyzer.analyze_full_time_series(output_dir)
    
    # Create movie
    base_name = os.path.splitext(os.path.basename(nd2_file))[0]
    movie_path = os.path.join(output_dir, f"{base_name}_enhanced_timelapse.mp4")
    enhanced_analyzer.create_enhanced_movie(time_series_df, movie_path)
    
    # Create survival visualization
    vis_path = os.path.join(output_dir, f"{base_name}_survival_analysis.png")
    enhanced_analyzer.create_killing_visualization(time_series_df, vis_path)
    
    # Close analyzer
    analyzer.close()
    
    print("\nEnhanced analysis complete!")
    print(f"Results saved to: {output_dir}")
    
    return {
        'time_series': time_series_df,
        'death_events': death_events_df,
        'cancer_tracks': cancer_tracker.tracks
    }


if __name__ == "__main__":
    # Test on your file
    nd2_file = r"D:\New\BrainBites\Cell\2.nd2"
    analyze_cancer_survival_enhanced(nd2_file)