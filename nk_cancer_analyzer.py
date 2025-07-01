"""
NK Cell Cancer Analysis Tool - Phase 1: ND2 File Reading and Visualization
Author: AI Assistant
Date: 2025-01-20

This script provides basic functionality to read and visualize ND2 files
for analyzing cancer cell death by NK cells.
"""

import numpy as np
import matplotlib.pyplot as plt
from nd2reader import ND2Reader
import cv2
from pathlib import Path
import os
from scipy import ndimage

class ND2Analyzer:
    def __init__(self, file_path):
        """Initialize the analyzer with an ND2 file."""
        self.file_path = Path(file_path)
        self.reader = None
        self.metadata = {}
        
    def load_file(self):
        """Load the ND2 file and extract metadata."""
        try:
            # Convert Path to string for nd2reader
            self.reader = ND2Reader(str(self.file_path))
            
            # Extract metadata
            self.metadata = {
                'width': self.reader.metadata['width'],
                'height': self.reader.metadata['height'],
                'channels': self.reader.metadata['channels'],
                'frames': self.reader.metadata['total_images_per_channel'],
                'time_interval': 15,  # minutes, as specified
                'pixel_microns': self.reader.metadata.get('pixel_microns', None)
            }
            
            print(f"Successfully loaded: {self.file_path.name}")
            print(f"Dimensions: {self.metadata['width']} x {self.metadata['height']}")
            print(f"Channels: {self.metadata['channels']}")
            print(f"Time points: {self.metadata['frames']}")
            print(f"Total duration: {self.metadata['frames'] * self.metadata['time_interval']} minutes")
            
            return True
            
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def get_frame(self, timepoint=0, channel='brightfield'):
        """Get a specific frame from the ND2 file."""
        if not self.reader:
            print("No file loaded!")
            return None
            
        try:
            # Method 1: Try using iteration axes
            self.reader.iter_axes = 't'
            
            # Select channel
            channel_idx = 0 if channel.lower() == 'brightfield' else 1
            
            # Access frame directly by timepoint
            self.reader.default_coords['c'] = channel_idx
            frame = self.reader[timepoint]
            
            if frame is not None:
                return np.array(frame)
            
            # Method 2: If that doesn't work, try manual iteration
            print(f"Trying alternative method for t={timepoint}")
            for i, img in enumerate(self.reader):
                if i == timepoint:
                    return np.array(img)
            
            print(f"Could not retrieve frame at t={timepoint}, c={channel}")
            return None
            
        except Exception as e:
            print(f"Error getting frame: {e}")
            return None
    
    def visualize_timepoint(self, timepoint=0, save=False, tritc_percentile=99.5):
        """Visualize both channels at a specific timepoint."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Get brightfield image
        bf_frame = self.get_frame(timepoint, 'brightfield')
        if bf_frame is not None:
            axes[0].imshow(bf_frame, cmap='gray')
            axes[0].set_title(f'Brightfield - T={timepoint*15} min')
            axes[0].axis('off')
        
        # Get TRITC image
        tritc_frame = self.get_frame(timepoint, 'TRITC')
        if tritc_frame is not None:
            # Original TRITC (for comparison)
            axes[1].imshow(tritc_frame, cmap='hot')
            axes[1].set_title(f'TRITC Raw - T={timepoint*15} min')
            axes[1].axis('off')
            
            # Enhanced TRITC with proper contrast
            # Calculate percentile for better visualization
            vmin = np.percentile(tritc_frame, 5)
            vmax = np.percentile(tritc_frame, tritc_percentile)
            
            # Apply contrast adjustment
            tritc_enhanced = np.clip((tritc_frame - vmin) / (vmax - vmin), 0, 1)
            
            # Apply threshold to show only bright cells
            threshold = np.percentile(tritc_frame, 90)
            tritc_binary = tritc_frame > threshold
            
            # Show enhanced TRITC
            axes[2].imshow(tritc_enhanced, cmap='hot', vmin=0, vmax=1)
            axes[2].set_title(f'TRITC Enhanced (Cancer cells) - T={timepoint*15} min')
            axes[2].axis('off')
            
            # Add cell count
            num_objects = self.count_cells_simple(tritc_binary)
            axes[2].text(10, 30, f'Detected cells: ~{num_objects}', 
                        color='white', fontsize=12, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        
        if save:
            output_path = f"timepoint_{timepoint}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")
        
        plt.show()
    
    def count_cells_simple(self, binary_image):
        """Simple cell counting using connected components."""
        from scipy import ndimage
        labeled, num_features = ndimage.label(binary_image)
        return num_features
    
    def create_time_montage(self, channel='TRITC', num_timepoints=6, enhance=True):
        """Create a montage showing the progression over time."""
        if not self.reader:
            print("No file loaded!")
            return
        
        total_frames = self.metadata['frames']
        
        # Calculate which timepoints to show
        if num_timepoints > total_frames:
            num_timepoints = total_frames
        
        time_indices = np.linspace(0, total_frames-1, num_timepoints, dtype=int)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, t_idx in enumerate(time_indices):
            if i < len(axes):
                frame = self.get_frame(t_idx, channel)
                if frame is not None:
                    if channel == 'TRITC' and enhance:
                        # Apply contrast enhancement for TRITC
                        vmin = np.percentile(frame, 5)
                        vmax = np.percentile(frame, 99.5)
                        frame_enhanced = np.clip((frame - vmin) / (vmax - vmin), 0, 1)
                        axes[i].imshow(frame_enhanced, cmap='hot', vmin=0, vmax=1)
                        
                        # Count cells
                        threshold = np.percentile(frame, 90)
                        binary = frame > threshold
                        num_cells = self.count_cells_simple(binary)
                        axes[i].text(10, 30, f'Cells: ~{num_cells}', 
                                   color='white', fontsize=10,
                                   bbox=dict(boxstyle="round,pad=0.3", 
                                           facecolor='black', alpha=0.7))
                    else:
                        axes[i].imshow(frame, cmap='hot' if channel=='TRITC' else 'gray')
                    
                    axes[i].set_title(f'T = {t_idx*15} min')
                    axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(time_indices), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'{channel} Channel Time Series - {self.file_path.name}', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def close(self):
        """Close the ND2 reader."""
        if self.reader:
            self.reader.close()

# Example usage
def main():
    # Example usage - replace with your file path
    nd2_file = "path/to/your/file.nd2"
    
    # Initialize analyzer
    analyzer = ND2Analyzer(nd2_file)
    
    # Load the file
    if analyzer.load_file():
        # Visualize first timepoint
        analyzer.visualize_timepoint(timepoint=0)
        
        # Create time montage
        analyzer.create_time_montage(channel='TRITC', num_timepoints=6)
        
        # Don't forget to close
        analyzer.close()

if __name__ == "__main__":
    # You can run this directly or import the class
    print("ND2 Analyzer - Phase 1")
    print("="*50)
    print("To use this analyzer:")
    print("1. Install required packages: pip install nd2reader numpy matplotlib opencv-python")
    print("2. Create an analyzer: analyzer = ND2Analyzer('your_file.nd2')")
    print("3. Load the file: analyzer.load_file()")
    print("4. Visualize data: analyzer.visualize_timepoint(0)")