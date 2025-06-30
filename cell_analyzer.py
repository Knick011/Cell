import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import os
import pandas as pd
from scipy import ndimage
from skimage import feature, filters, morphology, measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

try:
    from nd2reader import ND2Reader
    ND2_AVAILABLE = True
except ImportError:
    ND2_AVAILABLE = False
    print("nd2reader not installed. Install with: pip install nd2reader")

class CellAnalyzer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Cell Analysis - Enhanced Droplet & Cancer Cell Detection")
        self.root.geometry("1400x900")
        self.root.state('zoomed')  # Maximize window on Windows
        
        # Configure larger fonts
        self.large_font = ('Arial', 12, 'bold')
        self.medium_font = ('Arial', 10)
        self.button_font = ('Arial', 11, 'bold')
        
        # Data storage
        self.nd2_file = None
        self.current_timepoint = 0
        self.total_timepoints = 0
        self.channels = []
        self.current_channel = 0
        self.detected_droplets = []
        self.cell_data = {}
        
        self.setup_ui()
        
    def setup_ui(self):
        # Configure styles for larger elements
        style = ttk.Style()
        style.configure('Large.TButton', font=self.button_font, padding=(10, 8))
        style.configure('Large.TLabel', font=self.large_font)
        style.configure('Medium.TLabel', font=self.medium_font)
        
        # Main frame with larger padding
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Control panel
        control_frame = ttk.Frame(main_frame, height=80)
        control_frame.pack(fill=tk.X, pady=(0, 15))
        control_frame.pack_propagate(False)
        
        # File loading button
        load_btn = ttk.Button(control_frame, text="📁 Load ND2 File", 
                             command=self.load_nd2_file, style='Large.TButton')
        load_btn.pack(side=tk.LEFT, padx=(0, 15), pady=10)
        
        # File info label
        self.file_info_label = ttk.Label(control_frame, text="No file loaded", 
                                        style='Large.TLabel', foreground='blue')
        self.file_info_label.pack(side=tk.LEFT, padx=(0, 30), pady=10)
        
        # Analyze button
        self.analyze_btn = ttk.Button(control_frame, text="🔍 Analyze Current Frame", 
                                     command=self.analyze_current_frame, style='Large.TButton')
        self.analyze_btn.pack(side=tk.LEFT, padx=(0, 15), pady=10)
        self.analyze_btn.configure(state="disabled")
        
        # Export button
        self.export_btn = ttk.Button(control_frame, text="💾 Export Results", 
                                    command=self.export_results, style='Large.TButton')
        self.export_btn.pack(side=tk.LEFT, padx=(0, 15), pady=10)
        self.export_btn.configure(state="disabled")
        
        # Navigation frame
        nav_frame = ttk.Frame(main_frame, height=60)
        nav_frame.pack(fill=tk.X, pady=(0, 15))
        nav_frame.pack_propagate(False)
        
        # Timepoint controls
        ttk.Label(nav_frame, text="Timepoint:", style='Large.TLabel').pack(side=tk.LEFT, pady=15)
        
        self.timepoint_scale = ttk.Scale(nav_frame, from_=0, to=40, orient=tk.HORIZONTAL, 
                                        command=self.on_timepoint_change, length=300)
        self.timepoint_scale.pack(side=tk.LEFT, padx=(10, 15), pady=15, fill=tk.X, expand=True)
        
        self.timepoint_label = ttk.Label(nav_frame, text="0 / 0", style='Large.TLabel', 
                                        foreground='red', width=10)
        self.timepoint_label.pack(side=tk.LEFT, padx=(0, 30), pady=15)
        
        # Channel controls
        ttk.Label(nav_frame, text="Channel:", style='Large.TLabel').pack(side=tk.LEFT, pady=15)
        self.channel_combo = ttk.Combobox(nav_frame, state="readonly", width=20, 
                                         font=self.medium_font, height=8)
        self.channel_combo.pack(side=tk.LEFT, padx=(10, 15), pady=15)
        self.channel_combo.bind('<<ComboboxSelected>>', self.on_channel_change)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", 
                                      style='Large.TLabel', padding=15)
        results_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.results_text = tk.Text(results_frame, height=4, font=self.medium_font, 
                                   wrap=tk.WORD, bg='#f0f0f0')
        self.results_text.pack(fill=tk.X)
        
        # Image display frame
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(14, 8))
        self.fig.suptitle("Cell Analysis Dashboard", fontsize=16, fontweight='bold')
        
        # Embed plot in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, image_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_label = ttk.Label(main_frame, text="Ready - Load an ND2 file to begin", 
                                     style='Large.TLabel', foreground='green')
        self.status_label.pack(fill=tk.X, pady=(15, 0))
        
        # Disable controls initially
        self.timepoint_scale.configure(state="disabled")
        self.channel_combo.configure(state="disabled")
        
    def load_nd2_file(self):
        if not ND2_AVAILABLE:
            messagebox.showerror("Error", "nd2reader not installed.\nRun: pip install nd2reader")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select ND2 file",
            filetypes=[("ND2 files", "*.nd2"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # Load ND2 file
            self.status_label.config(text="Loading ND2 file...", foreground='orange')
            self.root.update()
            
            if self.nd2_file:
                self.nd2_file.close()
                
            self.nd2_file = ND2Reader(file_path)
            
            # Get file information
            self.total_timepoints = len(self.nd2_file)
            self.channels = list(self.nd2_file.metadata['channels'])
            
            # Update UI
            filename = os.path.basename(file_path)
            self.file_info_label.config(text=f"📄 {filename}")
            
            # Setup controls
            self.timepoint_scale.configure(from_=0, to=self.total_timepoints-1, state="normal")
            self.timepoint_label.config(text=f"1 / {self.total_timepoints}")
            
            self.channel_combo['values'] = self.channels
            self.channel_combo.current(0)
            self.channel_combo.configure(state="readonly")
            
            # Enable buttons
            self.analyze_btn.configure(state="normal")
            
            # Load first image
            self.current_timepoint = 0
            self.current_channel = 0
            self.update_display()
            
            self.status_label.config(text=f"✅ Loaded: {self.total_timepoints} timepoints, Channels: {', '.join(self.channels)}", 
                                   foreground='green')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load ND2 file:\n{str(e)}")
            self.status_label.config(text="❌ Error loading file", foreground='red')
    
    def detect_droplets_enhanced(self, image):
        """Enhanced droplet detection for both thick and thin rings"""
        try:
            # Normalize image
            if image.dtype != np.uint8:
                image_normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            else:
                image_normalized = image.copy()
            
            # Apply multiple preprocessing techniques
            droplets = []
            
            # Method 1: Gradient-based detection for rings
            droplets1 = self.detect_droplets_by_gradient(image_normalized)
            droplets.extend(droplets1)
            
            # Method 2: Circular Hough Transform with multiple parameters
            droplets2 = self.detect_droplets_by_hough(image_normalized)
            droplets.extend(droplets2)
            
            # Method 3: Template matching for circular patterns
            droplets3 = self.detect_droplets_by_template(image_normalized)
            droplets.extend(droplets3)
            
            # Merge overlapping detections
            merged_droplets = self.merge_droplet_detections(droplets)
            
            # Validate and rank droplets
            validated_droplets = []
            for i, droplet in enumerate(merged_droplets):
                if self.validate_droplet_enhanced(image_normalized, droplet['x'], droplet['y'], droplet['r']):
                    droplet['id'] = i + 1
                    validated_droplets.append(droplet)
            
            print(f"Detected {len(validated_droplets)} droplets")
            return validated_droplets
            
        except Exception as e:
            print(f"Error in enhanced droplet detection: {e}")
            return []
    
    def detect_droplets_by_gradient(self, image):
        """Detect droplets using gradient magnitude"""
        try:
            # Calculate gradients
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # Normalize gradient
            gradient_norm = ((gradient_magnitude - gradient_magnitude.min()) / 
                           (gradient_magnitude.max() - gradient_magnitude.min()) * 255).astype(np.uint8)
            
            # Apply threshold to get ring structures
            _, binary = cv2.threshold(gradient_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find circular patterns in gradient image
            circles = cv2.HoughCircles(
                binary,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=100,
                param1=50,
                param2=15,
                minRadius=60,
                maxRadius=180
            )
            
            droplets = []
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    droplets.append({
                        'x': x, 'y': y, 'r': r,
                        'center': (x, y),
                        'radius': r,
                        'method': 'gradient'
                    })
            
            return droplets
            
        except Exception as e:
            print(f"Error in gradient detection: {e}")
            return []
    
    def detect_droplets_by_hough(self, image):
        """Multiple Hough transform passes with different parameters"""
        try:
            # Preprocess
            blurred = cv2.GaussianBlur(image, (5, 5), 1)
            
            all_droplets = []
            
            # Multiple parameter sets for different droplet types
            param_sets = [
                # For thick rings
                {'dp': 1, 'minDist': 120, 'param1': 50, 'param2': 30, 
                 'minRadius': 80, 'maxRadius': 160},
                # For thin rings
                {'dp': 1.5, 'minDist': 100, 'param1': 30, 'param2': 20, 
                 'minRadius': 60, 'maxRadius': 180},
                # For edge cases
                {'dp': 2, 'minDist': 80, 'param1': 40, 'param2': 25, 
                 'minRadius': 70, 'maxRadius': 170}
            ]
            
            for params in param_sets:
                circles = cv2.HoughCircles(
                    blurred,
                    cv2.HOUGH_GRADIENT,
                    **params
                )
                
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    for (x, y, r) in circles:
                        all_droplets.append({
                            'x': x, 'y': y, 'r': r,
                            'center': (x, y),
                            'radius': r,
                            'method': 'hough'
                        })
            
            return all_droplets
            
        except Exception as e:
            print(f"Error in Hough detection: {e}")
            return []
    
    def detect_droplets_by_template(self, image):
        """Template matching for ring patterns"""
        try:
            droplets = []
            
            # Create ring templates of various sizes
            radii = range(70, 170, 20)
            
            for radius in radii:
                # Create ring template
                template_size = radius * 2 + 20
                template = np.zeros((template_size, template_size), dtype=np.uint8)
                center = template_size // 2
                
                # Draw outer circle
                cv2.circle(template, (center, center), radius, 255, 3)
                # Draw inner circle (for ring effect)
                cv2.circle(template, (center, center), int(radius * 0.85), 128, 2)
                
                # Match template
                result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
                
                # Find peaks
                threshold = 0.4
                locations = np.where(result >= threshold)
                
                for pt in zip(*locations[::-1]):
                    x = pt[0] + center
                    y = pt[1] + center
                    
                    droplets.append({
                        'x': x, 'y': y, 'r': radius,
                        'center': (x, y),
                        'radius': radius,
                        'method': 'template'
                    })
            
            return droplets
            
        except Exception as e:
            print(f"Error in template matching: {e}")
            return []
    
    def merge_droplet_detections(self, droplets):
        """Merge overlapping droplet detections"""
        if not droplets:
            return []
        
        merged = []
        used = set()
        
        for i, d1 in enumerate(droplets):
            if i in used:
                continue
            
            # Find overlapping droplets
            overlap_group = [d1]
            for j, d2 in enumerate(droplets[i+1:], i+1):
                if j in used:
                    continue
                
                # Calculate distance between centers
                dist = np.sqrt((d1['x'] - d2['x'])**2 + (d1['y'] - d2['y'])**2)
                
                # Check for overlap
                if dist < (d1['r'] + d2['r']) * 0.5:
                    overlap_group.append(d2)
                    used.add(j)
            
            # Merge overlapping detections
            if len(overlap_group) > 1:
                avg_x = int(np.mean([d['x'] for d in overlap_group]))
                avg_y = int(np.mean([d['y'] for d in overlap_group]))
                avg_r = int(np.mean([d['r'] for d in overlap_group]))
                
                merged.append({
                    'x': avg_x, 'y': avg_y, 'r': avg_r,
                    'center': (avg_x, avg_y),
                    'radius': avg_r
                })
                else:
                merged.append(d1)
        
        return merged
    
    def validate_droplet_enhanced(self, image, x, y, r):
        """Enhanced validation for both thick and thin ring droplets"""
        try:
            # Check bounds
            if x - r < 5 or y - r < 5 or x + r > image.shape[1] - 5 or y + r > image.shape[0] - 5:
                return False
            
            # Sample ring intensities at multiple radii
            angles = np.linspace(0, 2*np.pi, 48)
            
            # Check multiple ring positions (for thick and thin rings)
            ring_scores = []
            for radius_factor in [0.9, 0.95, 1.0, 1.05, 1.1]:
                ring_intensities = []
                for angle in angles:
                    ring_x = int(x + r * radius_factor * np.cos(angle))
                    ring_y = int(y + r * radius_factor * np.sin(angle))
                    
                    if 0 <= ring_x < image.shape[1] and 0 <= ring_y < image.shape[0]:
                        ring_intensities.append(image[ring_y, ring_x])
                
                if ring_intensities:
                    ring_scores.append(np.std(ring_intensities))
            
            # Check center vs ring contrast
            center_region = image[max(0, y-10):min(image.shape[0], y+10), 
                                max(0, x-10):min(image.shape[1], x+10)]
            center_mean = np.mean(center_region)
            
            # Calculate ring statistics
            best_ring_idx = np.argmax(ring_scores)
            radius_factor = [0.9, 0.95, 1.0, 1.05, 1.1][best_ring_idx]
            
            ring_pixels = []
            for angle in angles:
                ring_x = int(x + r * radius_factor * np.cos(angle))
                ring_y = int(y + r * radius_factor * np.sin(angle))
                if 0 <= ring_x < image.shape[1] and 0 <= ring_y < image.shape[0]:
                    ring_pixels.append(image[ring_y, ring_x])
            
            if not ring_pixels:
                return False
            
            ring_mean = np.mean(ring_pixels)
            ring_std = np.std(ring_pixels)
            
            # Validation criteria
            # 1. Ring should have contrast with center
            contrast = abs(center_mean - ring_mean)
            
            # 2. Ring should have some variation (not uniform)
            # 3. For thin rings, contrast might be lower
            is_valid = (contrast > 10 or ring_std > 15) and ring_std > 5
            
            return is_valid
            
        except Exception as e:
            print(f"Error validating droplet: {e}")
            return False
    
    def detect_cancer_cells_tritc_guided(self, bf_image, tritc_image, droplet):
        """TRITC-guided cancer cell detection"""
        try:
            x, y, r = droplet['x'], droplet['y'], droplet['r']
            
            # Create mask for droplet interior
            mask = np.zeros(bf_image.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), int(r * 0.85), 255, -1)
            
            # Normalize images
            bf_norm = self.normalize_image(bf_image)
            tritc_norm = self.normalize_image(tritc_image)
            
            # Find TRITC-positive regions first
            tritc_masked = cv2.bitwise_and(tritc_norm, tritc_norm, mask=mask)
            
            # Adaptive threshold on TRITC to find positive regions
            tritc_blur = cv2.GaussianBlur(tritc_masked, (5, 5), 1)
            
            # Use multiple thresholds to catch different intensity levels
            cancer_cells = []
            
            # Method 1: Otsu threshold on TRITC
            if np.max(tritc_blur) > 10:  # Only if there's signal
                _, tritc_thresh = cv2.threshold(tritc_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cells1 = self.extract_cells_from_tritc_mask(tritc_thresh, bf_norm, tritc_norm, mask)
                cancer_cells.extend(cells1)
            
            # Method 2: Percentile-based threshold
            tritc_in_droplet = tritc_blur[mask > 0]
            if len(tritc_in_droplet) > 0:
                threshold_percentiles = [75, 85, 95]
                for percentile in threshold_percentiles:
                    thresh_val = np.percentile(tritc_in_droplet[tritc_in_droplet > 0], percentile) if np.any(tritc_in_droplet > 0) else 0
                    if thresh_val > 5:
                        _, tritc_thresh = cv2.threshold(tritc_blur, thresh_val, 255, cv2.THRESH_BINARY)
                        cells2 = self.extract_cells_from_tritc_mask(tritc_thresh, bf_norm, tritc_norm, mask)
                        cancer_cells.extend(cells2)
            
            # Method 3: Local maxima in TRITC channel
            cells3 = self.detect_cells_by_tritc_peaks(tritc_blur, bf_norm, mask)
            cancer_cells.extend(cells3)
            
            # Method 4: Watershed on TRITC signal
            cells4 = self.detect_cells_by_tritc_watershed(tritc_blur, bf_norm, mask)
            cancer_cells.extend(cells4)
            
            # Merge duplicate detections
            merged_cells = self.merge_cancer_cells(cancer_cells)
            
            # Classify cells by viability based on TRITC intensity
            for cell in merged_cells:
                # Higher TRITC = more viable
                if cell['tritc_intensity'] > 30:
                    cell['viable'] = True
                    cell['viability_score'] = 'high'
                elif cell['tritc_intensity'] > 15:
                    cell['viable'] = True
                    cell['viability_score'] = 'medium'
                else:
                    cell['viable'] = False
                    cell['viability_score'] = 'low'
            
            return {
                'cancer_cells': merged_cells,
                'total_cells': len(merged_cells),
                'viable_cells': len([c for c in merged_cells if c['viable']])
            }
            
        except Exception as e:
            print(f"Error in TRITC-guided detection: {e}")
            return {'cancer_cells': [], 'total_cells': 0, 'viable_cells': 0}
    
    def normalize_image(self, image):
        """Normalize image to uint8"""
        if image.dtype == np.uint8:
            return image
        
        img_min = image.min()
        img_max = image.max()
        if img_max > img_min:
            return ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        return np.zeros_like(image, dtype=np.uint8)
    
    def extract_cells_from_tritc_mask(self, tritc_mask, bf_image, tritc_image, droplet_mask):
        """Extract individual cells from TRITC mask"""
        try:
            cells = []
            
            # Clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(tritc_mask, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            
            # Find connected components
            num_labels, labels = cv2.connectedComponents(cleaned)
            
            for label in range(1, num_labels):
                # Get component mask
                component_mask = (labels == label).astype(np.uint8) * 255
                
                # Check if component is within droplet
                if cv2.countNonZero(cv2.bitwise_and(component_mask, droplet_mask)) == 0:
                    continue
                
                # Get component properties
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                
                contour = contours[0]
                    area = cv2.contourArea(contour)
                
                # Filter by size
                if area < 20 or area > 5000:
                    continue
                
                # Get centroid
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Calculate properties
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                # Get intensity values
                tritc_intensity = cv2.mean(tritc_image, mask=component_mask)[0]
                bf_intensity = cv2.mean(bf_image, mask=component_mask)[0]
                
                cells.append({
                    'center': (cx, cy),
                    'area': area,
                    'contour': contour,
                    'circularity': circularity,
                    'tritc_intensity': tritc_intensity,
                    'bf_intensity': bf_intensity,
                    'method': 'tritc_mask'
                })
            
            return cells
            
        except Exception as e:
            print(f"Error extracting cells from TRITC mask: {e}")
            return []
    
    def detect_cells_by_tritc_peaks(self, tritc_image, bf_image, mask):
        """Detect cells by finding local maxima in TRITC signal"""
        try:
            cells = []
            
            # Apply mask
            tritc_masked = cv2.bitwise_and(tritc_image, tritc_image, mask=mask)
            
            # Find local maxima
            local_maxima = peak_local_max(tritc_masked, min_distance=10, 
                                         threshold_abs=10, indices=True)
            
            for peak in local_maxima:
                y, x = peak
                
                # Check if peak is within mask
                if mask[y, x] == 0:
                    continue
                
                # Grow region from peak
                region_mask = np.zeros_like(tritc_image)
                cv2.circle(region_mask, (x, y), 15, 255, -1)
                region_mask = cv2.bitwise_and(region_mask, mask)
                
                # Get region properties
                tritc_intensity = tritc_image[y, x]
                bf_intensity = bf_image[y, x]
                
                # Estimate cell area around peak
                area = np.pi * 15 * 15
                        
                        cells.append({
                    'center': (x, y),
                            'area': area,
                    'contour': None,
                    'circularity': 0.8,
                    'tritc_intensity': tritc_intensity,
                    'bf_intensity': bf_intensity,
                    'method': 'tritc_peak'
                        })
            
            return cells
            
        except Exception as e:
            print(f"Error in peak detection: {e}")
            return []
    
    def detect_cells_by_tritc_watershed(self, tritc_image, bf_image, mask):
        """Watershed segmentation on TRITC signal"""
        try:
            cells = []
            
            # Apply mask and threshold
            tritc_masked = cv2.bitwise_and(tritc_image, tritc_image, mask=mask)
            
            # Skip if no significant signal
            if np.max(tritc_masked) < 15:
                return []
            
            # Threshold
            _, binary = cv2.threshold(tritc_masked, 15, 255, cv2.THRESH_BINARY)
            
            # Distance transform
            dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            
            # Find peaks
            _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            
            # Find unknown region
            kernel = np.ones((3, 3), np.uint8)
            sure_bg = cv2.dilate(binary, kernel, iterations=2)
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            # Marker labelling
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            
            # Apply watershed
            img_for_watershed = cv2.cvtColor(tritc_image, cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(img_for_watershed, markers)
            
            # Extract cells from watershed regions
            for marker_id in np.unique(markers):
                if marker_id <= 1:  # Skip background and borders
                    continue
                
                # Create mask for this marker
                cell_mask = (markers == marker_id).astype(np.uint8) * 255
                cell_mask = cv2.bitwise_and(cell_mask, mask)
                
                # Get properties
                contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                
                    contour = contours[0]
                    area = cv2.contourArea(contour)
                
                if area < 20 or area > 3000:
                    continue
            
            # Get centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                    continue
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
                    # Calculate properties
                tritc_intensity = cv2.mean(tritc_image, mask=cell_mask)[0]
                bf_intensity = cv2.mean(bf_image, mask=cell_mask)[0]
                        
                        cells.append({
                            'center': (cx, cy),
                            'area': area,
                            'contour': contour,
                    'circularity': 0.7,
                    'tritc_intensity': tritc_intensity,
                    'bf_intensity': bf_intensity,
                    'method': 'tritc_watershed'
                        })
            
            return cells
            
        except Exception as e:
            print(f"Error in watershed segmentation: {e}")
            return []
    
    def merge_cancer_cells(self, cells):
        """Merge duplicate cancer cell detections"""
        if not cells:
                return []
            
            merged = []
            used = set()
            
            for i, cell1 in enumerate(cells):
                if i in used:
                    continue
                    
                # Find nearby cells
            nearby_group = [cell1]
            for j, cell2 in enumerate(cells[i+1:], i+1):
                if j in used:
                        continue
                    
                # Calculate distance
                    dist = np.sqrt((cell1['center'][0] - cell2['center'][0])**2 + 
                                 (cell1['center'][1] - cell2['center'][1])**2)
                    
                # Merge if very close
                if dist < 15:
                    nearby_group.append(cell2)
                        used.add(j)
                
            # Keep cell with highest TRITC intensity
            if len(nearby_group) > 1:
                best_cell = max(nearby_group, key=lambda c: c['tritc_intensity'])
                # Update area to combined area
                best_cell['area'] = sum(c['area'] for c in nearby_group) / len(nearby_group)
                else:
                    best_cell = cell1
                    
                merged.append(best_cell)
                used.add(i)
            
            return merged
    
    def analyze_current_frame(self):
        """Analyze the current frame for droplets and cells"""
        if not self.nd2_file:
            messagebox.showwarning("Warning", "Please load an ND2 file first")
            return
            
        try:
            self.status_label.config(text="Analyzing current frame...", foreground='orange')
            self.root.update()
            
            # Get current frame
            frame = self.nd2_file[self.current_timepoint]
            
            # Get brightfield and TRITC channels
            bf_channel = None
            tritc_channel = None
            
            for i, channel in enumerate(self.channels):
                if 'bf' in channel.lower() or 'bright' in channel.lower() or 'phase' in channel.lower():
                    bf_channel = i
                elif 'tritc' in channel.lower() or 'red' in channel.lower():
                    tritc_channel = i
            
            if bf_channel is None:
                bf_channel = 0  # Default to first channel
            
            # Get images
            bf_image = frame[bf_channel]
            tritc_image = frame[tritc_channel] if tritc_channel is not None else bf_image
            
            # Detect droplets using enhanced method
            self.detected_droplets = self.detect_droplets_enhanced(bf_image)
            
            # Analyze each droplet for cancer cells
            total_cells = 0
            total_viable = 0
            
            for droplet in self.detected_droplets:
                cell_analysis = self.detect_cancer_cells_tritc_guided(bf_image, tritc_image, droplet)
                droplet['cell_analysis'] = cell_analysis
                total_cells += cell_analysis['total_cells']
                total_viable += cell_analysis['viable_cells']
            
            # Update display
            self.update_analysis_display(bf_image, tritc_image)
            
            # Update results text
            results_summary = f"Analysis Results:\n"
            results_summary += f"• Droplets detected: {len(self.detected_droplets)}\n"
            results_summary += f"• Total cancer cells: {total_cells}\n"
            results_summary += f"• Viable cells: {total_viable}\n"
            results_summary += f"• Timepoint: {self.current_timepoint + 1}/{self.total_timepoints}"
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, results_summary)
            
            # Enable export
            self.export_btn.configure(state="normal")
            
            self.status_label.config(text=f"✅ Analysis complete: {len(self.detected_droplets)} droplets, {total_cells} cells", 
                                   foreground='green')
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed:\n{str(e)}")
            self.status_label.config(text="❌ Analysis failed", foreground='red')
    
    def update_analysis_display(self, bf_image, tritc_image):
        """Update the display with analysis results"""
        try:
            # Clear all axes
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.clear()
            
            # Normalize images for display
            bf_display = self.normalize_image(bf_image)
            tritc_display = self.normalize_image(tritc_image)
            
            # Plot 1: Original brightfield
            self.ax1.imshow(bf_display, cmap='gray')
            self.ax1.set_title('Brightfield Image', fontsize=12, fontweight='bold')
            self.ax1.axis('off')
            
            # Plot 2: TRITC channel
            self.ax2.imshow(tritc_display, cmap='hot')
            self.ax2.set_title('TRITC Channel', fontsize=12, fontweight='bold')
            self.ax2.axis('off')
            
            # Plot 3: Droplet detection
            self.ax3.imshow(bf_display, cmap='gray')
            for droplet in self.detected_droplets:
                circle = plt.Circle((droplet['x'], droplet['y']), droplet['r'], 
                                  fill=False, color='red', linewidth=2)
                self.ax3.add_patch(circle)
                self.ax3.text(droplet['x'], droplet['y'] - droplet['r'] - 10, 
                             f"D{droplet['id']}", color='red', fontsize=10, 
                             ha='center', fontweight='bold')
            self.ax3.set_title(f'Droplet Detection ({len(self.detected_droplets)} found)', 
                              fontsize=12, fontweight='bold')
            self.ax3.axis('off')
            
            # Plot 4: Cell detection
            self.ax4.imshow(bf_display, cmap='gray')
            cell_count = 0
            for droplet in self.detected_droplets:
                # Draw droplet boundary
                circle = plt.Circle((droplet['x'], droplet['y']), droplet['r'], 
                                  fill=False, color='blue', linewidth=2, alpha=0.7)
                self.ax4.add_patch(circle)
                
                # Draw cells
                if 'cell_analysis' in droplet:
                    for cell in droplet['cell_analysis']['cancer_cells']:
                        cell_count += 1
                        color = 'green' if cell['viable'] else 'yellow'
                        cell_circle = plt.Circle(cell['center'], 5, 
                                               fill=True, color=color, alpha=0.8)
                        self.ax4.add_patch(cell_circle)
            
            self.ax4.set_title(f'Cell Detection ({cell_count} cells)', 
                              fontsize=12, fontweight='bold')
            self.ax4.axis('off')
            
            # Adjust layout and redraw
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating display: {e}")
    
    def on_timepoint_change(self, value):
        """Handle timepoint slider change"""
        try:
        self.current_timepoint = int(float(value))
        self.timepoint_label.config(text=f"{self.current_timepoint + 1} / {self.total_timepoints}")
        self.update_display()
        except Exception as e:
            print(f"Error changing timepoint: {e}")
        
    def on_channel_change(self, event):
        """Handle channel selection change"""
        try:
        self.current_channel = self.channel_combo.current()
        self.update_display()
        except Exception as e:
            print(f"Error changing channel: {e}")
        
    def update_display(self):
        """Update the main display with current frame"""
        if not self.nd2_file:
            return
            
        try:
            # Get current frame
            frame = self.nd2_file[self.current_timepoint]
            
            # Get current channel image
            if self.current_channel < len(frame):
                image = frame[self.current_channel]
                
                # Clear all axes
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.clear()
            
                # Normalize image for display
                display_image = self.normalize_image(image)
                
                # Show same image in all panels initially
                for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                    ax.imshow(display_image, cmap='gray')
                ax.axis('off')
            
                self.ax1.set_title(f'Channel: {self.channels[self.current_channel]}', 
                                  fontsize=12, fontweight='bold')
                self.ax2.set_title('Channel 2', fontsize=12, fontweight='bold')
                self.ax3.set_title('Channel 3', fontsize=12, fontweight='bold')
                self.ax4.set_title('Channel 4', fontsize=12, fontweight='bold')
                
                # Adjust layout and redraw
                self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating display: {e}")
    
    def export_results(self):
        """Export analysis results to CSV"""
        if not self.detected_droplets:
            messagebox.showwarning("Warning", "No analysis results to export")
            return
            
        try:
            file_path = filedialog.asksaveasfilename(
                title="Save Results",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if not file_path:
                return
                
            # Prepare data for export
            export_data = []
            
            for droplet in self.detected_droplets:
                droplet_data = {
                    'Timepoint': self.current_timepoint + 1,
                            'Droplet_ID': droplet['id'],
                            'Droplet_X': droplet['x'],
                            'Droplet_Y': droplet['y'],
                            'Droplet_Radius': droplet['r'],
                    'Total_Cells': 0,
                    'Viable_Cells': 0
                }
                
                if 'cell_analysis' in droplet:
                    droplet_data['Total_Cells'] = droplet['cell_analysis']['total_cells']
                    droplet_data['Viable_Cells'] = droplet['cell_analysis']['viable_cells']
                
                export_data.append(droplet_data)
            
            # Create DataFrame and save
            df = pd.DataFrame(export_data)
            df.to_csv(file_path, index=False)
            
            messagebox.showinfo("Success", f"Results exported to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{str(e)}")
    
    def run(self):
        """Start the application"""
        self.root.mainloop()
        
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'nd2_file') and self.nd2_file:
            self.nd2_file.close()

if __name__ == "__main__":
        app = CellAnalyzer()
        app.run()