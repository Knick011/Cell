import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import os
import pandas as pd

try:
    from nd2reader import ND2Reader
    ND2_AVAILABLE = True
except ImportError:
    ND2_AVAILABLE = False
    print("nd2reader not installed. Install with: pip install nd2reader")

class CellAnalyzer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Cell Analysis - Droplet & Cell Detection")
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
            messagebox.showerror("Error", "nd2reader not installed.\\nRun: pip install nd2reader")
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
            messagebox.showerror("Error", f"Failed to load ND2 file:\\n{str(e)}")
            self.status_label.config(text="❌ Error loading file", foreground='red')
    
    def detect_droplets(self, image):
        """Detect droplets by finding the characteristic black circular rings"""
        try:
            # Convert to uint8 if needed
            if image.dtype != np.uint8:
                image_normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            else:
                image_normalized = image.copy()
            
            # Apply bilateral filter to preserve edges while reducing noise
            bilateral = cv2.bilateralFilter(image_normalized, 9, 75, 75)
            
            # Apply Canny edge detection
            edges = cv2.Canny(bilateral, 30, 80, apertureSize=3, L2gradient=True)
            
            # Dilate edges to make them more prominent
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Use HoughCircles on edge image
            circles = cv2.HoughCircles(
                edges,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=150,
                param1=50,
                param2=20,
                minRadius=80,
                maxRadius=160
            )
            
            droplets = []
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                # Filter circles that actually correspond to black rings
                valid_circles = []
                for (x, y, r) in circles:
                    if self.validate_droplet_ring(image_normalized, x, y, r):
                        valid_circles.append((x, y, r))
                
                # Sort by position (top to bottom, left to right)
                valid_circles = sorted(valid_circles, key=lambda c: (c[1], c[0]))
                
                for i, (x, y, r) in enumerate(valid_circles):
                    droplets.append({
                        'id': i + 1,
                        'center': (x, y),
                        'radius': r,
                        'x': x,
                        'y': y,
                        'r': r
                    })
            
            return droplets
            
        except Exception as e:
            print(f"Error in droplet detection: {e}")
            return []
    
    def validate_droplet_ring(self, image, x, y, r):
        """Validate that a detected circle actually corresponds to a black ring droplet"""
        try:
            # Check if circle is within image bounds
            if x - r < 10 or y - r < 10 or x + r > image.shape[1] - 10 or y + r > image.shape[0] - 10:
                return False
            
            # Sample pixels around the circle perimeter
            angles = np.linspace(0, 2*np.pi, 32)
            ring_intensities = []
            
            for angle in angles:
                ring_x = int(x + r * np.cos(angle))
                ring_y = int(y + r * np.sin(angle))
                
                if 0 <= ring_x < image.shape[1] and 0 <= ring_y < image.shape[0]:
                    ring_intensities.append(image[ring_y, ring_x])
            
            if len(ring_intensities) < 20:
                return False
            
            # Check if the ring area is darker than the center
            center_intensity = np.mean(image[max(0, y-10):min(image.shape[0], y+10), 
                                            max(0, x-10):min(image.shape[1], x+10)])
            ring_intensity = np.mean(ring_intensities)
            
            # Black ring should be significantly darker than center
            return ring_intensity < center_intensity - 20
            
        except Exception as e:
            print(f"Error validating droplet ring: {e}")
            return False
    
    def detect_cells_in_droplet(self, image, tritc_image, droplet):
        """Detect and analyze cells within a single droplet"""
        try:
            x, y, r = droplet['x'], droplet['y'], droplet['r']
            
            # Create circular mask for the droplet
            mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), int(r * 0.8), 255, -1)
            
            # Extract droplet regions
            droplet_region = cv2.bitwise_and(image, image, mask=mask)
            tritc_region = cv2.bitwise_and(tritc_image, tritc_image, mask=mask)
            
            # Convert to uint8 if needed
            if droplet_region.dtype != np.uint8:
                droplet_region = ((droplet_region - droplet_region.min()) / 
                                (droplet_region.max() - droplet_region.min() + 1e-8) * 255).astype(np.uint8)
            
            if tritc_region.dtype != np.uint8:
                tritc_region = ((tritc_region - tritc_region.min()) / 
                              (tritc_region.max() - tritc_region.min() + 1e-8) * 255).astype(np.uint8)
            
            # Detect all cells using brightfield morphology
            all_cells = self.detect_cells_brightfield(droplet_region, mask)
            
            # Classify cells based on size and TRITC signal
            k562_cells = []
            nk_cells = []
            
            for cell in all_cells:
                # Check TRITC signal at cell location
                cell_mask = np.zeros_like(tritc_region)
                cv2.drawContours(cell_mask, [cell['contour']], -1, 255, -1)
                tritc_intensity = cv2.mean(tritc_region, mask=cell_mask)[0]
                
                # Classify based on size and TRITC signal
                if cell['area'] > 200 or tritc_intensity > 15:
                    cell['type'] = 'K562'
                    cell['tritc_intensity'] = tritc_intensity
                    cell['viable'] = tritc_intensity > 20 and cell['circularity'] > 0.5
                    k562_cells.append(cell)
                else:
                    cell['type'] = 'NK'
                    cell['tritc_intensity'] = tritc_intensity
                    cell['viable'] = 50 < cell['area'] < 500 and cell['circularity'] > 0.4
                    nk_cells.append(cell)
            
            return {
                'k562_cells': k562_cells,
                'nk_cells': nk_cells,
                'total_cells': len(k562_cells) + len(nk_cells)
            }
            
        except Exception as e:
            print(f"Error detecting cells in droplet: {e}")
            return {'k562_cells': [], 'nk_cells': [], 'total_cells': 0}
    
    def detect_cells_brightfield(self, bf_image, mask):
        """Enhanced cell detection using multiple approaches"""
        try:
            # Apply mask
            masked_bf = cv2.bitwise_and(bf_image, bf_image, mask=mask)
            
            # Skip if no signal in masked region
            if cv2.mean(masked_bf, mask=mask)[0] < 10:
                return []
            
            # Enhance contrast more aggressively
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
            enhanced = clahe.apply(masked_bf)
            
            # Apply additional preprocessing
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            all_cells = []
            
            # Method 1: Very sensitive blob detection
            cells1 = self.detect_blobs_sensitive(blurred, mask)
            all_cells.extend(cells1)
            
            # Method 2: Laplacian of Gaussian for cell detection
            cells2 = self.detect_cells_log(blurred, mask)
            all_cells.extend(cells2)
            
            # Method 3: Multi-scale template matching
            cells3 = self.detect_cells_template_matching(blurred, mask)
            all_cells.extend(cells3)
            
            # Method 4: Watershed segmentation
            cells4 = self.detect_cells_watershed(blurred, mask)
            all_cells.extend(cells4)
            
            # Remove duplicates and filter
            merged_cells = self.merge_duplicate_cells(all_cells)
            
            # Additional filtering by reasonable cell criteria
            filtered_cells = []
            for cell in merged_cells:
                if (20 < cell['area'] < 8000 and  # Broader size range
                    cell['circularity'] > 0.1):     # Very loose circularity
                    filtered_cells.append(cell)
            
            print(f"Detected {len(filtered_cells)} cells using enhanced methods")
            return filtered_cells
            
        except Exception as e:
            print(f"Error in enhanced cell detection: {e}")
            return []
    
    def detect_blobs_sensitive(self, image, mask):
        """Very sensitive blob detection for small/faint cells"""
        try:
            cells = []
            
            # Multiple threshold levels to catch faint cells
            threshold_methods = [
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            ]
            
            for thresh_type in threshold_methods:
                _, thresh = cv2.threshold(image, 0, 255, thresh_type)
                thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
                
                # Very gentle morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
                
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 15 < area < 10000:  # Very broad range
                        cell = self.create_cell_from_contour(contour, image)
                        if cell:
                            cells.append(cell)
            
            return cells
            
        except Exception as e:
            print(f"Error in sensitive blob detection: {e}")
            return []
    
    def detect_cells_log(self, image, mask):
        """Laplacian of Gaussian for cell detection"""
        try:
            # Apply LoG filter
            log_filtered = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
            log_filtered = np.absolute(log_filtered)
            log_filtered = log_filtered.astype(np.uint8)
            
            # Apply mask
            log_filtered = cv2.bitwise_and(log_filtered, log_filtered, mask=mask)
            
            # Threshold
            _, thresh = cv2.threshold(log_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            cells = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 10 < area < 5000:
                    cell = self.create_cell_from_contour(contour, image)
                    if cell:
                        cells.append(cell)
            
            return cells
            
        except Exception as e:
            print(f"Error in LoG detection: {e}")
            return []
    
    def detect_cells_template_matching(self, image, mask):
        """Template matching for circular cell-like objects"""
        try:
            cells = []
            
            # Create circular templates of different sizes
            template_sizes = [8, 12, 16, 20, 25, 30]
            
            for size in template_sizes:
                # Create circular template
                template = np.zeros((size*2, size*2), dtype=np.uint8)
                cv2.circle(template, (size, size), size//2, 255, -1)
                
                # Apply template matching
                result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
                
                # Find good matches
                threshold = 0.3  # Lower threshold for more sensitivity
                locations = np.where(result >= threshold)
                
                for pt in zip(*locations[::-1]):
                    x, y = pt[0] + size, pt[1] + size
                    
                    # Check if point is within mask
                    if (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and 
                        mask[y, x] > 0):
                        
                        # Create synthetic contour
                        center = (x, y)
                        radius = size // 2
                        area = np.pi * radius * radius
                        
                        # Calculate intensity in this region
                        roi = image[max(0, y-radius):min(image.shape[0], y+radius),
                                   max(0, x-radius):min(image.shape[1], x+radius)]
                        intensity = np.mean(roi) if roi.size > 0 else 0
                        
                        cells.append({
                            'center': center,
                            'area': area,
                            'intensity': intensity,
                            'contour': None,  # No actual contour for template matches
                            'circularity': 1.0,  # Templates are perfectly circular
                            'detection_method': 'template'
                        })
            
            return cells
            
        except Exception as e:
            print(f"Error in template matching: {e}")
            return []
    
    def detect_cells_watershed(self, image, mask):
        """Watershed segmentation for overlapping cells"""
        try:
            # Apply mask
            masked = cv2.bitwise_and(image, image, mask=mask)
            
            # Distance transform
            _, thresh = cv2.threshold(masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Remove noise
            kernel = np.ones((2, 2), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=2)
            
            # Distance transform
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            
            # Sure foreground area
            _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            
            # Unknown region
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            # Marker labelling
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            
            # Apply watershed
            img_for_watershed = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(img_for_watershed, markers)
            
            # Extract cells from watershed regions
            cells = []
            for marker_id in np.unique(markers):
                if marker_id <= 1:  # Skip background and borders
                    continue
                
                # Create mask for this marker
                cell_mask = (markers == marker_id).astype(np.uint8) * 255
                
                # Find contour
                contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    contour = contours[0]
                    area = cv2.contourArea(contour)
                    if 20 < area < 5000:
                        cell = self.create_cell_from_contour(contour, image)
                        if cell:
                            cells.append(cell)
            
            return cells
            
        except Exception as e:
            print(f"Error in watershed detection: {e}")
            return []
    
    def create_cell_from_contour(self, contour, image):
        """Create cell object from contour with better error handling"""
        try:
            area = cv2.contourArea(contour)
            if area < 10:  # Skip tiny areas
                return None
            
            # Get centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                return None
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Check if center is within image bounds
            if not (0 <= cx < image.shape[1] and 0 <= cy < image.shape[0]):
                return None
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            # Calculate mean intensity
            try:
                cell_mask = np.zeros_like(image)
                cv2.drawContours(cell_mask, [contour], -1, 255, -1)
                mean_intensity = cv2.mean(image, mask=cell_mask)[0]
            except:
                mean_intensity = 0
            
            return {
                'center': (cx, cy),
                'area': area,
                'intensity': mean_intensity,
                'contour': contour,
                'circularity': circularity,
                'detection_method': 'contour'
            }
            
        except Exception as e:
            print(f"Error creating cell from contour: {e}")
            return None
    
    def find_cell_contours(self, thresh_image, original_image, method):
        """Find cell contours from thresholded image"""
        try:
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            cells = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 5000:  # Cell size range
                    # Calculate properties
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Calculate circularity
                        perimeter = cv2.arcLength(contour, True)
                        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                        
                        # Calculate mean intensity
                        cell_mask = np.zeros_like(original_image)
                        cv2.drawContours(cell_mask, [contour], -1, 255, -1)
                        mean_intensity = cv2.mean(original_image, mask=cell_mask)[0]
                        
                        cells.append({
                            'center': (cx, cy),
                            'area': area,
                            'intensity': mean_intensity,
                            'contour': contour,
                            'circularity': circularity,
                            'detection_method': method
                        })
            
            return cells
            
        except Exception as e:
            print(f"Error finding contours: {e}")
            return []
    
    def merge_duplicate_cells(self, cells):
        """Enhanced duplicate removal with better distance checking"""
        try:
            if len(cells) == 0:
                return []
            
            merged = []
            used = set()
            
            for i, cell1 in enumerate(cells):
                if i in used:
                    continue
                    
                # Find nearby cells
                nearby_cells = [cell1]
                for j, cell2 in enumerate(cells):
                    if j <= i or j in used:
                        continue
                    
                    # Calculate distance between centers
                    dist = np.sqrt((cell1['center'][0] - cell2['center'][0])**2 + 
                                 (cell1['center'][1] - cell2['center'][1])**2)
                    
                    # Use smaller merge distance for better precision
                    merge_distance = min(20, max(8, np.sqrt(max(cell1['area'], cell2['area'])) * 0.8))
                    
                    if dist < merge_distance:
                        nearby_cells.append(cell2)
                        used.add(j)
                
                # Keep the cell with best properties (largest area or highest intensity)
                if len(nearby_cells) > 1:
                    best_cell = max(nearby_cells, key=lambda c: c['area'] * c['intensity'])
                else:
                    best_cell = cell1
                    
                merged.append(best_cell)
                used.add(i)
            
            print(f"Merged {len(cells)} detections into {len(merged)} cells")
            return merged
            
        except Exception as e:
            print(f"Error merging cells: {e}")
            return cells
    
    def analyze_current_frame(self):
        """Analyze the current timepoint for all droplets"""
        if self.nd2_file is None:
            return
            
        try:
            self.status_label.config(text="🔍 Analyzing current frame...", foreground='orange')
            self.root.update()
            
            # Get both channel images
            bf_image = self.nd2_file.get_frame_2D(c=0, t=self.current_timepoint)
            tritc_image = self.nd2_file.get_frame_2D(c=1, t=self.current_timepoint)
            
            # Detect droplets
            self.detected_droplets = self.detect_droplets(bf_image)
            
            # Analyze each droplet
            analysis_results = []
            for droplet in self.detected_droplets:
                cell_data = self.detect_cells_in_droplet(bf_image, tritc_image, droplet)
                droplet['cell_data'] = cell_data
                
                k562_count = len(cell_data['k562_cells'])
                nk_count = len(cell_data['nk_cells'])
                viable_k562 = len([c for c in cell_data['k562_cells'] if c['viable']])
                
                analysis_results.append(f"Droplet {droplet['id']}: {k562_count} K562 cells ({viable_k562} viable), {nk_count} NK cells")
            
            # Update results display
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Analysis Results - Timepoint {self.current_timepoint + 1}:\\n")
            self.results_text.insert(tk.END, f"Found {len(self.detected_droplets)} droplets\\n\\n")
            for result in analysis_results:
                self.results_text.insert(tk.END, result + "\\n")
            
            # Store results
            self.cell_data[self.current_timepoint] = self.detected_droplets.copy()
            
            # Update display
            self.update_analysis_display(bf_image, tritc_image)
            
            # Enable export
            self.export_btn.configure(state="normal")
            
            self.status_label.config(text=f"✅ Analysis complete - Found {len(self.detected_droplets)} droplets", 
                                   foreground='green')
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed:\\n{str(e)}")
            self.status_label.config(text="❌ Analysis failed", foreground='red')
    
    def update_analysis_display(self, bf_image, tritc_image):
        """Update the four-panel display with analysis results"""
        try:
            # Clear all axes
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.clear()
            
            # Panel 1: Original Brightfield
            self.ax1.imshow(bf_image, cmap='gray')
            self.ax1.set_title(f"Brightfield - T{self.current_timepoint+1}", fontsize=14, fontweight='bold')
            self.ax1.axis('off')
            
            # Panel 2: Original TRITC
            self.ax2.imshow(tritc_image, cmap='hot')
            self.ax2.set_title(f"TRITC (K562 stain) - T{self.current_timepoint+1}", fontsize=14, fontweight='bold')
            self.ax2.axis('off')
            
            # Panel 3: Droplet Detection
            display_img = cv2.cvtColor(bf_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            for droplet in self.detected_droplets:
                x, y, r = droplet['x'], droplet['y'], droplet['r']
                cv2.circle(display_img, (x, y), r, (0, 255, 0), 3)
                cv2.putText(display_img, f"D{droplet['id']}", (x-15, y-r-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            self.ax3.imshow(display_img)
            self.ax3.set_title(f"Droplet Detection - Found {len(self.detected_droplets)}", 
                              fontsize=14, fontweight='bold')
            self.ax3.axis('off')
            
            # Panel 4: Cell Analysis
            analysis_img = cv2.cvtColor(bf_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            for droplet in self.detected_droplets:
                x, y, r = droplet['x'], droplet['y'], droplet['r']
                cv2.circle(analysis_img, (x, y), r, (255, 255, 255), 2)
                
                if 'cell_data' in droplet:
                    # Draw K562 cells
                    for cell in droplet['cell_data']['k562_cells']:
                        color = (0, 255, 0) if cell['viable'] else (0, 0, 255)
                        cv2.circle(analysis_img, cell['center'], 10, color, -1)
                        # Add TRITC intensity
                        tritc_val = int(cell.get('tritc_intensity', 0))
                        cv2.putText(analysis_img, str(tritc_val), 
                                   (cell['center'][0]-8, cell['center'][1]-12), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    # Draw NK cells
                    for cell in droplet['cell_data']['nk_cells']:
                        color = (255, 0, 0) if cell['viable'] else (128, 0, 128)
                        cv2.circle(analysis_img, cell['center'], 5, color, -1)
                    
                    # Add summary text
                    k562_count = len(droplet['cell_data']['k562_cells'])
                    nk_count = len(droplet['cell_data']['nk_cells'])
                    viable_k562 = len([c for c in droplet['cell_data']['k562_cells'] if c['viable']])
                    
                    text = f"K:{k562_count}({viable_k562}v) N:{nk_count}"
                    cv2.putText(analysis_img, text, (x-40, y+r+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            self.ax4.imshow(analysis_img)
            self.ax4.set_title("Cell Analysis (Green=Viable K562, Red=Dead K562, Blue=NK)", 
                              fontsize=14, fontweight='bold')
            self.ax4.axis('off')
            
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating display: {e}")
    
    def on_timepoint_change(self, value):
        if self.nd2_file is None:
            return
            
        self.current_timepoint = int(float(value))
        self.timepoint_label.config(text=f"{self.current_timepoint + 1} / {self.total_timepoints}")
        self.update_display()
        
    def on_channel_change(self, event):
        if self.nd2_file is None:
            return
            
        self.current_channel = self.channel_combo.current()
        self.update_display()
        
    def update_display(self):
        if self.nd2_file is None:
            return
            
        try:
            # Get current image
            current_image = self.nd2_file.get_frame_2D(c=self.current_channel, t=self.current_timepoint)
            
            # Clear previous plots
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.clear()
            
            # Simple display for navigation
            self.ax1.imshow(current_image, cmap='gray' if self.current_channel == 0 else 'hot')
            self.ax1.set_title(f"{self.channels[self.current_channel]} - T{self.current_timepoint+1}")
            self.ax1.axis('off')
            
            # Show message in other panels
            for ax, title in zip([self.ax2, self.ax3, self.ax4], 
                               ["Click 'Analyze Current Frame'", "to see detailed", "analysis results"]):
                ax.text(0.5, 0.5, title, ha='center', va='center', fontsize=16, 
                       transform=ax.transAxes)
                ax.axis('off')
            
            self.canvas.draw()
            
        except Exception as e:
            self.status_label.config(text=f"Error displaying image: {str(e)}")
    
    def export_results(self):
        """Export analysis results to CSV"""
        if not self.cell_data:
            messagebox.showwarning("Warning", "No analysis data to export")
            return
            
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save analysis results"
            )
            
            if not file_path:
                return
                
            # Prepare data for export
            export_data = []
            for timepoint, droplets in self.cell_data.items():
                for droplet in droplets:
                    if 'cell_data' in droplet:
                        k562_count = len(droplet['cell_data']['k562_cells'])
                        nk_count = len(droplet['cell_data']['nk_cells'])
                        viable_k562 = len([c for c in droplet['cell_data']['k562_cells'] if c['viable']])
                        dead_k562 = k562_count - viable_k562
                        
                        export_data.append({
                            'Timepoint': timepoint + 1,
                            'Droplet_ID': droplet['id'],
                            'Droplet_X': droplet['x'],
                            'Droplet_Y': droplet['y'],
                            'Droplet_Radius': droplet['r'],
                            'K562_Total': k562_count,
                            'K562_Viable': viable_k562,
                            'K562_Dead': dead_k562,
                            'NK_Count': nk_count,
                            'Total_Cells': k562_count + nk_count
                        })
            
            # Save to CSV
            df = pd.DataFrame(export_data)
            df.to_csv(file_path, index=False)
            
            messagebox.showinfo("Success", f"Results exported to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results:\n{str(e)}")
    
    def run(self):
        self.root.mainloop()
        
    def __del__(self):
        if self.nd2_file:
            self.nd2_file.close()

if __name__ == "__main__":
    if not ND2_AVAILABLE:
        print("Installing nd2reader...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nd2reader"])
        print("Please restart the program.")
    else:
        app = CellAnalyzer()
        app.run()