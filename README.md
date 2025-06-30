# Cell Analysis - Enhanced Droplet & Cancer Cell Detection

A comprehensive Python application for analyzing cell microscopy data from ND2 files, with enhanced droplet detection and TRITC-guided cancer cell analysis.

## Features

### Enhanced Droplet Detection
- **Multi-method detection**: Combines gradient-based, Hough transform, and template matching approaches
- **Thick and thin ring support**: Optimized for various droplet ring thicknesses
- **Overlap merging**: Intelligently merges duplicate detections
- **Enhanced validation**: Robust validation for both thick and thin ring droplets

### TRITC-Guided Cancer Cell Detection
- **Multi-threshold analysis**: Uses Otsu, percentile-based, and adaptive thresholds
- **Peak detection**: Local maxima analysis for cell identification
- **Watershed segmentation**: Advanced cell boundary detection
- **Viability assessment**: Classifies cells based on TRITC intensity levels

### User Interface
- **Modern GUI**: Large, accessible interface with enhanced fonts and styling
- **Real-time visualization**: Four-panel dashboard showing different analysis stages
- **Time-series navigation**: Slider control for multi-timepoint data
- **Channel selection**: Support for multiple fluorescence channels
- **Export functionality**: CSV export of analysis results

## Installation

### Prerequisites
```bash
pip install numpy matplotlib opencv-python pandas scipy scikit-image
```

### ND2 Reader Installation
```bash
pip install nd2reader
```

## Usage

1. **Run the application**:
   ```bash
   python cell_analyzer.py
   ```

2. **Load ND2 file**: Click "📁 Load ND2 File" and select your microscopy data

3. **Navigate data**: Use the timepoint slider and channel dropdown to explore your data

4. **Analyze**: Click "🔍 Analyze Current Frame" to detect droplets and cells

5. **Export results**: Click "💾 Export Results" to save analysis data as CSV

## Analysis Pipeline

1. **Image Preprocessing**: Normalization and noise reduction
2. **Droplet Detection**: Multi-method approach for robust detection
3. **Cell Analysis**: TRITC-guided cancer cell identification
4. **Viability Assessment**: Cell classification based on fluorescence intensity
5. **Results Visualization**: Real-time display of detection results

## Output

The application provides:
- **Droplet count and locations**
- **Cancer cell detection within droplets**
- **Cell viability assessment**
- **Exportable quantitative data**

## File Structure

```
Cell/
├── cell_analyzer.py    # Main application
├── 03.nd2             # Sample ND2 data file
├── 2.nd2              # Sample ND2 data file
└── README.md          # This documentation
```

## Technical Details

### Dependencies
- **tkinter**: GUI framework
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **opencv-python**: Image processing
- **pandas**: Data manipulation and export
- **scipy**: Scientific computing
- **scikit-image**: Advanced image analysis
- **nd2reader**: ND2 file format support

### Key Algorithms
- **Gradient-based detection**: Sobel operators for edge detection
- **Circular Hough Transform**: Multiple parameter sets for different droplet types
- **Template matching**: Ring pattern recognition
- **Watershed segmentation**: Cell boundary detection
- **Peak detection**: Local maxima analysis for cell centers

## License

This project is designed for research and educational purposes in cell biology and microscopy analysis.