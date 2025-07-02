"""
Batch processor for analyzing multiple ND2 files
"""

import os
import glob
import pandas as pd
from datetime import datetime
from nk_cancer_analyzer_phase4 import analyze_nk_killing

def batch_process_nd2_files(directory_path, output_base_dir=None):
    """
    Process all ND2 files in a directory.
    
    Args:
        directory_path: Path to directory containing ND2 files
        output_base_dir: Base directory for all outputs
        
    Returns:
        Summary DataFrame
    """
    # Find all ND2 files
    nd2_files = glob.glob(os.path.join(directory_path, "*.nd2"))
    
    if not nd2_files:
        print(f"No ND2 files found in {directory_path}")
        return None
    
    print(f"Found {len(nd2_files)} ND2 files to process")
    
    # Create output directory
    if output_base_dir is None:
        output_base_dir = os.path.join(directory_path, f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process each file
    all_summaries = []
    
    for i, nd2_file in enumerate(nd2_files):
        print(f"\n{'='*60}")
        print(f"Processing file {i+1}/{len(nd2_files)}: {os.path.basename(nd2_file)}")
        print(f"{'='*60}")
        
        # Create output directory for this file
        file_output_dir = os.path.join(output_base_dir, os.path.splitext(os.path.basename(nd2_file))[0])
        
        try:
            # Run analysis
            results = analyze_nk_killing(nd2_file, file_output_dir)
            
            if results:
                # Create summary for this file
                time_series = results['time_series']
                death_events = results['death_events']
                
                # Calculate summary metrics
                summary = {
                    'filename': os.path.basename(nd2_file),
                    'total_frames': time_series['timepoint'].max() + 1,
                    'duration_min': time_series['time_min'].max(),
                    'num_droplets': time_series['droplet_id'].nunique(),
                    'initial_cancer_cells': time_series[time_series['timepoint'] == 0]['cancer_cells_alive'].sum(),
                    'final_cancer_cells': time_series[time_series['timepoint'] == time_series['timepoint'].max()]['cancer_cells_alive'].sum(),
                    'total_deaths': len(death_events),
                    'max_nk_cells': time_series['nk_cells'].max(),
                    'avg_nk_cells': time_series['nk_cells'].mean(),
                    'killing_efficiency_%': 0
                }
                
                # Calculate killing efficiency
                if summary['initial_cancer_cells'] > 0:
                    summary['killing_efficiency_%'] = (summary['total_deaths'] / summary['initial_cancer_cells']) * 100
                
                # Add death timing if available
                if len(death_events) > 0:
                    summary['avg_death_time_min'] = death_events['death_time_min'].mean()
                    summary['first_death_min'] = death_events['death_time_min'].min()
                    summary['last_death_min'] = death_events['death_time_min'].max()
                else:
                    summary['avg_death_time_min'] = 'N/A'
                    summary['first_death_min'] = 'N/A'
                    summary['last_death_min'] = 'N/A'
                
                all_summaries.append(summary)
                print(f"✓ Successfully processed: {summary['total_deaths']} deaths detected")
                
        except Exception as e:
            print(f"✗ Error processing {os.path.basename(nd2_file)}: {str(e)}")
            all_summaries.append({
                'filename': os.path.basename(nd2_file),
                'error': str(e)
            })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(all_summaries)
    
    # Save batch summary
    summary_path = os.path.join(output_base_dir, 'batch_summary.xlsx')
    with pd.ExcelWriter(summary_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Format
        workbook = writer.book
        worksheet = workbook['Summary']
        
        # Header formatting
        from openpyxl.styles import PatternFill, Font, Alignment
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_font = Font(color="FFFFFF", bold=True)
        
        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")
        
        # Auto-adjust columns
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 40)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"Processed {len(nd2_files)} files")
    print(f"Results saved to: {output_base_dir}")
    print(f"Summary saved to: {summary_path}")
    
    # Create combined visualization
    create_batch_comparison_plots(all_summaries, output_base_dir)
    
    return summary_df


def create_batch_comparison_plots(summaries, output_dir):
    """Create comparison plots across all experiments."""
    import matplotlib.pyplot as plt
    
    # Filter out errors
    valid_summaries = [s for s in summaries if 'error' not in s]
    
    if not valid_summaries:
        print("No valid data to plot")
        return
    
    df = pd.DataFrame(valid_summaries)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Killing efficiency comparison
    ax = axes[0, 0]
    x = range(len(df))
    ax.bar(x, df['killing_efficiency_%'])
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Killing Efficiency (%)')
    ax.set_title('Killing Efficiency Across Experiments')
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('.nd2', '') for f in df['filename']], rotation=45, ha='right')
    
    # Plot 2: Initial vs Final cell counts
    ax = axes[0, 1]
    width = 0.35
    x = range(len(df))
    ax.bar([i - width/2 for i in x], df['initial_cancer_cells'], width, label='Initial', color='blue')
    ax.bar([i + width/2 for i in x], df['final_cancer_cells'], width, label='Final', color='red')
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Cancer Cell Count')
    ax.set_title('Initial vs Final Cancer Cell Counts')
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('.nd2', '') for f in df['filename']], rotation=45, ha='right')
    ax.legend()
    
    # Plot 3: Average death time
    ax = axes[1, 0]
    valid_death_times = df[df['avg_death_time_min'] != 'N/A']
    if len(valid_death_times) > 0:
        x = range(len(valid_death_times))
        ax.bar(x, valid_death_times['avg_death_time_min'])
        ax.set_xlabel('Experiment')
        ax.set_ylabel('Average Death Time (min)')
        ax.set_title('Average Time to Death')
        ax.set_xticks(x)
        ax.set_xticklabels([f.replace('.nd2', '') for f in valid_death_times['filename']], rotation=45, ha='right')
    
    # Plot 4: NK cell counts
    ax = axes[1, 1]
    ax.scatter(df['avg_nk_cells'], df['killing_efficiency_%'])
    ax.set_xlabel('Average NK Cells')
    ax.set_ylabel('Killing Efficiency (%)')
    ax.set_title('NK Cell Count vs Killing Efficiency')
    
    # Add trend line if enough data
    if len(df) > 2:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(df['avg_nk_cells'], df['killing_efficiency_%'])
        line_x = [df['avg_nk_cells'].min(), df['avg_nk_cells'].max()]
        line_y = [slope * x + intercept for x in line_x]
        ax.plot(line_x, line_y, 'r--', alpha=0.8, label=f'R² = {r_value**2:.3f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'batch_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()


# Quick test function
def test_batch_processing():
    """Test batch processing on your data directory."""
    # Update this path to your data directory
    data_directory = r"D:\New\BrainBites\Cell"
    
    # Process all ND2 files
    summary_df = batch_process_nd2_files(data_directory)
    
    if summary_df is not None:
        print("\nBatch Summary:")
        print(summary_df)


if __name__ == "__main__":
    test_batch_processing()