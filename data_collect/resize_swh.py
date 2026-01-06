#!/usr/bin/env python3
"""
Resize 'swh' variable in sfc_regular npz files from 96x96 to 192x192 using bilinear interpolation.
"""

import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom
import glob
from tqdm import tqdm
import shutil

def resize_swh_bilinear(swh_data, target_size=(192, 192)):
    """
    Resize swh data from current size to target size using bilinear interpolation.
    
    Args:
        swh_data (numpy.ndarray): Input swh data with shape (96, 96)
        target_size (tuple): Target size (192, 192)
    
    Returns:
        numpy.ndarray: Resized swh data with shape (192, 192)
    """
    current_shape = swh_data.shape
    
    # Calculate zoom factors for each dimension
    zoom_factors = [target_size[i] / current_shape[i] for i in range(len(current_shape))]
    
    # Use scipy.ndimage.zoom for bilinear interpolation
    resized_data = zoom(swh_data, zoom_factors, order=1, mode='reflect')
    
    return resized_data

def process_single_file(file_path, backup=True):
    """
    Process a single npz file to resize the swh variable.
    
    Args:
        file_path (str): Path to the npz file
        backup (bool): Whether to create a backup of the original file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load the original data
        data = np.load(file_path)
        
        # Check if swh exists and has the expected shape
        if 'swh' not in data:
            print(f"Warning: 'swh' not found in {file_path}")
            return False
            
        swh_original = data['swh']
        
        if swh_original.shape != (96, 96):
            print(f"Warning: swh shape is {swh_original.shape} instead of (96, 96) in {file_path}")
            # If it's already 192x192, skip processing
            if swh_original.shape == (192, 192):
                print(f"Skipping {file_path} - swh already at target size")
                return True
            
        # Create backup if requested
        if backup:
            backup_path = file_path + '.backup'
            if not os.path.exists(backup_path):
                shutil.copy2(file_path, backup_path)
        
        # Resize swh using bilinear interpolation
        swh_resized = resize_swh_bilinear(swh_original, (192, 192))
        
        # Prepare new data dictionary
        new_data = {}
        for key in data.keys():
            if key == 'swh':
                new_data[key] = swh_resized
            else:
                new_data[key] = data[key]
        
        # Save the modified data back to the file
        #np.savez_compressed(file_path, **new_data)
        np.savez(file_path, **new_data)
        print(f"Successfully processed {os.path.basename(file_path)}: swh resized from {swh_original.shape} to {swh_resized.shape}")
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def process_all_files(directory_path, backup=True):
    """
    Process all npz files in the given directory.
    
    Args:
        directory_path (str): Path to the directory containing npz files
        backup (bool): Whether to create backups of original files
    """
    # Find all npz files
    npz_files = glob.glob(os.path.join(directory_path, "*.npz"))
    
    if not npz_files:
        print("No npz files found in the directory!")
        return
    
    print(f"Found {len(npz_files)} npz files to process")
    
    # Process files with progress bar
    successful = 0
    failed = 0
    
    for file_path in tqdm(npz_files, desc="Processing files"):
        if process_single_file(file_path, backup=backup):
            successful += 1
        else:
            failed += 1
    
    print(f"\nProcessing completed!")
    print(f"Successfully processed: {successful} files")
    print(f"Failed to process: {failed} files")

def test_single_file(file_path):
    """
    Test the resizing on a single file and show before/after comparison.
    
    Args:
        file_path (str): Path to the test file
    """
    print(f"Testing resizing on: {file_path}")
    
    # Load original data
    data = np.load(file_path)
    swh_original = data['swh']
    
    print(f"Original swh shape: {swh_original.shape}")
    print(f"Original swh min/max: {swh_original.min():.4f} / {swh_original.max():.4f}")
    print(f"Original swh mean: {swh_original.mean():.4f}")
    
    # Resize
    swh_resized = resize_swh_bilinear(swh_original, (192, 192))
    
    print(f"Resized swh shape: {swh_resized.shape}")
    print(f"Resized swh min/max: {swh_resized.min():.4f} / {swh_resized.max():.4f}")
    print(f"Resized swh mean: {swh_resized.mean():.4f}")
    
    return swh_original, swh_resized

if __name__ == "__main__":
    # Current directory (should be the sfc/regular directory)
    current_dir = "test/sfc/regular"
    
    print("SWH Resize Tool")
    print("="*50)
    print(f"Working directory: {current_dir}")
    
    # Process all files without user interaction
    # No backup is created to directly overwrite original files
    print("\nStarting automatic processing of all npz files...")
    process_all_files(current_dir, backup=False)
