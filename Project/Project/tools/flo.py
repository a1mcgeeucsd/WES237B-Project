import os
import struct
import numpy as np
import sys

def validate_flo_file(filepath):
    """
    Validates a Middlebury .flo file for structural integrity and header correctness.
    """
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' does not exist.")
        return False

    file_size = os.path.getsize(filepath)
    
    # Minimal size check: 12 bytes header + at least 1 pixel (8 bytes)
    if file_size < 20:
        print(f"Error: File is too small ({file_size} bytes). It cannot be a valid .flo file.")
        return False

    with open(filepath, 'rb') as f:
        # 1. Check Magic Number / Tag
        # Middlebury format uses the float value 202021.25 as a signature ('PIEH' in ASCII reverse)
        tag = struct.unpack('f', f.read(4))[0]
        if not np.isclose(tag, 202021.25):
            print(f"Error: Invalid magic number tag ({tag}). Expected 202021.25.")
            return False

        # 2. Extract Dimensions
        width = struct.unpack('i', f.read(4))[0]
        height = struct.unpack('i', f.read(4))[0]

        if width <= 0 or height <= 0:
            print(f"Error: Invalid image dimensions found in header: {width}x{height}.")
            return False

        # 3. Size Matching Verification
        # Header (12 bytes) + Width * Height * 2 channels (U, V) * 4 bytes per float32
        expected_size = 12 + (width * height * 2 * 4)

        print(f"File Metadata for: {os.path.basename(filepath)}")
        print(f"   • Dimensions: {width} x {height}")
        print(f"   • Disk Size:  {file_size} bytes")
        print(f"   • Expected:   {expected_size} bytes")

        if file_size != expected_size:
            print(f"Error: File size mismatch! This file will crash C++ tools like color_flow.")
            print(f"   Difference: {file_size - expected_size} bytes.")
            return False

        # 4. Content Sanity Check (NaN / Inf scanning)
        try:
            raw_data = np.fromfile(f, dtype=np.float32)
            nan_count = np.isnan(raw_data).sum()
            inf_count = np.isinf(raw_data).sum()
            
            if nan_count > 0 or inf_count > 0:
                print(f"Warning: Found {nan_count} NaNs and {inf_count} Infs in the flow vectors.")
            else:
                print("   • Data Content: Clean (No NaNs or Infs detected)")
        except Exception as e:
            print(f"Error while reading vector data: {e}")
            return False

    print("Success: The .flo file structure is perfectly valid!")
    return True

def read_flo(filepath):
    """Parses a Middlebury .flo file and returns a (H, W, 2) numpy array."""
    with open(filepath, 'rb') as f:
        tag = struct.unpack('f', f.read(4))[0]
        if not np.isclose(tag, 202021.25):
            raise ValueError(f"Invalid .flo file: {filepath}")
        width = struct.unpack('i', f.read(4))[0]
        height = struct.unpack('i', f.read(4))[0]
        data = np.fromfile(f, dtype=np.float32)
    return data.reshape((height, width, 2))

def benchmark_flow(pred_path, gt_path, unknown_threshold=1e9):
    """
    Benchmarks a predicted .flo file against ground truth.
    Calculates EPE, Fl-all (outliers), and Angular Error.
    """
    if not (os.path.exists(pred_path) and os.path.exists(gt_path)):
        print("❌ Error: One or both files do not exist.")
        return None

    # Load flow fields
    pred = read_flo(pred_path)
    gt = read_flo(gt_path)

    if pred.shape != gt.shape:
        print(f"❌ Error: Shape mismatch! Pred: {pred.shape}, GT: {gt.shape}")
        return None

    # Middlebury convention: Filter out pixels with unknown ground truth
    # Values greater than 1e9 signify invalid/unknown flow
    mask = (np.abs(gt[..., 0]) < unknown_threshold) & (np.abs(gt[..., 1]) < unknown_threshold)
    
    if not np.any(mask):
        print("❌ Error: No valid pixels found in ground truth file.")
        return None

    # Extract valid vectors
    pred_valid = pred[mask]
    gt_valid = gt[mask]

    # 1. Calculate Endpoint Error (EPE)
    du = pred_valid[:, 0] - gt_valid[:, 0]
    dv = pred_valid[:, 1] - gt_valid[:, 1]
    epe = np.sqrt(du**2 + dv**2)
    
    mean_epe = np.mean(epe)
    median_epe = np.median(epe)

    # 2. Calculate Fl-all (Outliers: EPE > 3 pixels OR > 5% of ground truth magnitude)
    gt_mag = np.sqrt(gt_valid[:, 0]**2 + gt_valid[:, 1]**2)
    outliers = (epe > 3.0) & (epe > 0.05 * gt_mag)
    fl_all = np.mean(outliers) * 100  # Percentage

    # 3. Calculate Angular Error (AE) in degrees
    # Vectors are represented in 3D space as (u, v, 1)
    num = 1.0 + pred_valid[:, 0] * gt_valid[:, 0] + pred_valid[:, 1] * gt_valid[:, 1]
    den = np.sqrt(1.0 + pred_valid[:, 0]**2 + pred_valid[:, 1]**2) * np.sqrt(1.0 + gt_valid[:, 0]**2 + gt_valid[:, 1]**2)
    
    # Clip to avoid floating point precision errors outside [-1, 1]
    ae = np.arccos(np.clip(num / den, -1.0, 1.0))
    mean_ae = np.degrees(np.mean(ae))

    # Print Report
    print(f"📊 OPTICAL FLOW BENCHMARK REPORT")
    print(f"   • Evaluated Pixels: {np.sum(mask)} / {mask.size} ({np.mean(mask)*100:.1f}%)")
    print(f"   • Mean EPE:         {mean_epe:.4f} pixels")
    print(f"   • Median EPE:       {median_epe:.4f} pixels")
    print(f"   • Outlier Ratio (F1): {fl_all:.2f} %")
    print(f"   • Mean Angular Error: {mean_ae:.2f} °")

    return {
        "mean_epe": mean_epe,
        "median_epe": median_epe,
        "fl_all": fl_all,
        "mean_ae": mean_ae
    }

def get_average_magnitude(filepath, unknown_threshold=1e9):
    """
    Calculates the average vector magnitude of a valid .flo file.
    """
    if not os.path.exists(filepath):
        print(f"❌ Error: File '{filepath}' does not exist.")
        return None

    # 1. Parse the Middlebury binary file
    with open(filepath, 'rb') as f:
        tag = struct.unpack('f', f.read(4))[0]
        if not np.isclose(tag, 202021.25):
            raise ValueError(f"Invalid .flo file structure or header in {filepath}")
        
        width = struct.unpack('i', f.read(4))[0]
        height = struct.unpack('i', f.read(4))[0]
        data = np.fromfile(f, dtype=np.float32)
    
    flow = data.reshape((height, width, 2))

    # 2. Extract U (horizontal) and V (vertical) channels
    u = flow[..., 0]
    v = flow[..., 1]

    # 3. Mask out invalid/unknown pixels (Middlebury convention)
    valid_mask = (np.abs(u) < unknown_threshold) & (np.abs(v) < unknown_threshold)
    
    if not np.any(valid_mask):
        print("⚠️ Warning: No valid pixels found in this flow file.")
        return 0.0

    # 4. Calculate magnitudes for valid pixels
    # Magnitude = sqrt(u^2 + v^2)
    magnitudes = np.sqrt(u[valid_mask]**2 + v[valid_mask]**2)
    
    # 5. Compute statistics
    avg_magnitude = np.mean(magnitudes)
    max_magnitude = np.max(magnitudes)
    median_magnitude = np.median(magnitudes)
    
    print(f"📊 FLOW MAGNITUDE ANALYSIS: {os.path.basename(filepath)}")
    print(f"   • Valid Pixels:      {np.sum(valid_mask)} / {valid_mask.size} ({np.mean(valid_mask)*100:.1f}%)")
    print(f"   • Average Magnitude: {avg_magnitude:.4f} pixels/frame")
    print(f"   • Median Magnitude:  {median_magnitude:.4f} pixels/frame")
    print(f"   • Max Velocity:      {max_magnitude:.4f} pixels/frame")

    return avg_magnitude

validate_flo_file(sys.argv[1])
get_average_magnitude(sys.argv[2])
benchmark_flow(sys.argv[1], sys.argv[2])