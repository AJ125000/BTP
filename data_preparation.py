"""
Data Preparation Pipeline for High-Resolution Seismic Imaging
CORRECTED VERSION - Fixes critical grid spacing and free surface issues
Based on: "Deep learning for high-resolution seismic imaging" (Scientific Reports 2024)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import h5py
from tqdm import tqdm
import os

# ============================================================================
# STEP 1: LOAD THE SEG/EAGE 3D OVERTHRUST MODEL (UNCHANGED)
# ============================================================================

def load_overthrust_model_h5(filepath):
    """
    Load the SEG/EAGE 3D Overthrust velocity model from HDF5 file.
    
    HDF5 structure:
    - 'm': velocity model, shape (207, 801, 801) = (nz, nx, ny) in m/s
    - 'n': dimensions [nz, nx, ny]
    - 'd': grid spacing [dz, dx, dy] in meters
    - 'o': origin [oz, ox, oy]
    """
    print("="*70)
    print("LOADING 3D OVERTHRUST MODEL FROM HDF5")
    print("="*70)
    print(f"File: {filepath}\n")
    
    with h5py.File(filepath, 'r') as f:
        velocity_model_raw = f['m'][:]
        dimensions = f['n'][:]
        grid_spacing = f['d'][:]
        origin = f['o'][:]
        
        print("Model Information:")
        print(f"  Raw shape (nz, nx, ny): {velocity_model_raw.shape}")
        print(f"  Dimensions [nz, nx, ny]: {dimensions}")
        print(f"  Grid spacing [dz, dx, dy]: {grid_spacing} meters")
        print(f"  Origin [oz, ox, oy]: {origin}")
    
    # Transpose to (nx, ny, nz)
    velocity_model = np.transpose(velocity_model_raw, (1, 2, 0))
    
    # Convert from m/s to km/s
    velocity_model = velocity_model / 1000.0
    
    spacing = {
        'dx': float(grid_spacing[1]),
        'dy': float(grid_spacing[2]),
        'dz': float(grid_spacing[0]),
        'origin': origin.tolist()
    }
    
    print(f"\nProcessed Model:")
    print(f"  Final shape (nx, ny, nz): {velocity_model.shape}")
    print(f"  Velocity range: {velocity_model.min():.3f} - {velocity_model.max():.3f} km/s")
    print(f"  Grid spacing: dx={spacing['dx']}m, dy={spacing['dy']}m, dz={spacing['dz']}m")
    print("="*70 + "\n")
    
    return velocity_model, spacing


# ============================================================================
# STEP 2: EXTRACT DIVERSE 2D VELOCITY MODELS (UNCHANGED)
# ============================================================================

def extract_2d_slices(velocity_3d, num_slices=3000, slice_type='mixed', seed=42):
    """Extract 3000 diverse 2D velocity models from 3D Overthrust model."""
    print("="*70)
    print(f"EXTRACTING {num_slices} DIVERSE 2D SLICES")
    print("="*70)
    
    np.random.seed(seed)
    
    nx, ny, nz = velocity_3d.shape
    print(f"3D Model shape: nx={nx}, ny={ny}, nz={nz}")
    print(f"Slice type: {slice_type}\n")
    
    slices_2d = []
    slice_info = []
    
    if slice_type == 'mixed':
        num_per_type = num_slices // 3
        
        print(f"Extracting {num_per_type} inline slices (yz-plane)...")
        x_indices = np.linspace(50, nx-50, num_per_type, dtype=int)
        for x_idx in tqdm(x_indices, desc="Inline"):
            slice_2d = velocity_3d[x_idx, :, :]
            slices_2d.append(slice_2d)
            slice_info.append({
                'type': 'inline',
                'index': int(x_idx),
                'original_shape': slice_2d.shape
            })
        
        print(f"Extracting {num_per_type} crossline slices (xz-plane)...")
        y_indices = np.linspace(50, ny-50, num_per_type, dtype=int)
        for y_idx in tqdm(y_indices, desc="Crossline"):
            slice_2d = velocity_3d[:, y_idx, :]
            slices_2d.append(slice_2d)
            slice_info.append({
                'type': 'crossline',
                'index': int(y_idx),
                'original_shape': slice_2d.shape
            })
        
        remaining = num_slices - 2*num_per_type
        print(f"Extracting {remaining} depth slices (xy-plane)...")
        z_indices = np.linspace(10, nz-10, remaining, dtype=int)
        for z_idx in tqdm(z_indices, desc="Depth"):
            slice_2d = velocity_3d[:, :, z_idx]
            slices_2d.append(slice_2d)
            slice_info.append({
                'type': 'depth',
                'index': int(z_idx),
                'original_shape': slice_2d.shape
            })
    
    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"Total slices extracted: {len(slices_2d)}")
    print(f"  Inline: {sum(1 for s in slice_info if s['type']=='inline')}")
    print(f"  Crossline: {sum(1 for s in slice_info if s['type']=='crossline')}")
    print(f"  Depth: {sum(1 for s in slice_info if s['type']=='depth')}")
    print(f"{'='*70}\n")
    
    return slices_2d, slice_info


# ============================================================================
# STEP 3: GENERATE RICKER WAVELET (UNCHANGED)
# ============================================================================

def ricker_wavelet(freq, dt, nt):
    """
    Generate a Ricker wavelet with specified frequency.
    Paper: "30 Hz Ricker wavelet as source signal"
    """
    time = np.arange(nt) * dt
    t0 = 1.0 / freq
    time_centered = time - t0
    
    a = (np.pi * freq * time_centered) ** 2
    wavelet = (1 - 2 * a) * np.exp(-a)
    wavelet = wavelet / np.max(np.abs(wavelet))
    
    return wavelet, time


# ============================================================================
# STEP 4: FINITE-DIFFERENCE SOLVER - ** WITH CRITICAL FIXES **
# ============================================================================

def apply_pml_damping(field, pml_width, pml_strength=0.015):
    """
    Apply Perfectly Matched Layer (PML) absorbing boundary conditions.
    Paper: "Three PML at bottom, left, and right sides"
    """
    nx, nz = field.shape
    
    # Left boundary
    for i in range(pml_width):
        damp = np.exp(-pml_strength * (pml_width - i) ** 2)
        field[i, :] *= damp
    
    # Right boundary
    for i in range(pml_width):
        damp = np.exp(-pml_strength * i ** 2)
        field[nx - pml_width + i, :] *= damp
    
    # Bottom boundary (NO PML at top - free surface)
    for i in range(pml_width):
        damp = np.exp(-pml_strength * i ** 2)
        field[:, nz - pml_width + i] *= damp
    
    return field


def finite_difference_acoustic_2d(velocity_model, source_wavelet, dt, dx, dz, 
                                   source_position, receiver_positions, pml_width=10):
    """
    Solve 2D acoustic wave equation using finite-difference in time domain.
    
    ** CRITICAL FIX #2: Enforces free surface boundary condition **
    
    Paper specifications:
    - "Finite-difference modeling in time domain"
    - "Ricker wavelet 30 Hz source"
    - "Three PML at bottom, left, right"
    - "Free boundary conditions at top" <- FIXED
    
    Parameters:
    -----------
    velocity_model : numpy array (nx, nz)
        2D velocity model in km/s
    source_wavelet : numpy array
        Source time function (Ricker wavelet)
    dt : float
        Time step in seconds
    dx, dz : float
        Spatial grid spacing in meters (CORRECTED via Fix #1)
    source_position : tuple (x, z)
        Source location
    receiver_positions : list of tuples
        Receiver locations
    pml_width : int
        PML boundary width
        
    Returns:
    --------
    seismic_record : numpy array (n_receivers, n_time_steps)
        Recorded seismic traces
    """
    nx, nz = velocity_model.shape
    nt = len(source_wavelet)
    n_receivers = len(receiver_positions)
    
    # Convert velocity from km/s to m/s
    velocity = velocity_model * 1000.0
    
    # Initialize wavefields
    u = np.zeros((nx, nz), dtype=np.float32)
    u_prev = np.zeros((nx, nz), dtype=np.float32)
    u_next = np.zeros((nx, nz), dtype=np.float32)
    
    # CFL stability check
    vmax = np.max(velocity)
    cfl = vmax * dt * np.sqrt(1.0/(dx**2) + 1.0/(dz**2))
    if cfl > 1.0:
        print(f"  Warning: CFL={cfl:.3f} > 1.0. May be unstable!")
    
    # Precompute coefficients
    vel_sq = velocity ** 2
    c_x = (vel_sq * dt**2) / (dx**2)
    c_z = (vel_sq * dt**2) / (dz**2)
    
    # Storage for seismic records
    seismic_record = np.zeros((n_receivers, nt), dtype=np.float32)
    
    # Time stepping loop
    for it in range(nt):
        # Compute Laplacian using 2nd-order finite difference
        laplacian = np.zeros_like(u)
        
        # x-direction second derivative
        laplacian[1:-1, :] += c_x[1:-1, :] * (u[2:, :] - 2*u[1:-1, :] + u[:-2, :])
        
        # z-direction second derivative
        laplacian[:, 1:-1] += c_z[:, 1:-1] * (u[:, 2:] - 2*u[:, 1:-1] + u[:, :-2])
        
        # Update wavefield (acoustic wave equation)
        u_next = 2 * u - u_prev + laplacian
        
        # Inject source at source position
        sx, sz = source_position
        if 0 <= sx < nx and 0 <= sz < nz:
            u_next[sx, sz] += source_wavelet[it] * vel_sq[sx, sz] * dt**2
        
        # *** CRITICAL FIX #2: Enforce free surface at top ***
        # Paper: "free boundary conditions at the models' top"
        # This implements a pressure-release free surface
        u_next[:, 0] = 0.0
        
        # Apply PML absorbing boundaries (bottom, left, right)
        u_next = apply_pml_damping(u_next, pml_width)
        
        # Record at receiver positions
        for ir, (rx, rz) in enumerate(receiver_positions):
            if 0 <= rx < nx and 0 <= rz < nz:
                seismic_record[ir, it] = u[rx, rz]
        
        # Update time steps
        u_prev = u.copy()
        u = u_next.copy()
    
    return seismic_record


# ============================================================================
# STEP 5: REFLECTION COEFFICIENTS (UNCHANGED)
# ============================================================================

def compute_reflection_model(velocity_model, dx, dz):
    """
    Calculate reflection coefficients from velocity model.
    Paper: "calculating velocity differences between adjacent layers"
    """
    nx, nz = velocity_model.shape
    reflection_model = np.zeros_like(velocity_model)
    
    # Vertical reflection coefficients
    for i in range(nx):
        for j in range(1, nz):
            v1 = velocity_model[i, j-1]
            v2 = velocity_model[i, j]
            if (v1 + v2) > 0:
                reflection_model[i, j] = (v2 - v1) / (v2 + v1)
    
    # Horizontal reflection coefficients
    for i in range(1, nx):
        for j in range(nz):
            v1 = velocity_model[i-1, j]
            v2 = velocity_model[i, j]
            if (v1 + v2) > 0:
                r_h = (v2 - v1) / (v2 + v1)
                reflection_model[i, j] = np.sqrt(reflection_model[i, j]**2 + r_h**2)
    
    return reflection_model


# ============================================================================
# STEP 6: DATA GENERATION - ** WITH CRITICAL FIX #1 **
# ============================================================================

def generate_dataset(velocity_slices, slice_info, output_dir, 
                     freq=30.0, dt=0.001, t_max=2.0,
                     dx=10.0, dz=10.0, pml_width=10,
                     target_size=(256, 256)):
    """
    Complete data generation pipeline with CRITICAL FIXES.
    
    ** CRITICAL FIX #1: Corrects grid spacing after resizing **
    
    For each 2D velocity slice:
    1. Resize if needed
    2. COMPUTE EFFECTIVE GRID SPACING (dx_eff, dz_eff) <- FIX #1
    3. Run finite-difference simulation with corrected spacing
    4. Compute reflection coefficients with corrected spacing
    """
    os.makedirs(output_dir, exist_ok=True)
    
    nt = int(t_max / dt)
    wavelet, time_axis = ricker_wavelet(freq, dt, nt)
    
    print("="*70)
    print("GENERATING SEISMIC DATASET (WITH CRITICAL FIXES)")
    print("="*70)
    print(f"Number of models: {len(velocity_slices)}")
    print(f"Source: {freq} Hz Ricker wavelet")
    print(f"Time: {nt} samples (dt={dt}s, tmax={t_max}s)")
    print(f"Base grid spacing: dx={dx}m, dz={dz}m")
    print(f"Target size: {target_size}")
    print(f"PML width: {pml_width}")
    print(f"** FIX #1: Grid spacing corrected after resizing **")
    print(f"** FIX #2: Free surface enforced at top **")
    print("="*70 + "\n")
    
    all_seismic_data = []
    all_reflection_labels = []
    all_velocity_models = []
    metadata = []
    
    for idx, (velocity_2d, info) in enumerate(tqdm(zip(velocity_slices, slice_info), 
                                                     desc="Processing models", 
                                                     total=len(velocity_slices))):
        try:
            # Get original shape BEFORE resizing
            orig_shape = info['original_shape']
            orig_nx, orig_nz = orig_shape
            
            # *** CRITICAL FIX #1: Track physical size ***
            # Resize to target size if needed
            if velocity_2d.shape != target_size:
                zoom_factors = (target_size[0]/velocity_2d.shape[0], 
                               target_size[1]/velocity_2d.shape[1])
                velocity_2d = zoom(velocity_2d, zoom_factors, order=3)
                
                # *** CRITICAL: Compute effective grid spacing ***
                # Physical size must remain constant!
                # Original: orig_nx points × dx meters = total_distance
                # Resized: target_size[0] points × dx_eff meters = total_distance
                # Therefore: dx_eff = dx * (orig_nx / target_size[0])
                
                dx_effective = dx * (orig_nx / velocity_2d.shape[0])
                dz_effective = dz * (orig_nz / velocity_2d.shape[1])
                
                # Debug output (only for first few samples)
                if idx < 3:
                    print(f"\n  Sample {idx}: Resizing correction")
                    print(f"    Original shape: {orig_shape}")
                    print(f"    Resized shape: {velocity_2d.shape}")
                    print(f"    Base spacing: dx={dx}m, dz={dz}m")
                    print(f"    Effective spacing: dx={dx_effective:.2f}m, dz={dz_effective:.2f}m")
                    print(f"    Physical size maintained: {orig_nx*dx:.0f}m × {orig_nz*dz:.0f}m")
            else:
                # No resizing, use base spacing
                dx_effective = dx
                dz_effective = dz
            
            nx, nz = velocity_2d.shape
            
            # Source position (center-top)
            source_position = (nx // 2, pml_width + 5)
            
            # Receiver array
            n_receivers = min(128, nx - 2*pml_width)
            receiver_positions = [(i, pml_width + 2) 
                                 for i in np.linspace(pml_width, nx-pml_width, 
                                                     n_receivers, dtype=int)]
            
            # *** Generate seismic data with CORRECTED spacing (FIX #1) ***
            seismic_record = finite_difference_acoustic_2d(
                velocity_2d, 
                wavelet, 
                dt, 
                dx_effective,  # <- Use effective spacing!
                dz_effective,  # <- Use effective spacing!
                source_position, 
                receiver_positions, 
                pml_width
            )
            
            # *** Compute reflection model with CORRECTED spacing (FIX #1) ***
            reflection_model = compute_reflection_model(
                velocity_2d, 
                dx_effective,  # <- Use effective spacing!
                dz_effective   # <- Use effective spacing!
            )
            
            # Store
            all_seismic_data.append(seismic_record)
            all_reflection_labels.append(reflection_model)
            all_velocity_models.append(velocity_2d)
            metadata.append({
                'index': idx,
                'slice_type': info['type'],
                'slice_index': info['index'],
                'original_shape': orig_shape,
                'resized_shape': velocity_2d.shape,
                'dx_effective': dx_effective,
                'dz_effective': dz_effective,
                'source_pos': source_position,
                'n_receivers': n_receivers
            })
            
        except Exception as e:
            print(f"\nError processing slice {idx}: {e}")
            continue
    
    # Save to HDF5
    print(f"\nSaving dataset to HDF5...")
    dataset_path = os.path.join(output_dir, 'seismic_dataset.h5')
    
    with h5py.File(dataset_path, 'w') as hf:
        hf.create_dataset('seismic_data', 
                         data=np.array(all_seismic_data, dtype=np.float32), 
                         compression='gzip', compression_opts=4)
        hf.create_dataset('reflection_labels', 
                         data=np.array(all_reflection_labels, dtype=np.float32), 
                         compression='gzip', compression_opts=4)
        hf.create_dataset('velocity_models', 
                         data=np.array(all_velocity_models, dtype=np.float32), 
                         compression='gzip', compression_opts=4)
        
        # Save hyperparameters
        hf.attrs['freq'] = freq
        hf.attrs['dt'] = dt
        hf.attrs['t_max'] = t_max
        hf.attrs['dx_base'] = dx
        hf.attrs['dz_base'] = dz
        hf.attrs['n_samples'] = len(all_seismic_data)
        hf.attrs['target_size'] = target_size
        hf.attrs['pml_width'] = pml_width
        hf.attrs['fixes_applied'] = 'grid_spacing_correction, free_surface_boundary'
        
        # Save metadata
        meta_dt = np.dtype([
            ('index', 'i4'), 
            ('slice_type', 'S10'), 
            ('slice_index', 'i4'), 
            ('n_receivers', 'i4'),
            ('dx_eff', 'f4'),
            ('dz_eff', 'f4')
        ])
        meta_array = np.array([
            (m['index'], m['slice_type'].encode(), m['slice_index'], 
             m['n_receivers'], m['dx_effective'], m['dz_effective']) 
            for m in metadata
        ], dtype=meta_dt)
        hf.create_dataset('metadata', data=meta_array)
    
    file_size_mb = os.path.getsize(dataset_path) / (1024**2)
    
    print(f"\n{'='*70}")
    print(f"DATASET SAVED SUCCESSFULLY (WITH FIXES)")
    print(f"{'='*70}")
    print(f"Location: {dataset_path}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"\nData shapes:")
    print(f"  Seismic data: {np.array(all_seismic_data).shape}")
    print(f"  Reflection labels: {np.array(all_reflection_labels).shape}")
    print(f"  Velocity models: {np.array(all_velocity_models).shape}")
    print(f"\nCritical fixes applied:")
    print(f"  ✓ Grid spacing corrected after resizing (Fix #1)")
    print(f"  ✓ Free surface boundary enforced (Fix #2)")
    print(f"{'='*70}\n")
    
    return all_seismic_data, all_reflection_labels, metadata


# ============================================================================
# REMAINING STEPS (UNCHANGED - split_dataset, visualize_sample, main)
# ============================================================================

def split_dataset(dataset_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Split dataset into train/val/test (80/10/10)."""
    print("="*70)
    print("SPLITTING DATASET")
    print("="*70)
    print(f"Split ratios - Train: {train_ratio*100:.0f}%, Val: {val_ratio*100:.0f}%, Test: {test_ratio*100:.0f}%\n")
    
    np.random.seed(seed)
    
    with h5py.File(dataset_path, 'r') as hf:
        seismic_data = hf['seismic_data'][:]
        reflection_labels = hf['reflection_labels'][:]
        velocity_models = hf['velocity_models'][:]
        attrs = dict(hf.attrs)
    
    n_samples = len(seismic_data)
    indices = np.random.permutation(n_samples)
    
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    splits = {
        'train': (seismic_data[train_idx], reflection_labels[train_idx], velocity_models[train_idx]),
        'val': (seismic_data[val_idx], reflection_labels[val_idx], velocity_models[val_idx]),
        'test': (seismic_data[test_idx], reflection_labels[test_idx], velocity_models[test_idx])
    }
    
    output_dir = os.path.dirname(dataset_path)
    
    for split_name, (seismic, reflection, velocity) in splits.items():
        save_path = os.path.join(output_dir, f'{split_name}_data.h5')
        
        with h5py.File(save_path, 'w') as hf:
            hf.create_dataset('seismic', data=seismic, compression='gzip', compression_opts=4)
            hf.create_dataset('reflection', data=reflection, compression='gzip', compression_opts=4)
            hf.create_dataset('velocity', data=velocity, compression='gzip', compression_opts=4)
            
            for key, val in attrs.items():
                hf.attrs[key] = val
        
        file_size = os.path.getsize(save_path) / (1024**2)
        print(f"{split_name.upper():5s}: {len(seismic):4d} samples | {file_size:7.2f} MB | {save_path}")
    
    print("="*70 + "\n")
    
    return splits


def visualize_sample(seismic_data, reflection_label, velocity_model=None, 
                     save_path=None, idx=0):
    """Visualize input-output pair."""
    n_plots = 3 if velocity_model is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    
    if n_plots == 2:
        axes = [axes[0], axes[1]]
    
    # Seismic record (INPUT)
    im1 = axes[0].imshow(seismic_data, aspect='auto', cmap='seismic', 
                        extent=[0, seismic_data.shape[1], seismic_data.shape[0], 0])
    axes[0].set_title(f'Seismic Record (INPUT)\nSample {idx}', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Receiver Number')
    axes[0].set_ylabel('Time (samples)')
    axes[0].grid(alpha=0.3)
    plt.colorbar(im1, ax=axes[0], label='Amplitude')
    
    # Reflection model (LABEL)
    im2 = axes[1].imshow(reflection_label, aspect='auto', cmap='gray', vmin=-0.2, vmax=0.2)
    axes[1].set_title(f'Reflection Model (LABEL)\nSample {idx}', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Distance (grid points)')
    axes[1].set_ylabel('Depth (grid points)')
    axes[1].grid(alpha=0.3)
    plt.colorbar(im2, ax=axes[1], label='Reflection Coefficient')
    
    # Velocity model
    if velocity_model is not None:
        im3 = axes[2].imshow(velocity_model, aspect='auto', cmap='jet')
        axes[2].set_title(f'Velocity Model\nSample {idx}', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Distance (grid points)')
        axes[2].set_ylabel('Depth (grid points)')
        axes[2].grid(alpha=0.3)
        plt.colorbar(im3, ax=axes[2], label='Velocity (km/s)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved: {save_path}")
    
    plt.show()
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("DATA PREPARATION FOR HIGH-RESOLUTION SEISMIC IMAGING")
    print("** CORRECTED VERSION - WITH CRITICAL FIXES **")
    print("Based on: Deep learning for high-resolution seismic imaging")
    print("Scientific Reports (2024) 14:10319")
    print("="*70 + "\n")
    
    # Configuration
    CONFIG = {
        'overthrust_h5_path': 'zenodo_3d_overthrust_model/overthrust_3D_true_model.h5',
        'output_dir': './seismic_dataset',
        'num_slices': 3000,
        'source_freq': 30.0,
        'dt': 0.001,
        't_max': 2.0,
        'dx': 10.0,
        'dz': 10.0,
        'pml_width': 10,
        'target_size': (256, 256),
        'test_run': False,
        'test_samples': 10,
    }
    
    # STEP 1: Load model
    print("STEP 1: LOADING 3D OVERTHRUST MODEL")
    print("-" * 70 + "\n")
    
    try:
        velocity_3d, spacing = load_overthrust_model_h5(CONFIG['overthrust_h5_path'])
    except FileNotFoundError:
        print("\n" + "!"*70)
        print("ERROR: Overthrust model file not found!")
        print("Download from: https://zenodo.org/records/4252588")
        print(f"Save to: {CONFIG['overthrust_h5_path']}")
        print("!"*70 + "\n")
        exit(1)
    
    # STEP 2: Extract slices
    print("STEP 2: EXTRACTING 2D SLICES")
    print("-" * 70 + "\n")
    
    num_to_process = CONFIG['test_samples'] if CONFIG['test_run'] else CONFIG['num_slices']
    print(f"Mode: {'TEST RUN' if CONFIG['test_run'] else 'FULL RUN'}")
    print(f"Processing {num_to_process} slices\n")
    
    velocity_slices, slice_info = extract_2d_slices(
        velocity_3d,
        num_slices=num_to_process,
        slice_type='mixed'
    )
    
    # STEP 3-6: Generate dataset WITH FIXES
    print("STEP 3-6: GENERATING SEISMIC DATASET (WITH FIXES)")
    print("-" * 70 + "\n")
    
    seismic_data, reflection_labels, metadata = generate_dataset(
        velocity_slices,
        slice_info,
        CONFIG['output_dir'],
        freq=CONFIG['source_freq'],
        dt=CONFIG['dt'],
        t_max=CONFIG['t_max'],
        dx=CONFIG['dx'],
        dz=CONFIG['dz'],
        pml_width=CONFIG['pml_width'],
        target_size=CONFIG['target_size']
    )
    
    # STEP 7: Split dataset
    print("STEP 7: SPLITTING DATASET")
    print("-" * 70 + "\n")
    
    dataset_path = os.path.join(CONFIG['output_dir'], 'seismic_dataset.h5')
    splits = split_dataset(dataset_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    
    # STEP 8: Visualize
    print("STEP 8: VISUALIZING SAMPLES")
    print("-" * 70 + "\n")
    
    for i in range(min(3, len(seismic_data))):
        save_path = os.path.join(CONFIG['output_dir'], f'sample_visualization_{i}.png')
        visualize_sample(
            seismic_data[i],
            reflection_labels[i],
            velocity_slices[i],
            save_path=save_path,
            idx=i
        )
    
    # Summary
    print("\n" + "="*70)
    print("DATA PREPARATION COMPLETE (WITH CRITICAL FIXES)")
    print("="*70)
    print(f"\nDataset location: {CONFIG['output_dir']}")
    print(f"Total samples: {len(seismic_data)}")
    print(f"\nCritical fixes applied:")
    print(f"  ✓ FIX #1: Grid spacing corrected after resizing")
    print(f"  ✓ FIX #2: Free surface boundary enforced at top")
    print(f"\nNext steps:")
    if CONFIG['test_run']:
        print(f"  1. Verify fixes worked correctly (check visualizations)")
        print(f"  2. Set test_run=False for full 3000 samples")
    print(f"  3. Train Transformer-CNN-ASFF network")
    print("="*70 + "\n")
