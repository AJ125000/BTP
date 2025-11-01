import h5py
import numpy as np

# Open the HDF5 file
file_path = r'zenodo_3d_overthrust_model/overthrust_3D_true_model.h5'

with h5py.File(file_path, 'r') as f:
    print("Keys and Data Types in the HDF5 file:")
    print("=" * 60)
    
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}")
            print(f"  - Shape: {obj.shape}")
            print(f"  - Data Type: {obj.dtype}")
            print(f"  - Size: {obj.size}")
            print("-" * 60)
        elif isinstance(obj, h5py.Group):
            print(f"Group: {name}")
            print("-" * 60)
    
    # Visit all items in the file
    f.visititems(print_structure)
    
    # Also print top-level keys
    print("\nTop-level keys:")
    print(list(f.keys()))
    
    m = f['m'][:]
    print(f"Velocity model 'm': min={m.min()}, max={m.max()}")

    d = f['d'][:]
    print(f"Grid spacing 'd': {d}")


