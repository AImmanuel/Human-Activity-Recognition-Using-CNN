import os
import numpy as np
import shutil

def downsample_DS(source, output, delete_original):
    
    dataset = []
    for root, _, files in os.walk(source):
        for file in files:
            # Check if filetype is NumPy (.npy) and extract data
            if file.endswith('.npy'):
                full_path = os.path.join(root, file)
                get_data = np.load(full_path, allow_pickle=True).item()
                get_label = get_data['label']
                dataset.append((full_path, get_label))
                
    # Activity ID 1-5 are Fall cases and rest are Non-Fall
    falls = [info for info in dataset if 1 <= info[1] <= 5]
    non_falls = [info for info in dataset if info[1] > 5]
    
    # Count 80% of fall windows
    count_falls = int(len(falls) * 0.8)
    
    # Choose 80% of the fall windows and equal number of non-fall windows
    # Without replacement
    chosen_fall_index = np.random.choice(len(falls), size = count_falls, replace = False)
    chosen_falls = [falls[i] for i in chosen_fall_index]

    chosen_non_fall_index = np.random.choice(len(non_falls), size = count_falls, replace = False)
    chosen_non_falls = [non_falls[i] for i in chosen_non_fall_index]
    
    for full_path, _ in np.concatenate((chosen_falls, chosen_non_falls)):
        rel_path = os.path.relpath(full_path, source)
        new_full_path = os.path.join(output, rel_path)
        
        os.makedirs(os.path.dirname(new_full_path), exist_ok=True)
        shutil.copyfile(full_path, new_full_path)
        
        # Remove chosen falls and non-falls from source folder (no data leakage)
        if delete_original:
            os.remove(full_path)
        
