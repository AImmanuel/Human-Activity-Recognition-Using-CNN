import os
import numpy as np
import shutil

def downsample(source_folder, output_folder, delete_original):   
    data_records = []
    for dirpath, _, filenames in os.walk(source_folder):
        for filename in filenames:
            # Check if the file is a NumPy array (.npy extension)
            if filename.endswith('.npy'):
                full_path = os.path.join(dirpath, filename)
                content = np.load(full_path, allow_pickle=True).item()
                annotation = content['label']
                data_records.append((full_path, annotation))
    
    # Activity ID 1-5 are Falls and the rest are non-falls
    fall_records = [record for record in data_records if 1 <= record[1] <= 5]
    non_fall_records = [record for record in data_records if record[1] > 5]
    num_fall_samples = int(len(fall_records) * 0.8)
    
    # Randomly choose 80% of the fall and equal number of non-fall windows without replacement
    chosen_fall_indices = np.random.choice(len(fall_records), size=num_fall_samples, replace=False)
    selected_fall_records = [fall_records[i] for i in chosen_fall_indices]
    chosen_non_fall_indices = np.random.choice(len(non_fall_records), size=num_fall_samples, replace=False)
    selected_non_fall_records = [non_fall_records[i] for i in chosen_non_fall_indices]
    
    for full_path, _ in np.concatenate((selected_fall_records, selected_non_fall_records)):
        relative_path = os.path.relpath(full_path, source_folder)
        new_full_path = os.path.join(output_folder, relative_path)
        os.makedirs(os.path.dirname(new_full_path), exist_ok=True)
        shutil.copyfile(full_path, new_full_path)
    
    # Remove selected windows from the source folder
    if delete_original:
        os.remove(full_path)

if __name__ == "__main__":
    source_folder = "C:/Users/ac22aci/Desktop/nparray_raw_38x51"
    output_folder = "C:/Users/ac22aci/Desktop/nparray_raw_38x51_bal"
    
    downsample(source_folder, output_folder, remove_original = True)