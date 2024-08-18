import os
import numpy as np
import pandas as pd
import datetime
import shutil


file_path = "C:/Users/ac22aci/Desktop/Clones/Features_1&0.5_Vision.csv"
labels_data_frame = pd.read_csv(file_path, skiprows=1)
folders = ["C:/Users/ac22aci/Desktop/Clones/Unbalanced"]    

for folder in folders:
    backup_dir = '../Outputs/DL/backup'       
    print(f"File: {folder}")
    label_dict = {}
    backup_dir(folder, backup_dir)
    for dir_root, subdirs, file_list in os.walk(folder):
        for file_name in file_list:
            if file_name.endswith('.npy'):
                file_parts = file_name.split('_')
                if len(file_parts) < 4:  # Skip files if parts < 4
                    print(f"Invalid filename {file_name}, skipping.")
                    continue                    
                time_stamp = '_'.join(file_parts[1:])  # Join the timestamp
                time_stamp = time_stamp.rsplit('.', 1)[0]  # Remove file extension
                time_stamp = time_stamp.replace('_', ':', 2)
                print(f"Processing {file_name} with timestamp {time_stamp}")

                matching_rows = labels_data_frame[labels_data_frame['Timestamp'].str.contains(time_stamp, na=False)]                    
                if not matching_rows.empty:
                    label_row = matching_rows.iloc[0]
                    tag = label_row['Tag']
                    npy_file_path = os.path.join(dir_root, file_name)
                    npy_array = np.load(npy_file_path, allow_pickle=True)
                    if isinstance(npy_array, dict) and 'array' in npy_array:
                        npy_array['label'] = tag
                    else:
                        npy_array = {'array': npy_array, 'label': tag}
                    np.save(npy_file_path, npy_array)
                    label_dict[file_name] = tag
                else:
                    print(f"No label found for {file_name}, deleting the file.")
                    os.remove(os.path.join(dir_root, file_name))
                    print(f"No label found for {file_name}")

    for file_name, label_assigned in label_dict.items():
        print(f"Verifying {file_name}")
        file_parts = file_name.split('_')
        formatted_timestamp = f"{file_parts[1]}T{file_parts[2]}:{file_parts[3].split('.', 1)[0]}"
        matching_row = labels_data_frame[labels_data_frame['Timestamp'].str.contains(formatted_timestamp, na=False)]
        
        if not matching_row.empty:
            original_tag = matching_row.iloc[0]['Tag']
            if label_assigned != original_tag:
                print(f"Label mismatch for {file_name}: assigned {label_assigned}, original {original_tag}")
