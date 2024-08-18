import os
import numpy as np
import pandas as pd
import datetime
import shutil


if __name__ == "__main__":   
     
    csv_path = ["C:/Users/ac22aci/Desktop/Clones/Features_1&0.5_Vision.csv"]
    label_dataframe = pd.read_csv(csv_path, skiprows=1)

    source_folders = ["C:/Users/ac22aci/Desktop/Clones/Unbalanced"]    
    
    for source_folder in source_folders:
        print(f"File: {source_folder}")
        file_label_dct = {}

        for root, dirs, files in os.walk(source_folder):
            for file in files:
                if file.endswith('.npy'):
                    parts = file.split('_')
                    #Skip file if it has less than 4 parts
                    if len(parts) < 4:  
                        print(f"Filename Invalid: {file}, Skipped.")
                        continue

                    #Filename operations
                    time_stamp = '_'.join(parts[1:])  
                    time_stamp = time_stamp.rsplit('.', 1)[0]  
                    time_stamp = time_stamp.replace('_', ':', 2)
                    print(f"Processing {file} timestamp: {time_stamp}")

                    row_matching = label_dataframe[label_dataframe['Timestamp'].str.contains(time_stamp, na=False)]
                    
                    if not row_matching.empty:
                        label_row = row_matching.iloc[0]
                        label = label_row['Tag']

                        npy_file_path = os.path.join(root, file)
                        npy_array = np.load(npy_file_path, allow_pickle=True)
                        
                        # Replace existing label/add new label
                        if isinstance(npy_array, dict) and 'array' in npy_array:
                            npy_array['label'] = label
                        else:
                            npy_array = {'array': npy_array, 'label': label}
                        
                        # Save updated data back to NumPy file
                        np.save(npy_file_path, npy_array)
                        
                        # Update file_label_dict
                        file_label_dct[file] = label
                        
                    else:
                        print(f"Label not found for {file}, Deleted.")
                        os.remove(os.path.join(root, file))

                        print(f"Label not found for {file}")


        for filename, label_assigned in file_label_dct.items():
            print(f"verifying {filename}")
            parts = filename.split('_')
            time_stamp = f"{parts[1]}T{parts[2]}:{parts[3].split('.', 1)[0]}"
            
            # Find row in CSV file using timestamp and extract original label
            label_row = label_dataframe[label_dataframe['Timestamp'].str.contains(time_stamp, na=False)]
            if not label_row.empty:
                original_label = label_row.iloc[0]['Tag']
                
                # Check if the original and assigned label match
                if label_assigned != original_label:
                    print(f"Mismatched label in {filename}: Assigned Label: {label_assigned}, Original Label: {original_label}")
                    
    print(f"-------------------------------------------------------------")