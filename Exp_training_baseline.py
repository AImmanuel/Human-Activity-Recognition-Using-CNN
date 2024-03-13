import torch
import torch.nn as nn
import os
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from dataset_prep_3d import OpticalFlow3DDataset
import numpy as np
from timeit import default_timer as timer
from datetime import datetime
import cv2
import csv
import re
import seaborn as sns
import traceback


class TemporalCNN(nn.Module):
    def __init__(self):
        super(TemporalCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv3d(2, 64, (3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 128, (3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(128, 256, (3, 3, 3), padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.conv4 = nn.Conv3d(256, 256, (3, 3, 3), padding=1)
        self.bn4 = nn.BatchNorm3d(256)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 2) 

    def forward(self, t):
        t = F.relu(self.bn1(self.conv1(t)))
        t = F.max_pool3d(t, 2)
        t = F.relu(self.bn2(self.conv2(t)))
        t = F.max_pool3d(t, 2)
        t = F.relu(self.bn3(self.conv3(t)))
        t = F.max_pool3d(t, 2)
        t = F.relu(self.bn4(self.conv4(t)))
        t = self.global_avg_pool(t)
        t = t.view(t.size(0), -1)
        t = F.relu(self.fc1(t))
        t = self.dropout1(t)
        t = F.relu(self.fc2(t))
        t = self.dropout2(t)
        t = self.fc3(t)

        return t


class SpatialCNN(nn.Module):
    def __init__(self):
        super(SpatialCNN, self).__init__()

        # Convolutional layers
        self.conv1_1 = nn.Conv3d(2, 64, (3, 3, 3), padding=1)
        self.bn1_1 = nn.BatchNorm3d(64)
        self.conv2_1 = nn.Conv3d(64, 128, (3, 3, 3), padding=1)
        self.bn2_1 = nn.BatchNorm3d(128)
        self.conv3_1 = nn.Conv3d(128, 256, (3, 3, 3), padding=1)
        self.bn3_1 = nn.BatchNorm3d(256)
        self.conv4_1 = nn.Conv3d(256, 256, (3, 3, 3), padding=1)
        self.bn4_1 = nn.BatchNorm3d(256)

        # Global average pooling
        self.global_avg_pool_1 = nn.AdaptiveAvgPool3d(1)

        # Fully connected layers
        self.fc1_1 = nn.Linear(256, 128)
        self.dropout1_1 = nn.Dropout(0.5)
        self.fc2_1 = nn.Linear(128, 64)
        self.dropout2_1 = nn.Dropout(0.5)
        self.fc3_1 = nn.Linear(64, 2) 

    def forward(self, s):
        s = F.relu(self.bn1_1(self.conv1_1(s)))
        s = F.max_pool3d(s, 2)
        s = F.relu(self.bn2_1(self.conv2_1(s)))
        s = F.max_pool3d(s, 2)
        s = F.relu(self.bn3_1(self.conv3_1(s)))
        s = F.max_pool3d(s, 2)
        s = F.relu(self.bn4_1(self.conv4_1(s)))
        s = self.global_avg_pool_1(s)
        s = s.view(s.size(0), -1)
        s = F.relu(self.fc1_1(s))
        s = self.dropout1_1(s)
        s = F.relu(self.fc2_1(s))
        s = self.dropout2_1(s)
        s = self.fc3_1(s)

        return s
    



class DatasetDirectoryHandler:
    def __init__(self, base_folder):
        self.base_folder = base_folder

    def get_subject_folders(self):
        return self._get_subfolders(self.base_folder)

    def get_activity_folders(self, subject_folder):
        subject_path = os.path.join(self.base_folder, subject_folder)
        return self._get_subfolders(subject_path)

    def get_trial_folders(self, subject_folder, activity_folder):
        activity_path = os.path.join(self.base_folder, subject_folder, activity_folder)
        return self._get_subfolders(activity_path)

    def get_camera_folders(self, subject_folder, activity_folder, trial_folder):
        trial_path = os.path.join(self.base_folder, subject_folder, activity_folder, trial_folder)
        return self._get_subfolders(trial_path)

    def _get_subfolders(self, folder):
        folders = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        return sorted(folders, key=lambda x: int(re.search(r'\d+', x).group()))

class FrameLoader:
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        
    def get_video_folders(self):
        return[d for d in os.listdir(self.dataset_folder) if os.path.isdir(os.path.join(self.dataset_folder, d))]
        
    def load_frames_from_video(self, video_folder):
        image_folder = os.path.join(self.dataset_folder, video_folder)
        file_names = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')])
                
        return [(fn[:-4], cv2.imread(os.path.join(image_folder, fn), cv2.IMREAD_GRAYSCALE)) for fn in file_names]
    
class OpticalFlowComputer:
    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=30, detectShadows=True)
        self.fgbg.setShadowValue(0)
        self.fgbg.setShadowThreshold(0.5)
        self.learning_rate = -1
        
    def compute_optical_flow(self, prev_frame, current_frame):
        prev_frame = cv2.normalize(src=prev_frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        current_frame = cv2.normalize(src=current_frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        fgmask_prev = self.fgbg.apply(prev_frame, learningRate = self.learning_rate)
        fgmask_curr = self.fgbg.apply(current_frame, learningRate = self.learning_rate)
        
        kernel = np.ones((4,4), np.uint8)
        fgmask_prev = cv2.morphologyEx(fgmask_prev, cv2.MORPH_CLOSE, kernel)
        fgmask_prev = cv2.morphologyEx(fgmask_prev, cv2.MORPH_OPEN, kernel)
        fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_CLOSE, kernel)
        fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_OPEN, kernel)
        
        prev_frame_masked = cv2.bitwise_and(prev_frame, prev_frame, mask=fgmask_prev)
        current_frame_masked = cv2.bitwise_and(current_frame, current_frame, mask=fgmask_curr)
        
        # concatenated_frames = cv2.hconcat([current_frame, current_frame_masked])
        # cv2.imshow('Original vs Preprocessed Frame', concatenated_frames)
        # cv2.waitKey(1)
        
        prev_frame = cv2.medianBlur(prev_frame_masked, 5)
        current_frame = cv2.medianBlur(current_frame_masked, 5)
        
        prev_blurred = cv2.GaussianBlur(prev_frame, (5, 5), 0)
        curr_blurred = cv2.GaussianBlur(current_frame, (5, 5), 0)
        
        flow = cv2.calcOpticalFlowFarneback(prev_blurred, curr_blurred, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        u_component = flow[..., 0]
        v_component = flow[..., 1]

        # Uncomment below to view optical flow as it runs
        # magnitude, angle = cv2.cartToPolar(u_component, v_component, angleInDegrees=True)
        # hsv = np.zeros((prev_frame.shape[0], prev_frame.shape[1], 3), dtype=np.uint8)
        # hsv[..., 1] = 255
        # hsv[..., 0] = angle * 180 / np.pi / 2
        # hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # cv2.imshow('Optical Flow', rgb)
        # cv2.waitKey(1)

        #New Resize: 320x240
        resized_u = cv2.resize(u_component, (320, 240))
        resized_v = cv2.resize(v_component, (320, 240))
        
        return resized_u, resized_v
    
    def compute_resize(self, current_frame):
        current_frame = cv2.normalize(
            src=current_frame,
            dst=None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
        fgmask_curr = self.fgbg.apply(current_frame)
        kernel = np.ones((5, 5), np.uint8)
        fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_OPEN, kernel)
        fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_CLOSE, kernel)
        current_frame_masked = cv2.bitwise_and(
            current_frame, current_frame, mask=fgmask_curr
        )
        equalized_curr= cv2.equalizeHist(current_frame)
        blurred_img = cv2.blur(equalized_curr,ksize=(5,5))
        med_val = np.median(blurred_img) 
        lower = int(max(0 ,0.5*med_val))
        upper = int(min(255,1.5*med_val))
        current_frame_masked = cv2.Canny(current_frame_masked, lower, upper)
        
        #New Resize: 320x240
        resized_frame = cv2.resize(current_frame_masked, (320, 240))
        
        return resized_frame


class NumpyWriter:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        
    def write_array(self, array, name):
        file_path = os.path.join(self.output_folder, f"{name}.npy")
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        np.save(file_path, array)
        

class OpticalFlowProcessor:
    def __init__(self, dataset_folder, output_folder, fps = 18):
        self.frame_loader = FrameLoader(dataset_folder)
        self.numpy_writer = NumpyWriter(output_folder)
        self.fps = fps
        self.window_size = fps
        self.overlap = fps // 2
        self.optical_flow_computer = OpticalFlowComputer()
        
    def total_seconds_from_timestamp(timestamp: str) -> float:
        hours, minutes, seconds = map(float, timestamp.split('T')[1].split('_'))
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds
       
    def increment_timestamp(timestamp: str) -> str:
        date, time = timestamp.split('T')
        try:
            hours, minutes, remainder = time.split('_')
            seconds, ms = remainder.split('.')
        except ValueError:
            print(f"Error with timestamp: {time}")
            raise
        
        ms = int(ms)
        seconds = int(seconds)
        minutes = int(minutes)
        hours = int(hours)
        
        ms += 500000
        if ms >= 1000000:
            ms -= 1000000
            seconds += 1
        
        if seconds >= 60:
            seconds -= 60
            minutes += 1
        
        if minutes >= 60:
            minutes -= 60
            hours += 1

        time_str = f"{hours:02}_{minutes:02}_{seconds:02}.{ms:06}"
        return f"{date}T{time_str}"
    
    def process_video(self, video_folder):
        frames = self.frame_loader.load_frames_from_video(video_folder)
        #print(f"First frame timestamp for {video_folder}: {frames[0][0]}")
        #print(f"Last frame timestamp for {video_folder}: {frames[-1][0]}")

        i = 0
        num_frames_in_window = int(self.fps)
        overlap_frames = int(self.overlap)

        last_frame_time = frames[-1][0]
        last_frame_seconds = OpticalFlowProcessor.total_seconds_from_timestamp(last_frame_time)
        timestamp = frames[i][0]

        while i < len(frames) - num_frames_in_window:
            window_end = min(i + num_frames_in_window, len(frames))
            
            optical_flows_u= []
            optical_flows_v = []

            canny_frames = []

            for j,k in range(i, window_end - 1):
                try:
                    u_component, v_component = self.optical_flow_computer.compute_optical_flow(frames[j][1], frames[j + 1][1])
                    optical_flows_u.append(u_component)
                    optical_flows_v.append(v_component)

                    final_components = self.optical_flow_computer.compute_resize(frames[k + 1][1])
                    canny_frames.append(final_components)

                except cv2.error as e:
                    print(f"[OF] Error processing frame {frames[j][0]} from video {video_folder}. Error: {e}")
                    print(f"[RAW] Error processing frame {frames[k][0]} from video {video_folder}. Error: {e}")
                    continue
            # Commented out to test preprocess
            optical_flows_u_array = np.stack(optical_flows_u, axis = 0)
            optical_flows_v_array = np.stack(optical_flows_v, axis = 0)
            
            combined_optical_flow = np.stack([optical_flows_u_array, optical_flows_v_array], axis=-1)
            components_stacked = np.stack(canny_frames, axis=0)
            
            window_name = f"{video_folder}_{timestamp}"

            #self.numpy_writer.write_array(combined_optical_flow, window_name)
            #self.numpy_writer.write_array(components_stacked, window_name)


            timestamp = OpticalFlowProcessor.increment_timestamp(timestamp)
            #print(f"Incremented timestamp for {video_folder}: {timestamp}")
            next_increment_seconds = OpticalFlowProcessor.total_seconds_from_timestamp(timestamp)

            if last_frame_seconds - next_increment_seconds < 1.0:
                break

            i += (num_frames_in_window - overlap_frames)


    def run(self):
        dir_handler = DatasetDirectoryHandler(self.frame_loader.dataset_folder)
        
        for subject_folder in dir_handler.get_subject_folders():
            for activity_folder in dir_handler.get_activity_folders(subject_folder):
                for trial_folder in dir_handler.get_trial_folders(subject_folder, activity_folder):
                    for camera_folder in dir_handler.get_camera_folders(subject_folder, activity_folder, trial_folder):
                        print(f"Processing video: {camera_folder}")
                        self.process_video(os.path.join(subject_folder, activity_folder, trial_folder, camera_folder))









class SequentialCNN(nn.Module):
    def __init__(self):
        super(SequentialCNN, self).__init__()
        self.spatial =  SpatialCNN()
        self.temporal = TemporalCNN() 

    def forward(self, components_stacked, combined_optical_flow):
        temporal_features = self.temporal(combined_optical_flow)

        x = torch.concat(components_stacked, temporal_features)

        output = self.spatial(x)

        return output
    




def compute_metrics(true_labels, predictions):
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    specificity = tn / (tn + fp)
    f1 = f1_score(true_labels, predictions)
    return accuracy, precision, recall, specificity, f1

def plot_metrics(accuracies, precisions, recalls, specificities, f1_scores, val_losses):
    epochs_range = range(1, len(accuracies) + 1)
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(epochs_range, accuracies, 'o-', label='Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])

    plt.subplot(2, 3, 2)
    plt.plot(epochs_range, precisions, 'o-', label='Precision')
    plt.title('Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.ylim([0.5, 1])

    plt.subplot(2, 3, 3)
    plt.plot(epochs_range, recalls, 'o-', label='Recall')
    plt.title('Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.ylim([0.5, 1])

    plt.subplot(2, 3, 4)
    plt.plot(epochs_range, specificities, 'o-', label='Specificity')
    plt.title('Specificity')
    plt.xlabel('Epochs')
    plt.ylabel('Specificity')
    plt.ylim([0.5, 1])

    plt.subplot(2, 3, 5)
    plt.plot(epochs_range, f1_scores, 'o-', label='F1-Score')
    plt.title('F1-Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')
    plt.ylim([0.5, 1])
    
    plt.subplot(2, 3, 6)
    plt.plot(epochs_range, val_losses, 'o-', label='Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(true_labels, predictions, classes):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    



def train_model(dataloader_train, dataloader_val, num_epochs=50, learning_rate=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Creating Model")

    model = SequentialCNN().to(device)
    print(f"Completed model creation")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    best_loss = float('inf')
    epochs_no_improve = 0
    
    accuracies = []
    precisions = []
    recalls = []
    specificities = []
    f1_scores = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        start =timer()
        model.train()
        for batch_features, batch_labels in dataloader_train:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        epoch_val_losses = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels in dataloader_val:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                epoch_val_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        accuracy, precision, recall, specificity, f1 = compute_metrics(all_labels, all_preds)
        
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        specificities.append(specificity)
        f1_scores.append(f1)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}, F1-Score: {f1:.4f}")
        
        logging_output(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}, F1-Score: {f1:.4f}",log_path)
        print("Time Taken: ", timer() -start)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Process Completed at : ", current_time)

        plot_metrics(accuracies, precisions, recalls, specificities, f1_scores, val_losses)
        
        return model

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    accuracy, precision, recall, specificity, f1 = compute_metrics(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy, precision, recall, specificity, f1, all_labels, all_preds

# Function to print to console and write to a file
def logging_output(message, file_path='./def_log.txt'):
    try:
    # Write to file
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        with open(file_path, 'a') as file:
            file.write(f'{current_time} : {message} \n')
    except Exception as e:
        print(f"Error logging: {e}")    


if __name__ == "__main__":
    try:
        #Optical Flow
        dataset_folder = '../dataset/UP-Fall'
        output_folder  = 'preprocessed_dataset/nparray_uv'
    
        processor = OpticalFlowProcessor(dataset_folder, output_folder)
        processor.run()
        
        script_path = os.path.abspath(__file__)
         # Extract the file name from the path
        script_name = os.path.basename(script_path)
        name_only=os.path.splitext(script_name)
        model_folder=name_only[0]

        features_path = f'C:\[LINK]\{model_folder}'
        batch_size=32
        num_epochs=50
        learning_rate=0.0001

        str_model_type=f'{model_folder}_b{batch_size}e{num_epochs}L{learning_rate}'

        if not os.path.exists(f'./Results/{model_folder}'):
            try:
                # Create the directory and its parents if they don't exist
                os.makedirs(f'./Results/{model_folder}')
            except OSError as e:
                print(f"Error creating directory ./Results/{model_folder}: {e}")

        log_path=f'./Results/{model_folder}/Trainlog.log'
        print(f"Path identified: {features_path}")
        logging_output(f"Path identified is {features_path} ",log_path)
        print(f"Processing: {str_model_type}")
        logging_output(f"Processing: {str_model_type}",log_path)

        code_start=timer()

        dataset = OpticalFlow3DDataset(features_path)
        train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42, stratify=dataset.labels)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=42, stratify=np.array(dataset.labels)[train_idx])

        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)

        dataloader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
        dataloader_val = DataLoader(val_dataset, batch_size=32, shuffle=False)
        dataloader_test = DataLoader(test_dataset, batch_size=32, shuffle=False)

        if torch.cuda.is_available() :
            print(f"Running on GPU")
            logging_output("Running on GPU",log_path)
        else:
            print(f"CPU Only")
            logging_output("Running on CPU Only",log_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        print(f"Model is {str_model_type}")
        logging_output(f"Model is {str_model_type}",log_path)
        model = SequentialCNN().to(device)
        #model.load_state_dict(torch.load('fall_detection_model_3d_new.pth'))
        #model.eval()
        model = train_model(dataloader_train, dataloader_val)
    
        criterion = nn.CrossEntropyLoss().to(device)
        test_loss, test_accuracy, test_precision, test_recall, test_specificity, test_f1, true_labels, predictions = evaluate_model(model, dataloader_test, criterion, device)
    
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test Specificity: {test_specificity:.4f}, Test F1-Score: {test_f1:.4f}")
        logging_output(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test Specificity: {test_specificity:.4f}, Test F1-Score: {test_f1:.4f}",log_path)

        print(f"Total Time taken: {timer() -code_start } seconds")
        logging_output(f"Total Time taken: { timer() -code_start } Seconds", log_path)

        plot_confusion_matrix(true_labels, predictions, classes = ["No Fall", "Fall"])
    
        torch.save(model.state_dict(), 'fall_detection_model_3d_new.pth')
        
        
    except Exception as E :
        err_msg=f"Error occured {E}"
        print(err_msg)
        error_stack = traceback.format_exc()
        print(f"Error stack:\n{error_stack}")
        logging_output("******************************",log_path)
        logging_output(err_msg,log_path)
        logging_output(error_stack,log_path)
        logging_output("******************************",log_path)

