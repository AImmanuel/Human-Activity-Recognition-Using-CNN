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
from dataset_prep_2d import OpticalFlow2DDataset
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

        #### Fully connected layers
        #self.fc1 = nn.Linear(256, 128)
        #self.dropout1 = nn.Dropout(0.5)
        #self.fc2 = nn.Linear(128, 64)
        #self.dropout2 = nn.Dropout(0.5)
        #self.fc3 = nn.Linear(64, 2) 

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
        ###
        #t = F.relu(self.fc1(t))
        #t = self.dropout1(t)
        #t = F.relu(self.fc2(t))
        #t = self.dropout2(t)
        #t = self.fc3(t)

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
        
        #logging_output(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}, F1-Score: {f1:.4f}",log_path)
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
#def logging_output(message, file_path='./def_log.txt'):
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
        script_path = os.path.abspath(__file__)
         # Extract the file name from the path
        script_name = os.path.basename(script_path)
        name_only = os.path.splitext(script_name)
        model_folder = name_only[0]

        features_path_OF = 'E:/MscProject/Outputs/2StreamConv_Seq/Unbalanced/OF'
        features_path_RAW = 'E:/MscProject/Outputs/2StreamConv_Seq/Unbalanced/RAW'

        batch_size = 32
        num_epochs = 50
        learning_rate = 0.0001

        str_model_type = f'{model_folder}_b{batch_size}e{num_epochs}L{learning_rate}'

        if not os.path.exists(f'./Results/{model_folder}'):
            try:
                # Create the directory and its parents if they don't exist
                os.makedirs(f'./Results/{model_folder}')
            except OSError as e:
                print(f"Error creating directory ./Results/{model_folder}: {e}")

        #log_path=f'./Results/{model_folder}/Trainlog.log'
        print(f"Path identified: {features_path_OF}")
        print(f"Path identified: {features_path_RAW}")
        #logging_output(f"Path identified is {features_path} ",log_path)

        print(f"Processing: {str_model_type}")
        #logging_output(f"Processing: {str_model_type}",log_path)

        code_start=timer()

        dataset_OF = OpticalFlow3DDataset(features_path_OF)
        dataset_RAW = OpticalFlow2DDataset(features_path_RAW)

#OF
        train_idx_OF, test_idx_OF = train_test_split(range(len(dataset_OF)), test_size=0.2, random_state=42, stratify=dataset_OF.labels)
        train_idx_OF, val_idx_OF = train_test_split(train_idx_OF, test_size=0.25, random_state=42, stratify=np.array(dataset_OF.labels)[train_idx_OF])

        train_dataset_OF = torch.utils.data.Subset(dataset_OF, train_idx_OF)
        val_dataset_OF = torch.utils.data.Subset(dataset_OF, val_idx_OF)
        test_dataset_OF = torch.utils.data.Subset(dataset_OF, test_idx_OF)

        dataloader_train_OF = DataLoader(train_dataset_OF, batch_size=32, shuffle=True)
        dataloader_val_OF = DataLoader(val_dataset_OF, batch_size=32, shuffle=False)
        dataloader_test_OF = DataLoader(test_dataset_OF, batch_size=32, shuffle=False)

#RAW
        train_idx_RAW, test_idx_RAW = train_test_split(range(len(dataset_RAW)), test_size=0.2, random_state=42, stratify=dataset_RAW.labels)
        train_idx_RAW, val_idx_RAW = train_test_split(train_idx_RAW, test_size=0.25, random_state=42, stratify=np.array(dataset_RAW.labels)[train_idx_RAW])

        train_dataset_RAW = torch.utils.data.Subset(dataset_RAW, train_idx_RAW)
        val_dataset_RAW = torch.utils.data.Subset(dataset_RAW, val_idx_RAW)
        test_dataset_RAW = torch.utils.data.Subset(dataset_RAW, test_idx_RAW)

        dataloader_train_RAW = DataLoader(train_dataset_RAW, batch_size=32, shuffle=True)
        dataloader_val_RAW = DataLoader(val_dataset_RAW, batch_size=32, shuffle=False)
        dataloader_test_RAW = DataLoader(test_dataset_RAW, batch_size=32, shuffle=False)



        if torch.cuda.is_available() :
            print(f"Running on GPU")
            #logging_output("Running on GPU",log_path)
        else:
            print(f"CPU Only")
            #logging_output("Running on CPU Only",log_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        print(f"Model is {str_model_type}")
        #logging_output(f"Model is {str_model_type}",log_path)
        model = SequentialCNN().to(device)
        #model.load_state_dict(torch.load('fall_detection_model_3d_new.pth'))
        #model.eval()
        model = train_model(dataloader_train, dataloader_val)
    
        criterion = nn.CrossEntropyLoss().to(device)
        test_loss, test_accuracy, test_precision, test_recall, test_specificity, test_f1, true_labels, predictions = evaluate_model(model, dataloader_test, criterion, device)
    
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test Specificity: {test_specificity:.4f}, Test F1-Score: {test_f1:.4f}")
        #logging_output(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test Specificity: {test_specificity:.4f}, Test F1-Score: {test_f1:.4f}",log_path)

        print(f"Total Time taken: {timer() -code_start } seconds")
        #logging_output(f"Total Time taken: { timer() -code_start } Seconds", log_path)

        plot_confusion_matrix(true_labels, predictions, classes = ["No Fall", "Fall"])
    
        torch.save(model.state_dict(), 'fall_detection_model_3d_new.pth')
        
        
    except Exception as E :
        err_msg=f"Error occured {E}"
        print(err_msg)
        error_stack = traceback.format_exc()
        print(f"Error stack:\n{error_stack}")
        
        #logging_output("******************************",log_path)
        #logging_output(err_msg,log_path)
        #logging_output(error_stack,log_path)
        #logging_output("******************************",log_path)

