import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from dataset_prep_3d import OpticalFlow3DDataset
import cv2
import seaborn as sns
import os

class CNN(nn.Module): 
    def __init__(self):
        super(CNN, self).__init__()

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
       
    def forward(self, t):
        t = F.gelu(self.bn1(self.conv1(t)))
        t = F.max_pool3d(t, 2)
        t = F.gelu(self.bn2(self.conv2(t)))
        t = F.max_pool3d(t, 2)
        t = F.gelu(self.bn3(self.conv3(t)))
        t = F.max_pool3d(t, 2)
        t = F.gelu(self.bn4(self.conv4(t)))
        t = self.global_avg_pool(t)
        t = t.view(t.size(0), -1)
       
        return t

class LSTMModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModule, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])  # Taking the last time step's output
        output = self.fc(lstm_out)
        return output

class SequentialCNN(nn.Module):
    def __init__(self):
        super(SequentialCNN, self).__init__()
        self.cnn = CNN() 
        self.lstm_module = LSTMModule(input_size=256, hidden_size=128, num_layers=1, dropout=0.5)

    def forward(self, combined_optical_flow):
        cnn_features = self.cnn(combined_optical_flow)
        lstm_input = cnn_features.unsqueeze(1)  
        output = self.lstm_module(lstm_input)
        return output    

        
def compute_metrics(true_labels, predictions):
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    specificity = tn / (tn + fp)
    f1 = f1_score(true_labels, predictions)
    return accuracy, precision, recall, specificity, f1

def plot_metrics(accuracies, precisions, recalls, specificities, f1_scores, val_losses, train_losses):
    epochs_range = range(1, len(accuracies) + 1)
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(epochs_range, accuracies, 'o-', label='Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])

    plt.subplot(2, 3, 2)
    plt.plot(epochs_range, precisions, 'o-', label='Precision')
    plt.title('Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.ylim([0, 1])

    plt.subplot(2, 3, 3)
    plt.plot(epochs_range, recalls, 'o-', label='Recall')
    plt.title('Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.ylim([0, 1])

    plt.subplot(2, 3, 4)
    plt.plot(epochs_range, specificities, 'o-', label='Specificity')
    plt.title('Specificity')
    plt.xlabel('Epochs')
    plt.ylabel('Specificity')
    plt.ylim([0, 1])

    plt.subplot(2, 3, 5)
    plt.plot(epochs_range, f1_scores, 'o-', label='F1-Score')
    plt.title('F1-Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')
    plt.ylim([0, 1])
    
    plt.subplot(2, 3, 6)
    plt.plot(epochs_range, val_losses, 'o-', label='Validation Loss', color='red')
    plt.plot(epochs_range, train_losses, 'o-', label='Training Loss', color='blue')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def calculate_cm_percentages(cm):
    cm_percentages = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    cm_percentages = np.around(cm_percentages * 100, decimals = 2)
    return cm_percentages

def plot_confusion_matrix(true_labels, predictions, classes):
    cm = confusion_matrix(true_labels, predictions)
    cm_percentages = calculate_cm_percentages(cm)
    
    labels = (np.asarray(["{0}\n({1}%)".format(value, percentage)
                         for value, percentage in zip(cm.flatten(), cm_percentages.flatten())])
                ).reshape(cm.shape)
    
    plt.figure(figsize = (10, 7))
    sns.heatmap(cm, annot = labels, fmt = '', cmap='Blues', xticklabels = classes, yticklabels = classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    
def train_model(dataloader_train, dataloader_val, num_epochs = 50, learning_rate = 0.0001, weight_decay = 1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequentialCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    
    best_loss = float('inf')
    epochs_no_improve = 0
    
    accuracies = []
    precisions = []
    recalls = []
    specificities = []
    f1_scores = []
    val_losses = []
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_losses = []
        for batch_features, batch_labels in dataloader_train:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())
        
        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses.append(avg_train_loss)
        
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
        val_losses.append(avg_val_loss)
        
        accuracy, precision, recall, specificity, f1 = compute_metrics(all_labels, all_preds)
        
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        specificities.append(specificity)
        f1_scores.append(f1)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}, F1-Score: {f1:.4f}")
        
        # if avg_val_loss < best_loss:
        #     best_loss = avg_val_loss
        #     epochs_no_improve = 0
        #     torch.save(model.state_dict(), 'best_model.pth')
        # else:
        #     epochs_no_improve += 1
        #     if epochs_no_improve == 5:
        #         print("Early stopping!")    
        #         model.load_state_dict(torch.load('best_model.pth'))
        #         break
            
    plot_metrics(accuracies, precisions, recalls, specificities, f1_scores, val_losses, train_losses)
        
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

if __name__ == "__main__": 
    features_path = "C:/Users/ac22aci/Desktop/Exp_6_2_BG+OF_Baseline_80_Split/Balanced/OF"
    test_path = "C:/Users/ac22aci/Desktop/Exp_6_2_BG+OF_Baseline_80_Split/Unbalanced/OF"
    
    train_val_dataset = OpticalFlow3DDataset(features_path)
    test_dataset = OpticalFlow3DDataset(test_path)
    
    train_idx, val_idx = train_test_split(range(len(train_val_dataset)), test_size=0.25, random_state=42, stratify=train_val_dataset.labels)

    train_dataset = torch.utils.data.Subset(train_val_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(train_val_dataset, val_idx)

    dataloader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dataloader_val = DataLoader(val_dataset, batch_size=32, shuffle=False)
    dataloader_test = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SequentialCNN().to(device)
    
    saved_model_path = 'Sequential_model_3d.pth'
    
    if os.path.isfile(saved_model_path):
        model.load_state_dict(torch.load(saved_model_path, map_location=device))
        print("Loaded saved model.")
    
    else:
        print("No saved model found. Training a new model.")
        model = train_model(dataloader_train, dataloader_val)
    
    model.eval()
    
    criterion = nn.CrossEntropyLoss().to(device)
    test_loss, test_accuracy, test_precision, test_recall, test_specificity, test_f1, true_labels, predictions = evaluate_model(model, dataloader_test, criterion, device)
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test Specificity: {test_specificity:.4f}, Test F1-Score: {test_f1:.4f}")
    
    plot_confusion_matrix(true_labels, predictions, classes = ["No Fall", "Fall"])
    
    torch.save(model.state_dict(), 'Sequential_model_3d_1to1.pth')
    
    # visualize model predictions
    # misclassified_samples = visualize_misclassified_optical_flow(model, dataloader_test, device)

    # for optical_flow_rgb, true_label, predicted_label in misclassified_samples:
    #     while True:
    #         for frame in optical_flow_rgb:
    #             cv2.imshow(f"Optical Flow (True Label: {true_label}, Predicted: {predicted_label})", frame)
    #             key = cv2.waitKey(200)  # Display each frame for 100ms
    #             if key == ord('n'):
    #                 break
                
    #         if key == ord('n'):
    #             break

    #     cv2.destroyAllWindows()