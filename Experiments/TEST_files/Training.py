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

class FallDetectionCNN(nn.Module):
    def __init__(self):
        super(FallDetectionCNN, self).__init__()

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

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
        

def calculate_metrics(actual_labels, predicted_labels):
    true_neg, false_pos, false_neg, true_pos = confusion_matrix(actual_labels, predicted_labels).ravel()
    model_accuracy = accuracy_score(actual_labels, predicted_labels)
    model_precision = precision_score(actual_labels, predicted_labels)
    model_recall = recall_score(actual_labels, predicted_labels)
    model_specificity = true_neg / (true_neg + false_pos)
    f1_score_value = f1_score(actual_labels, predicted_labels)
    return model_accuracy, model_precision, model_recall, model_specificity, f1_score_value

def display_metrics(accuracy_values, precision_values, recall_values, specificity_values, f1_scores_values, validation_losses, training_losses):
    epochs = range(1, len(accuracy_values) + 1)
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(epochs, accuracy_values, 'o-', label='Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])

    plt.subplot(2, 3, 2)
    plt.plot(epochs, precision_values, 'o-', label='Precision')
    plt.title('Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.ylim([0, 1])

    plt.subplot(2, 3, 3)
    plt.plot(epochs, recall_values, 'o-', label='Recall')
    plt.title('Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.ylim([0, 1])

    plt.subplot(2, 3, 4)
    plt.plot(epochs, specificity_values, 'o-', label='Specificity')
    plt.title('Specificity')
    plt.xlabel('Epochs')
    plt.ylabel('Specificity')
    plt.ylim([0, 1])

    plt.subplot(2, 3, 5)
    plt.plot(epochs, f1_scores_values, 'o-', label='F1-Score')
    plt.title('F1-Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')
    plt.ylim([0, 1])
    
    plt.subplot(2, 3, 6)
    plt.plot(epochs, validation_losses, 'o-', label='Validation Loss', color='red')
    plt.plot(epochs, training_losses, 'o-', label='Training Loss', color='blue')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def calculate_confusion_matrix_percentages(conf_matrix):
    conf_matrix_percentages = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    conf_matrix_percentages = np.around(conf_matrix_percentages * 100, decimals=2)
    return conf_matrix_percentages

def visualize_confusion_matrix(actual_labels, predicted_labels, class_names):
    conf_matrix = confusion_matrix(actual_labels, predicted_labels)
    conf_matrix_percentages = calculate_confusion_matrix_percentages(conf_matrix)
    
    labels = (np.asarray(["{0}\n({1}%)".format(value, percentage)
                         for value, percentage in zip(conf_matrix.flatten(), conf_matrix_percentages.flatten())])
                ).reshape(conf_matrix.shape)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=labels, fmt='', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    
def execute_training(train_data_loader, validation_data_loader, total_epochs=50, learning_rate_value=0.0001, decay_rate=1e-5):
    computation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fall_detection_model = FallDetectionCNN().to(computation_device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(fall_detection_model.parameters(), lr=learning_rate_value, weight_decay=decay_rate)
    
    optimal_loss = float('inf')
    no_improvement_epochs = 0
    
    accuracy_metrics = []
    precision_metrics = []
    recall_metrics = []
    specificity_metrics = []
    f1_score_metrics = []
    validation_losses = []
    training_losses = []
    
    for epoch in range(total_epochs):
        fall_detection_model.train()
        epoch_training_losses = []
        for features_batch, labels_batch in train_data_loader:
            features_batch, labels_batch = features_batch.to(computation_device), labels_batch.to(computation_device)
            optimizer.zero_grad()
            predictions = fall_detection_model(features_batch)
            loss = loss_function(predictions, labels_batch)
            loss.backward()
            optimizer.step()
            epoch_training_losses.append(loss.item())
        
        average_training_loss = sum(epoch_training_losses) / len(epoch_training_losses)
        training_losses.append(average_training_loss)
        
        fall_detection_model.eval()
        epoch_validation_losses = []
        all_predictions = []
        all_actual_labels = []
        
        with torch.no_grad():
            for features_batch, labels_batch in validation_data_loader:
                features_batch, labels_batch = features_batch.to(computation_device), labels_batch.to(computation_device)
                
                predictions = fall_detection_model(features_batch)
                loss = loss_function(predictions, labels_batch)
                epoch_validation_losses.append(loss.item())
                _, predicted_labels = torch.max(predictions.data, 1)
                all_predictions.extend(predicted_labels.cpu().numpy())
                all_actual_labels.extend(labels_batch.cpu().numpy())
        
        average_validation_loss = sum(epoch_validation_losses) / len(epoch_validation_losses)
        validation_losses.append(average_validation_loss)
        
        model_accuracy, model_precision, model_recall, model_specificity, model_f1_score = calculate_metrics(all_actual_labels, all_predictions)
        
        accuracy_metrics.append(model_accuracy)
        precision_metrics.append(model_precision)
        recall_metrics.append(model_recall)
        specificity_metrics.append(model_specificity)
        f1_score_metrics.append(model_f1_score)
        
        print(f"Epoch [{epoch+1}/{total_epochs}], Loss: {loss.item():.4f}, Accuracy: {model_accuracy:.4f}, Precision: {model_precision:.4f}, Recall: {model_recall:.4f}, Specificity: {model_specificity:.4f}, F1-Score: {model_f1_score:.4f}")
        
        # if average_validation_loss < optimal_loss:
        #     optimal_loss = average_validation_loss
        #     no_improvement_epochs = 0
        #     torch.save(fall_detection_model.state_dict(), 'best_model.pth')
        # else:
        #     no_improvement_epochs += 1
        #     if no_improvement_epochs == 5:
        #         print("Early stopping!")    
        #         fall_detection_model.load_state_dict(torch.load('best_model.pth'))
        #         break
            
    display_metrics(accuracy_metrics, precision_metrics, recall_metrics, specificity_metrics, f1_score_metrics, validation_losses, training_losses)
        
    return fall_detection_model

def assess_model(fall_detection_model, data_loader, loss_function, computation_device):
    fall_detection_model.eval()
    all_predictions = []
    all_actual_labels = []
    cumulative_loss = 0.0
    with torch.no_grad():
        for features_batch, labels_batch in data_loader:
            features_batch, labels_batch = features_batch.to(computation_device), labels_batch.to(computation_device)
            
            predictions = fall_detection_model(features_batch)
            loss = loss_function(predictions, labels_batch)
            cumulative_loss += loss.item()
            _, predicted_labels = torch.max(predictions.data, 1)
            all_predictions.extend(predicted_labels.cpu().numpy())
            all_actual_labels.extend(labels_batch.cpu().numpy())
    
    model_accuracy, model_precision, model_recall, model_specificity, model_f1_score = calculate_metrics(all_actual_labels, all_predictions)
    average_loss = cumulative_loss / len(data_loader)
    
    return average_loss, model_accuracy, model_precision, model_recall, model_specificity, model_f1_score, all_actual_labels, all_predictions

if __name__ == "__main__": 
    dataset_train_path = "C:/Users/ac22aci/Desktop/Exp_6_2_BG+OF_Baseline_80_Split/Balanced/OF"
    dataset_test_path = "C:/Users/ac22aci/Desktop/Exp_6_2_BG+OF_Baseline_80_Split/Unbalanced/OF"
    
    dataset_train_val = OpticalFlow3DDataset(dataset_train_path)
    dataset_test = OpticalFlow3DDataset(dataset_test_path)
    
    train_indices, validation_indices = train_test_split(range(len(dataset_train_val)), test_size=0.25, random_state=42, stratify=dataset_train_val.labels)

    training_dataset = torch.utils.data.Subset(dataset_train_val, train_indices)
    validation_dataset = torch.utils.data.Subset(dataset_train_val, validation_indices)

    data_loader_train = DataLoader(training_dataset, batch_size=32, shuffle=True)
    data_loader_validation = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    data_loader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)

    computation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    fall_detection_model = FallDetectionCNN().to(computation_device)
    
    saved_model_filename = 'Relu_model_3d.pth'
    
    if os.path.isfile(saved_model_filename):
        fall_detection_model.load_state_dict(torch.load(saved_model_filename, map_location=computation_device))
        print("Loaded saved model.")
    
    else:
        print("No saved model found. Training a new model.")
        fall_detection_model = execute_training(data_loader_train, data_loader_validation)
    
    fall_detection_model.eval()
    
    loss_function = nn.CrossEntropyLoss().to(computation_device)
    test_loss_value, test_accuracy_value, test_precision_value, test_recall_value, test_specificity_value, test_f1_score_value, actual_labels, predicted_labels = assess_model(fall_detection_model, data_loader_test, loss_function, computation_device)
    
    print(f"Test Loss: {test_loss_value:.4f}, Test Accuracy: {test_accuracy_value:.4f}, Test Precision: {test_precision_value:.4f}, Test Recall: {test_recall_value:.4f}, Test Specificity: {test_specificity_value:.4f}, Test F1-Score: {test_f1_score_value:.4f}")
    
    visualize_confusion_matrix(actual_labels, predicted_labels, classes=["No Fall", "Fall"])
    
    torch.save(fall_detection_model.state_dict(), 'Relu_model_3d_1to1.pth')
