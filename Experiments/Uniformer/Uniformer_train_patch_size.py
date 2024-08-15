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

from functools import partial
from timm.models.layers import trunc_normal_, DropPath, to_2tuple

########################################################

def conv_3xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (2, stride, stride), (1, 0, 0), groups=groups)
    
def conv_1xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (1, kernel_size, kernel_size), (1, stride, stride), (0, 0, 0), groups=groups)

def conv_3xnxn_std(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (1, stride, stride), (1, 0, 0), groups=groups)

def conv_1x1x1(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=groups)

def conv_3x3x3(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (3, 3, 3), (1, 1, 1), (1, 1, 1), groups=groups)

def conv_5x5x5(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (5, 5, 5), (1, 1, 1), (2, 2, 2), groups=groups)

def bn_3d(dim):
    return nn.BatchNorm3d(dim)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = conv_1x1x1(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = conv_1x1x1(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    
class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = bn_3d(dim)
        self.conv1 = conv_1x1x1(dim, dim, 1)
        self.conv2 = conv_1x1x1(dim, dim, 1)
        self.attn = conv_5x5x5(dim, dim, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = bn_3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x   


class SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, C, T, H, W)
        return x    


class SplitSABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.t_norm = norm_layer(dim)
        self.t_attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, T, H, W = x.shape
        attn = x.view(B, C, T, H * W).permute(0, 3, 2, 1).contiguous()
        attn = attn.view(B * H * W, T, C)
        attn = attn + self.drop_path(self.t_attn(self.t_norm(attn)))
        attn = attn.view(B, H * W, T, C).permute(0, 2, 1, 3).contiguous()
        attn = attn.view(B * T, H * W, C)
        residual = x.view(B, C, T, H * W).permute(0, 2, 3, 1).contiguous()
        residual = residual.view(B * T, H * W, C)
        attn = residual + self.drop_path(self.attn(self.norm1(attn)))
        attn = attn.view(B, T * H * W, C)
        out = attn + self.drop_path(self.mlp(self.norm2(attn)))
        out = out.transpose(1, 2).reshape(B, C, T, H, W)
        return out


class SpeicalPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(51,38), patch_size=16, in_chans=2, embed_dim=768):
        super().__init__()
        img_size = (img_size[0], img_size[1])
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = conv_3xnxn(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])

    def forward(self, x):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x
    

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(51,38), patch_size=16, in_chans=2, embed_dim=768, std=False):
        super().__init__()
        #img_size = to_2tuple(img_size)
        img_size = (img_size[0], img_size[1])
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        if std:
            self.proj = conv_3xnxn_std(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])
        else:
            self.proj = conv_1xnxn(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])

    def forward(self, x):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x


class Uniformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, depth=[5, 8, 20, 7], num_classes=2, img_size=(51,38), in_chans=2, embed_dim=[64, 128, 320, 512],
                 head_dim=64, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0.3, attn_drop_rate=0., drop_path_rate=0., norm_layer=None, split=False, std=False):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = partial(nn.LayerNorm, eps=1e-6) 
        
        self.patch_embed1 = SpeicalPatchEmbed(img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(img_size=(img_size[0] // 4, img_size[1] // 4), patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1], std=std)
        self.patch_embed3 = PatchEmbed(img_size=(img_size[0] // 8, img_size[1] // 8), patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2], std=std)
        self.patch_embed4 = PatchEmbed(img_size=(img_size[0] // 16, img_size[1] // 16), patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3], std=std)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dim]
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]], norm_layer=norm_layer)
            for i in range(depth[1])])
        if split:
            self.blocks3 = nn.ModuleList([
                SplitSABlock(
                    dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer)
                for i in range(depth[2])])
            self.blocks4 = nn.ModuleList([
                SplitSABlock(
                    dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer)
            for i in range(depth[3])])
        else:
            self.blocks3 = nn.ModuleList([
                SABlock(
                    dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer)
                for i in range(depth[2])])
            self.blocks4 = nn.ModuleList([
                SABlock(
                    dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer)
            for i in range(depth[3])])
        self.norm = bn_3d(embed_dim[-1])
        
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
        
        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)

        for name, p in self.named_parameters():
            if 't_attn.qkv.weight' in name:
                nn.init.constant_(p, 0)
            if 't_attn.qkv.bias' in name:
                nn.init.constant_(p, 0)
            if 't_attn.proj.weight' in name:
                nn.init.constant_(p, 1)
            if 't_attn.proj.bias' in name:
                nn.init.constant_(p, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head.weight.data.uniform_()

    def forward_features(self, x):
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
        x = self.patch_embed4(x)
        for blk in self.blocks4:
            x = blk(x)
        x = self.norm(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.flatten(2).mean(-1)
        x = self.head(x)
        return x
        

def uniformer_small():
    return Uniformer(
        depth=[3, 4, 8, 3], embed_dim=[64, 128, 320, 512], 
        head_dim=64, drop_rate=0.1)

def uniformer_small600():
    return Uniformer(
        depth=[3, 4, 8, 3], embed_dim=[64, 128, 320, 512], 
        head_dim=64, drop_rate=0.1, num_classes=600)

def uniformer_base():
    return Uniformer(
        depth=[5, 8, 20, 7], embed_dim=[64, 128, 320, 512], 
        head_dim=64, drop_rate=0.3)

def uniformer_base600():
    return Uniformer(
        depth=[5, 8, 20, 7], embed_dim=[64, 128, 320, 512], 
        head_dim=64, drop_rate=0.3, num_classes=600)


#########################################################

       
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
    model = Uniformer().to(device)
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
    
    model = Uniformer().to(device)
    
    saved_model_path = 'Uniformer_PS_model_3d.pth'
    
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
    
    torch.save(model.state_dict(), 'Uniformer_PS_model_3d_1to1.pth')
    
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