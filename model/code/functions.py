import os

import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F

import numpy as np
import pandas as pd
from sklearn import metrics

from .encoding import encoding, encoding_single

import warnings
warnings.filterwarnings("ignore")

# 模型架构 AA + LSTM + MLP
class LSTM_Double_AA(nn.Module):
    def __init__(self, feature_num, hidden_num, rnn_layer, dropout):
        super(LSTM_Double_AA, self).__init__()
        # 参数存储
        self.feature_num = feature_num
        self.hidden_num = hidden_num
        self.dropout = dropout
        self.rnn_layer = rnn_layer
        # LSTM层
        self.lstm_cdr3 = nn.LSTM(feature_num, hidden_num, num_layers=rnn_layer, dropout=dropout)
        self.lstm_pep = nn.LSTM(feature_num, hidden_num, num_layers=rnn_layer, dropout=dropout)
        # MLP
        self.hidden_layer = nn.Linear(hidden_num*2, hidden_num)
        self.relu = torch.nn.LeakyReLU()
        self.output_layer = nn.Linear(hidden_num, 1)
        self.dropout = nn.Dropout(p=dropout)
    
    
    def forward(self, cdr3_input, pep_input):
        # cdr3 encoding
        cdr3_embedding = cdr3_input.float()
        cdr3_embedding = cdr3_embedding.permute(1, 0, 2)
        cdr3_lstm, _ = self.lstm_cdr3(cdr3_embedding)
        cdr3_last_cell = cdr3_lstm[-1]  # 取最后一个时间步的隐状态作为cdr3的编码
        
        # pep encoding
        pep_embedding = pep_input.float()
        pep_embedding = pep_embedding.permute(1, 0, 2)
        pep_lstm, _ = self.lstm_pep(pep_embedding)
        pep_last_cell = pep_lstm[-1]  # shape: (batch, hidden_num)
        
        # MLP
        cdr3_pep_cat = torch.cat([cdr3_last_cell, pep_last_cell], dim=1)
        hidden_output = self.dropout(self.relu(self.hidden_layer(cdr3_pep_cat)))
        mlp_output = self.output_layer(hidden_output)
        output = torch.sigmoid(mlp_output)
        return output
    
# 模型架构 BLOSUM + CNN + MLP
class CNN_Double_BLOSUM(nn.Module):
    def __init__(self, feature_num, filter_num=16, hidden_num=32, dropout=0.1):
        super(CNN_Double_BLOSUM, self).__init__()
        # 参数存储
        self.feature_num = feature_num
        self.filter_num = filter_num
        self.dropout = dropout
        
        # CNN层
        # cdr3
        self.cnn_cdr3_block_1 = nn.Sequential(nn.Conv1d(in_channels=feature_num, out_channels=filter_num, 
                                                        kernel_size=1, padding='same'), nn.ReLU(),
                                              nn.AdaptiveMaxPool1d(output_size=1))
        self.cnn_cdr3_block_3 = nn.Sequential(nn.Conv1d(in_channels=feature_num, out_channels=filter_num, 
                                                        kernel_size=3, padding='same'), nn.ReLU(),
                                              nn.AdaptiveMaxPool1d(output_size=1))
        self.cnn_cdr3_block_5 = nn.Sequential(nn.Conv1d(in_channels=feature_num, out_channels=filter_num, 
                                                        kernel_size=5, padding='same'), nn.ReLU(),
                                              nn.AdaptiveMaxPool1d(output_size=1))
        self.cnn_cdr3_block_7 = nn.Sequential(nn.Conv1d(in_channels=feature_num, out_channels=filter_num, 
                                                        kernel_size=7, padding='same'), nn.ReLU(),
                                              nn.AdaptiveMaxPool1d(output_size=1))
        self.cnn_cdr3_block_9 = nn.Sequential(nn.Conv1d(in_channels=feature_num, out_channels=filter_num, 
                                                        kernel_size=9, padding='same'), nn.ReLU(),
                                              nn.AdaptiveMaxPool1d(output_size=1))
        
        # pep
        self.cnn_pep_block_1 = nn.Sequential(nn.Conv1d(in_channels=feature_num, out_channels=filter_num, 
                                                       kernel_size=1, padding='same'), nn.ReLU(),
                                             nn.AdaptiveMaxPool1d(output_size=1))
        self.cnn_pep_block_3 = nn.Sequential(nn.Conv1d(in_channels=feature_num, out_channels=filter_num, 
                                                       kernel_size=3, padding='same'), nn.ReLU(),
                                             nn.AdaptiveMaxPool1d(output_size=1))
        self.cnn_pep_block_5 = nn.Sequential(nn.Conv1d(in_channels=feature_num, out_channels=filter_num, 
                                                       kernel_size=5, padding='same'), nn.ReLU(),
                                             nn.AdaptiveMaxPool1d(output_size=1))
        self.cnn_pep_block_7 = nn.Sequential(nn.Conv1d(in_channels=feature_num, out_channels=filter_num, 
                                                       kernel_size=7, padding='same'), nn.ReLU(),
                                             nn.AdaptiveMaxPool1d(output_size=1))
        self.cnn_pep_block_9 = nn.Sequential(nn.Conv1d(in_channels=feature_num, out_channels=filter_num, 
                                                       kernel_size=9, padding='same'), nn.ReLU(),
                                             nn.AdaptiveMaxPool1d(output_size=1))
        
        # MLP层
        self.dense_block = nn.Sequential(nn.Linear(in_features=filter_num*5*2, out_features=hidden_num), nn.ReLU(),
                                         nn.Dropout(p=dropout), nn.Linear(hidden_num, 1), nn.Sigmoid())
        
    def forward(self, cdr3_input, pep_input):
        # cdr3 encoding
        cdr3_embedding = cdr3_input.float()
        cdr3_embedding = cdr3_embedding.permute(0, 2, 1)
        cnn_cdr3_pool_1 = self.cnn_cdr3_block_1(cdr3_embedding).squeeze(-1)
        cnn_cdr3_pool_3 = self.cnn_cdr3_block_3(cdr3_embedding).squeeze(-1)
        cnn_cdr3_pool_5 = self.cnn_cdr3_block_5(cdr3_embedding).squeeze(-1)
        cnn_cdr3_pool_7 = self.cnn_cdr3_block_7(cdr3_embedding).squeeze(-1)
        cnn_cdr3_pool_9 = self.cnn_cdr3_block_9(cdr3_embedding).squeeze(-1)
        cnn_cdr3_cat = torch.cat([cnn_cdr3_pool_1, cnn_cdr3_pool_3, cnn_cdr3_pool_5,
                                  cnn_cdr3_pool_7, cnn_cdr3_pool_9], dim=1)
        
        # pep encoding
        pep_embedding = pep_input.float()
        pep_embedding = pep_embedding.permute(0, 2, 1)
        cnn_pep_pool_1 = self.cnn_pep_block_1(pep_embedding).squeeze(-1)
        cnn_pep_pool_3 = self.cnn_pep_block_3(pep_embedding).squeeze(-1)
        cnn_pep_pool_5 = self.cnn_pep_block_5(pep_embedding).squeeze(-1)
        cnn_pep_pool_7 = self.cnn_pep_block_7(pep_embedding).squeeze(-1)
        cnn_pep_pool_9 = self.cnn_pep_block_9(pep_embedding).squeeze(-1)
        cnn_pep_cat = torch.cat([cnn_pep_pool_1, cnn_pep_pool_3, cnn_pep_pool_5,
                                  cnn_pep_pool_7, cnn_pep_pool_9], dim=1)
        
        # MLP
        cdr3_pep_cat = torch.cat([cnn_cdr3_cat, cnn_pep_cat], dim=1)
        out = self.dense_block(cdr3_pep_cat)
        return out

# 定义模型
# Ensemble Model
class Ensemble(nn.Module):
    def __init__(self, Model1, Model2, model1_paras: dict, model2_paras: dict, e_device, model1_weights_path=None, model2_weights_path=None):
        super(Ensemble, self).__init__()
        # 存放 路径
        self.model1_weights_path = model1_weights_path
        self.model2_weights_path = model2_weights_path
        # 存放 模型
        self.model1_list = []
        self.model2_list = []
        # 存放 模型的个数
        self.n_model1 = 0
        self.n_model2 = 0
        
        if not model1_weights_path is None:
            for _, _, weights_list in os.walk(model1_weights_path):
                self.n_model1 = len(weights_list)
                # print(f'\n找到{self.n_model1}个Model1')
                self.model1_weights_list = weights_list
                break
            # 创建模型
            for i in range(len(self.model1_weights_list)):
                weight_path = os.path.join(model1_weights_path, self.model1_weights_list[i])
                model1 = Model1(**model1_paras)
                model1 = model1.to(e_device)
                # 加载模型参数, map_location用于确保加载参数成功
                model1.load_state_dict(torch.load(weight_path, map_location=e_device))
                self.model1_list.append(model1)
        else:
            print("Please provide the weight path for Model 1!!!")
            pass
        
        if not model2_weights_path is None:
            for _, _, weights_list in os.walk(model2_weights_path):
                self.n_model2 = len(weights_list)
                # print(f'找到{self.n_model2}个Model2')
                self.model2_weights_list = weights_list
                break
            # 创建模型
            for i in range(len(self.model2_weights_list)):
                weight_path = os.path.join(model2_weights_path, self.model2_weights_list[i])
                model2 = Model2(**model2_paras)
                model2 = model2.to(e_device)
                model2.load_state_dict(torch.load(weight_path, map_location=e_device))
                self.model2_list.append(model2)
        else:
            print("Please provide the weight path for Model 2!!!")
            pass
        
        print('Pep2TCR has been initialized successfully!')
        
    def change_status(self, training_flag):
        if training_flag is True:
            self.model1_list = [model1.train() for model1 in self.model1_list]
            self.model2_list = [model2.train() for model2 in self.model2_list]
        else:  # training_flag is False
            self.model1_list = [model1.eval() for model1 in self.model1_list]
            self.model2_list = [model2.eval() for model2 in self.model2_list]

    # 考虑到不同的编码方式，需要单独编码
    def forward(self, model1_dat: list, model2_dat: list):
        self.model1_tensor_out = self.model1_list[0](*model1_dat)
        self.model2_tensor_out = self.model2_list[0](*model2_dat)
        
        for i in range(1, self.n_model1):
            model1 = self.model1_list[i]
            self.model1_tensor_out = torch.concat([self.model1_tensor_out, model1(*model1_dat)], dim=1)
        
        for i in range(1, self.n_model2):
            model2 = self.model2_list[i]
            self.model2_tensor_out = torch.concat([self.model2_tensor_out, model2(*model2_dat)], dim=1)
        
        # 合并
        self.model_tensor_out = torch.concat([self.model1_tensor_out, self.model2_tensor_out], dim=1)
        
        return self.model_tensor_out

# Average Ensemble
# 不用训练
# w1代表Model1代表的权重
class Avg_Ensemble(nn.Module):
    def __init__(self, Model1, Model2, model1_paras: dict, model2_paras: dict, e_device, w1: float = 0.5,
                 model1_weights_path=None, model2_weights_path=None):
        super(Avg_Ensemble, self).__init__()
        self.w1 = w1
        self.w2 = 1.0 - w1
        self.ensemble = Ensemble(Model1, Model2, model1_paras, model2_paras, e_device, model1_weights_path, model2_weights_path)
    
    def forward(self, model1_dat: list, model2_dat: list):
        out = self.ensemble(model1_dat, model2_dat)
        model1_out, model2_out = torch.split(out, [self.ensemble.n_model1, self.ensemble.n_model2], dim=1)
        model1_out = torch.mean(model1_out, dim=1) * self.w1
        model2_out = torch.mean(model2_out, dim=1) * self.w2
        # 几何平均值
        out = model1_out + model2_out
        return out

# 模型验证函数
def ensemble_model_validation(model, loss, test_data, device, has_label=True):
    # 评估
    model.eval()
    # 改变ensemble的状态为eval()
    model.ensemble.change_status(training_flag=False)
    pred = []
    y_true = []
    avg_eval_loss = []
    eval_loss = 0.0
    for dat in test_data:
        dat = [e.to(device) for e in dat]
        pred_hat = model(dat[0:2], dat[2:4])
        if has_label:
            y = dat[-1]
            cost = loss(pred_hat.squeeze(), y)
            eval_loss += cost.item() * y.size(0)
            avg_eval_loss.append(cost.item())
            y_true.extend(list(y.cpu().detach().numpy()))
        pred.extend(list(pred_hat.squeeze().cpu().detach().numpy()))
    
    roauc = 0
    prauc = 0
    y_true_array = np.array(0)
    pred_array = np.array(pred)
    if has_label:
        print("The total loss of the dataset is %f" % (eval_loss))
        y_true_array = np.array(y_true)
        fpr, tpr, _ = metrics.roc_curve(y_true_array, pred_array)
        precision, recall, _ = metrics.precision_recall_curve(y_true_array, pred_array)
        roauc = metrics.auc(fpr, tpr)
        prauc = metrics.auc(recall, precision)
        print("The ROAUC of the dataset is %.5f, PRAUC is %.5f" % (roauc, prauc))
    return roauc, prauc, y_true_array, pred_array


# 只提供文件和模型权重路径，即可预测，后续还是调用model_validation
# data_path是csv格式的文件
# 编码数据集
def ensemble_data_encoding(data_path, style_path, styles: list, 
                           num_workers=5, batch_size=128, cdr3_len_max=20, pep_len_max=20, pad_char='X', sample_flag=False, has_label=True):
    val_dat = pd.read_csv(data_path)
    if sample_flag is True:
        val_dat = val_dat.sample(frac=1, axis=0, ignore_index=True)
    # 编码
    style1_label = torch.tensor(0)  # 预设
    style2_label = torch.tensor(0)
    if has_label:
        style1_cdr3, style1_pep, style1_label = encoding(val_dat, style_path,  # type: ignore
                                                         cdr3_len_max=cdr3_len_max, pep_len_max=pep_len_max, 
                                                         pad_char=pad_char, style=styles[0], has_label=has_label)  # type: ignore
        style2_cdr3, style2_pep, style2_label = encoding(val_dat, style_path,  # type: ignore
                                                         cdr3_len_max=cdr3_len_max, pep_len_max=pep_len_max, 
                                                         pad_char=pad_char, style=styles[1], has_label=has_label)  # type: ignore
    else:
        style1_cdr3, style1_pep = encoding(val_dat, style_path, # type: ignore
                                           cdr3_len_max=cdr3_len_max, pep_len_max=pep_len_max, 
                                           pad_char=pad_char, style=styles[0], has_label=has_label)  # type: ignore
        style2_cdr3, style2_pep = encoding(val_dat, style_path, # type: ignore
                                           cdr3_len_max=cdr3_len_max, pep_len_max=pep_len_max, 
                                           pad_char=pad_char, style=styles[1], has_label=has_label)  # type: ignore
    # 转成tensor格式
    style1_cdr3 = torch.from_numpy(style1_cdr3)
    style1_pep = torch.from_numpy(style1_pep)
    if has_label:
        style1_label = torch.from_numpy(style1_label).float()
    style2_cdr3 = torch.from_numpy(style2_cdr3)
    style2_pep = torch.from_numpy(style2_pep)
    if has_label:
        style2_label = torch.from_numpy(style2_label).float()

    if has_label:
        val_tmp = data.TensorDataset(style1_cdr3, style1_pep, style2_cdr3, style2_pep, style2_label) # type: ignore
    else:
        val_tmp = data.TensorDataset(style1_cdr3, style1_pep, style2_cdr3, style2_pep)

    val_data = data.DataLoader(dataset=val_tmp, batch_size=batch_size, shuffle=False, 
                                drop_last=False, num_workers=num_workers, pin_memory=True)
    return val_data


def ensemble_model_validation_simple(data_path, style_path, Model, batch_size, model_weight_path, device, has_label=True, num_workers=5,
                                     cdr3_len_max=20, pep_len_max=20, pad_char='X', styles=['AAindex_11', 'BLOSUM50'], **model_paras):
    # 获取编码数据集
    val_data = ensemble_data_encoding(data_path, style_path, styles,
                                      num_workers, batch_size,
                                      cdr3_len_max, pep_len_max, pad_char, has_label=has_label)

    # 建立模型并加载参数
    model = Model(**model_paras)
    model = model.to(device)
    loss = nn.BCELoss().to(device)
    # 加载模型权重，map_location用于确保模型参数加载成功
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    # 模型测试
    roauc, prauc, y_true_array, pred_array = ensemble_model_validation(model, loss, val_data, device, has_label)
    print("Successful Prediction!")
    return roauc, prauc, y_true_array, pred_array

# single prediction
def sin_pre(cdr3, pep, style_path, Model, model_weight_path, device, 
            cdr3_len_max=20, pep_len_max=20, pad_char='X', styles=['AAindex_11', 'BLOSUM50'], **model_paras):

    style1_cdr3, style1_pep = encoding_single(cdr3, pep, style_path, cdr3_len_max, pep_len_max, 
                                              pad_char, style=styles[0])
    style2_cdr3, style2_pep = encoding_single(cdr3, pep, style_path, cdr3_len_max, pep_len_max, 
                                              pad_char, style=styles[1])
    
    style1_cdr3 = torch.from_numpy(style1_cdr3).to(device).unsqueeze(0)
    style1_pep = torch.from_numpy(style1_pep).to(device).unsqueeze(0)
    style2_cdr3 = torch.from_numpy(style2_cdr3).to(device).unsqueeze(0)
    style2_pep = torch.from_numpy(style2_pep).to(device).unsqueeze(0)
    
    # 建立模型并加载参数
    model = Model(**model_paras)
    model = model.to(device)
    # 加载模型权重，map_location用于确保模型参数加载成功
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    
    # 模型测试
    model.eval()
    model.ensemble.change_status(training_flag=False)
    
    pred_hat = model([style1_cdr3, style1_pep], [style2_cdr3, style2_pep])

    print("Prediction score is {:.4f}".format(pred_hat.item()))
    
    return round(pred_hat.item(), 4)
