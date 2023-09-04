import os

import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F

import numpy as np
import pandas as pd
from sklearn import metrics

import encoding

import warnings
warnings.filterwarnings("ignore")

class LSTM_Double_AA(nn.Module):
    def __init__(self, feature_num, hidden_num, rnn_layer, dropout):
        super(LSTM_Double_AA, self).__init__()

        self.feature_num = feature_num
        self.hidden_num = hidden_num
        self.dropout = dropout
        self.rnn_layer = rnn_layer

        self.lstm_cdr3 = nn.LSTM(feature_num, hidden_num, num_layers=rnn_layer, dropout=dropout)
        self.lstm_pep = nn.LSTM(feature_num, hidden_num, num_layers=rnn_layer, dropout=dropout)

        self.hidden_layer = nn.Linear(hidden_num*2, hidden_num)
        self.relu = torch.nn.LeakyReLU()
        self.output_layer = nn.Linear(hidden_num, 1)
        self.dropout = nn.Dropout(p=dropout)
    
    
    def forward(self, cdr3_input, pep_input):
        # cdr3 encoding
        cdr3_embedding = cdr3_input.float()
        cdr3_embedding = cdr3_embedding.permute(1, 0, 2)
        cdr3_lstm, _ = self.lstm_cdr3(cdr3_embedding)
        cdr3_last_cell = cdr3_lstm[-1]
        
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
    

class CNN_Double_BLOSUM(nn.Module):
    def __init__(self, feature_num, filter_num=16, hidden_num=32, dropout=0.1):
        super(CNN_Double_BLOSUM, self).__init__()

        self.feature_num = feature_num
        self.filter_num = filter_num
        self.dropout = dropout
        
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
        
        # MLP
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

# Ensemble Model
class Ensemble(nn.Module):
    def __init__(self, Model1, Model2, model1_paras: dict, model2_paras: dict, e_device, model1_weights_path=None, model2_weights_path=None):
        super(Ensemble, self).__init__()

        self.model1_weights_path = model1_weights_path
        self.model2_weights_path = model2_weights_path

        self.model1_list = []
        self.model2_list = []

        self.n_model1 = 0
        self.n_model2 = 0
        
        if not model1_weights_path is None:
            for _, _, weights_list in os.walk(model1_weights_path):
                self.n_model1 = len(weights_list)
                self.model1_weights_list = weights_list
                break
 
            for i in range(len(self.model1_weights_list)):
                weight_path = os.path.join(model1_weights_path, self.model1_weights_list[i])
                model1 = Model1(**model1_paras)
                model1 = model1.to(e_device)
                model1.load_state_dict(torch.load(weight_path, map_location=e_device))
                self.model1_list.append(model1)
        else:
            pass
        
        if not model2_weights_path is None:
            for _, _, weights_list in os.walk(model2_weights_path):
                self.n_model2 = len(weights_list)
                self.model2_weights_list = weights_list
                break

            for i in range(len(self.model2_weights_list)):
                weight_path = os.path.join(model2_weights_path, self.model2_weights_list[i])
                model2 = Model2(**model2_paras)
                model2 = model2.to(e_device)
                model2.load_state_dict(torch.load(weight_path, map_location=e_device))
                self.model2_list.append(model2)
        else:
            pass

        
    def change_status(self, training_flag):
        if training_flag is True:
            self.model1_list = [model1.train() for model1 in self.model1_list]
            self.model2_list = [model2.train() for model2 in self.model2_list]
        else:  # training_flag is False
            self.model1_list = [model1.eval() for model1 in self.model1_list]
            self.model2_list = [model2.eval() for model2 in self.model2_list]


    def forward(self, model1_dat: list, model2_dat: list):
        self.model1_tensor_out = self.model1_list[0](*model1_dat)
        self.model2_tensor_out = self.model2_list[0](*model2_dat)
        
        for i in range(1, self.n_model1):
            model1 = self.model1_list[i]
            self.model1_tensor_out = torch.concat([self.model1_tensor_out, model1(*model1_dat)], dim=1)
        
        for i in range(1, self.n_model2):
            model2 = self.model2_list[i]
            self.model2_tensor_out = torch.concat([self.model2_tensor_out, model2(*model2_dat)], dim=1)
        
        self.model_tensor_out = torch.concat([self.model1_tensor_out, self.model2_tensor_out], dim=1)
        
        return self.model_tensor_out

# Average Ensemble
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
        out = model1_out + model2_out
        return out


def ensemble_model_validation(model, loss, test_data, device, has_label=True):
    model.eval()
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
        y_true_array = np.array(y_true)
        fpr, tpr, _ = metrics.roc_curve(y_true_array, pred_array)
        precision, recall, _ = metrics.precision_recall_curve(y_true_array, pred_array)
        roauc = metrics.auc(fpr, tpr)
        prauc = metrics.auc(recall, precision)
    return roauc, prauc, y_true_array, pred_array


def ensemble_data_encoding(data_path, style_path, styles: list, 
                           num_workers=5, batch_size=128, cdr3_len_max=20, pep_len_max=20, pad_char='X', sample_flag=False, has_label=True):
    val_dat = pd.read_csv(data_path)
    if sample_flag is True:
        val_dat = val_dat.sample(frac=1, axis=0, ignore_index=True)

    style1_label = torch.tensor(0)
    style2_label = torch.tensor(0)
    if has_label:
        style1_cdr3, style1_pep, style1_label = encoding.encoding(val_dat, style_path,  # type: ignore
                                                                  cdr3_len_max=cdr3_len_max, pep_len_max=pep_len_max, 
                                                                  pad_char=pad_char, style=styles[0], has_label=has_label)  # type: ignore
        style2_cdr3, style2_pep, style2_label = encoding.encoding(val_dat, style_path,  # type: ignore
                                                                  cdr3_len_max=cdr3_len_max, pep_len_max=pep_len_max, 
                                                                  pad_char=pad_char, style=styles[1], has_label=has_label)  # type: ignore
    else:
        style1_cdr3, style1_pep = encoding.encoding(val_dat, style_path, # type: ignore
                                                    cdr3_len_max=cdr3_len_max, pep_len_max=pep_len_max, 
                                                    pad_char=pad_char, style=styles[0], has_label=has_label)  # type: ignore
        style2_cdr3, style2_pep = encoding.encoding(val_dat, style_path, # type: ignore
                                                    cdr3_len_max=cdr3_len_max, pep_len_max=pep_len_max, 
                                                    pad_char=pad_char, style=styles[1], has_label=has_label)  # type: ignore
 
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

    val_data = ensemble_data_encoding(data_path, style_path, styles,
                                      num_workers, batch_size,
                                      cdr3_len_max, pep_len_max, pad_char, has_label=has_label)

    model = Model(**model_paras)
    model = model.to(device)
    loss = nn.BCELoss().to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    roauc, prauc, y_true_array, pred_array = ensemble_model_validation(model, loss, val_data, device, has_label)

    return roauc, prauc, y_true_array, pred_array

# single prediction
def sin_pre(cdr3, pep, style_path, Model, model_weight_path, device, 
            cdr3_len_max=20, pep_len_max=20, pad_char='X', styles=['AAindex_11', 'BLOSUM50'], **model_paras):

    style1_cdr3, style1_pep = encoding.encoding_single(cdr3, pep, style_path, cdr3_len_max, pep_len_max, 
                                                       pad_char, style=styles[0])
    style2_cdr3, style2_pep = encoding.encoding_single(cdr3, pep, style_path, cdr3_len_max, pep_len_max, 
                                                       pad_char, style=styles[1])
    
    style1_cdr3 = torch.from_numpy(style1_cdr3).to(device).unsqueeze(0)
    style1_pep = torch.from_numpy(style1_pep).to(device).unsqueeze(0)
    style2_cdr3 = torch.from_numpy(style2_cdr3).to(device).unsqueeze(0)
    style2_pep = torch.from_numpy(style2_pep).to(device).unsqueeze(0)
    
    model = Model(**model_paras)
    model = model.to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    
    model.eval()
    model.ensemble.change_status(training_flag=False)
    
    pred_hat = model([style1_cdr3, style1_pep], [style2_cdr3, style2_pep])
    
    return round(pred_hat.item(), 4)

# calculate rank percentile
def ensemble_data_encoding_loader(dat, style_path, cdr3_len_max, pep_len_max, pad_char, styles,
                                  batch_size, num_workers):
    style1_cdr3, style1_pep = encoding.encoding(dat, style_path, # type: ignore
                                                cdr3_len_max=cdr3_len_max, pep_len_max=pep_len_max, 
                                                pad_char=pad_char, style=styles[0], has_label=False)  # type: ignore
    style2_cdr3, style2_pep = encoding.encoding(dat, style_path, # type: ignore
                                                cdr3_len_max=cdr3_len_max, pep_len_max=pep_len_max, 
                                                pad_char=pad_char, style=styles[1], has_label=False)  # type: ignore
    style1_cdr3 = torch.from_numpy(style1_cdr3)
    style1_pep = torch.from_numpy(style1_pep)
    style2_cdr3 = torch.from_numpy(style2_cdr3)
    style2_pep = torch.from_numpy(style2_pep)
    
    tmp = data.TensorDataset(style1_cdr3, style1_pep, style2_cdr3, style2_pep)
    dataloader = data.DataLoader(dataset=tmp, batch_size=batch_size, shuffle=False, 
                                 drop_last=False, num_workers=num_workers, pin_memory=True)
    return dataloader

def ensemble_data_encoding_rank(data_path, background_file,
                                style_path, styles: list, num_workers=5, batch_size=128, cdr3_len_max=20, pep_len_max=20, pad_char='X'):
    val_dat = pd.read_csv(data_path)
    unique_epitopes = list(val_dat["Epitope"].drop_duplicates())
    background_dat = pd.read_csv(background_file)
    background_num = len(background_dat)
    
    background_pre_dat = pd.DataFrame({'CDR3': list(background_dat['CDR3'])*len(unique_epitopes),
                                       'Epitope': [epitope for epitope in unique_epitopes for _ in range(background_num)]})
    # construct dataloader
    val_dataloader = ensemble_data_encoding_loader(val_dat, style_path, cdr3_len_max, pep_len_max, pad_char, styles,
                                                   batch_size, num_workers)
    background_dataloader = ensemble_data_encoding_loader(background_pre_dat, style_path, cdr3_len_max, pep_len_max, pad_char, styles,
                                                          batch_size, num_workers)
    return val_dataloader, background_dataloader, list(val_dat["Epitope"]), unique_epitopes, background_num

def rank_cal(pred, background_preds):
    if sum([background_pred>=pred for background_pred in background_preds]) != 0:
        return sum([background_pred>=pred for background_pred in background_preds]) / len(background_preds)
    else:
        return 0

def ensemble_model_rank_calculation(data_path, background_file,
                                    style_path, Model, batch_size, model_weight_path, device, num_workers=5,
                                    cdr3_len_max=20, pep_len_max=20, pad_char='X', styles=['AAindex_11', 'BLOSUM50'], **model_paras):
    val_dataloader, background_dataloader, epitopes, unique_epitopes, background_num = ensemble_data_encoding_rank(data_path, background_file,
                                                                                                                   style_path, styles, num_workers, 
                                                                                                                   batch_size, cdr3_len_max, pep_len_max, pad_char)
    model = Model(**model_paras)
    model = model.to(device)
    loss = nn.BCELoss().to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    # prediction
    _, __, ___, pred_array_val = ensemble_model_validation(model, loss, val_dataloader, device, has_label=False)
    _, __, ___, pred_array_background = ensemble_model_validation(model, loss, background_dataloader, device, has_label=False)
    rank_percentile = []
    for i, epitope in enumerate(epitopes):
        start = unique_epitopes.index(epitope)*background_num
        end = start + background_num
        rank_percentile.append(rank_cal(pred_array_val[i], pred_array_background[start:end]))
    return list(pred_array_val), rank_percentile
  
def sin_pre_rank(cdr3, pep, background_file, style_path, Model, model_weight_path, device, 
                 cdr3_len_max=20, pep_len_max=20, pad_char='X', styles=['AAindex_11', 'BLOSUM50'], **model_paras):
    background_dat = pd.read_csv(background_file)
    background_len = len(background_dat)
    sin_pre_dat = pd.DataFrame({"CDR3": [cdr3] + list(background_dat["CDR3"]),
                                "Epitope": [pep] * (background_len+1)})
    # construct dataloader
    sin_pre_dataloader = ensemble_data_encoding_loader(sin_pre_dat, style_path, cdr3_len_max, pep_len_max, pad_char, styles,
                                                       batch_size=256, num_workers=10)
    
    model = Model(**model_paras)
    model = model.to(device)
    loss = nn.BCELoss().to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    
    _, __, ___, pred_array_sin = ensemble_model_validation(model, loss, sin_pre_dataloader, device, has_label=False)
    
    sin_pre_score = round(float(pred_array_sin[0]), 4)
    sin_pre_rank = rank_cal(pred_array_sin[0], pred_array_sin[1:])
    
    return sin_pre_score, sin_pre_rank
