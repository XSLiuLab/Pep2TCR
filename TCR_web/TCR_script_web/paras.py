import torch
from functions import LSTM_Double_AA, CNN_Double_BLOSUM
import os

################################# Change ab_path ####################################
# Please provide the absolute path to TCR_web dir
ab_path = "~/R/TCR_Researchs/TCR_web/"
#####################################################################################

cd4_model_weight_path_avg =  os.path.join(ab_path, "TCR_script_web/weights/Avg/cd4_avg_ensemble_random_1.pt")
has_label = False

batch_size = 256

device_id = 0
device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

num_workers = 10

cdr3_len_max = 20
pep_len_max = 20
pad_char = 'X'
styles = ['AAindex_11', 'BLOSUM50']

prefix = os.path.join(ab_path, 'TCR_script_web/Encoding_data/')
style_path = [prefix+'One-Hotting.csv', prefix+'One-Hotting.csv',
              prefix+'BLOSUM50.csv', prefix+'BLOSUM62.csv',
              prefix+'aaindex_processed_pca.csv']

#####################################################################################
w1 = 0.5

Model1 = LSTM_Double_AA
Model2 = CNN_Double_BLOSUM

model1_paras = {
    'feature_num': 11,
    'hidden_num': 80,
    'rnn_layer': 1,
    'dropout': 0.3
}

model2_paras = {
    'feature_num': 20,
    'filter_num': 4,
    'hidden_num': 32,
    'dropout': 0.3
}

model1_weights_path = os.path.join(ab_path, 'TCR_script_web/weights/LSTM')
model2_weights_path = os.path.join(ab_path, 'TCR_script_web/weights/CNN')
