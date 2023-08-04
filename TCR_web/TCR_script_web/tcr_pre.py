import argparse
from functions import sin_pre, ensemble_model_validation_simple, Avg_Ensemble
from paras import *

import pandas as pd
import random

def main(args):
    if args.mode == "single":
        prediction = sin_pre(args.cdr3, args.pep, style_path, Avg_Ensemble, 
                             cd4_model_weight_path_avg, device, cdr3_len_max, pep_len_max, pad_char, styles, 
                             Model1=Model1, Model2=Model2, model1_paras=model1_paras, 
                             model2_paras=model2_paras, e_device=device, w1=w1,
                             model1_weights_path=model1_weights_path, model2_weights_path=model2_weights_path)
        print(prediction)
        return prediction
    else:
        _, __, ___, pred_array = ensemble_model_validation_simple(args.data_path, style_path, Avg_Ensemble, 
                                                                  batch_size, cd4_model_weight_path_avg, device, has_label, num_workers, 
                                                                  cdr3_len_max, pep_len_max, pad_char, styles, 
                                                                  Model1=Model1, Model2=Model2, model1_paras=model1_paras, 
                                                                  model2_paras=model2_paras, e_device=device, w1=w1,
                                                                  model1_weights_path=model1_weights_path, model2_weights_path=model2_weights_path)
        df = pd.read_csv(args.data_path)
        df["Score"] = list(pred_array)
        random_int = random.randint(1, 10000)
        df.to_csv(ab_path + f'tmp/batch_{random_int}.csv', index=False)
        print(random_int)
        return random_int  # 根据随机数确定文件

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="single", choices=["single", "batch"])
    parser.add_argument('--cdr3', type=str)
    parser.add_argument('--pep', type=str)
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()
    
    main(args)