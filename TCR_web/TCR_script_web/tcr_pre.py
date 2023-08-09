import argparse
from functions import sin_pre, ensemble_model_validation_simple, Avg_Ensemble
from paras import *

import pandas as pd
import random
import os

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
        df = pd.read_csv(args.data_path)
        cdr3_lens = df.CDR3.str.len()
        pep_lens = df.Epitope.str.len()
        mask_cdr3 = (cdr3_lens>=8) & (cdr3_lens<=20)
        mask_pep = (pep_lens>=9) & (pep_lens<=20)
        mask = mask_cdr3 & mask_pep
        
        _, __, ___, pred_array = ensemble_model_validation_simple(args.data_path, style_path, Avg_Ensemble, 
                                                                  batch_size, cd4_model_weight_path_avg, device, has_label, num_workers, 
                                                                  cdr3_len_max, pep_len_max, pad_char, styles, 
                                                                  Model1=Model1, Model2=Model2, model1_paras=model1_paras, 
                                                                  model2_paras=model2_paras, e_device=device, w1=w1,
                                                                  model1_weights_path=model1_weights_path, model2_weights_path=model2_weights_path)
        df["Score"] = list(pred_array)
        df.loc[~mask, "Score"] = None # prediction failure
        random_int = random.randint(1, 10000)
        df.to_csv(os.path.join(ab_path, f'tmp/batch_{random_int}.csv'), index=False)
        print(random_int)
        return random_int  # According to random_int to ensure a file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="single", choices=["single", "batch"])
    parser.add_argument('--cdr3', type=str)
    parser.add_argument('--pep', type=str)
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()
    
    main(args)
