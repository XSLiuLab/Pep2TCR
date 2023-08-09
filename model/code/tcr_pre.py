from .functions import sin_pre, ensemble_model_validation_simple, Avg_Ensemble
from .paras import *

import pandas as pd
import random
import os
import torch

def main(args):
    if torch.cuda.is_available():
        print("Find GPU device avaliable, using GPU mode!")
    else:
        print("Don't find GPU device avaliable, using CPU mode!")

    if args.mode == "single":
        if (8<=len(args.cdr3)<=20) & (9<=len(args.pep)<=20):
            prediction = sin_pre(args.cdr3, args.pep, style_path, Avg_Ensemble, 
                                cd4_model_weight_path_avg, device, cdr3_len_max, pep_len_max, pad_char, styles, 
                                Model1=Model1, Model2=Model2, model1_paras=model1_paras, 
                                model2_paras=model2_paras, e_device=device, w1=w1,
                                model1_weights_path=model1_weights_path, model2_weights_path=model2_weights_path)
        else:
            print("Please ensure the length of CDR3 and peptide are 8-20 and 9-20, respectively!!!")

        return None
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
        if ~all(mask):
            info = ', '.join(list(map(str, ((~mask[~mask]).index + 1).tolist()[:5]))) + " ..."
            print(f'Notice: the sample(s) of {info}, please ensure the length of CDR3 and peptide are 8-20 and 9-20, respectively!!!')
        df.to_csv(os.path.join(args.outdir, os.path.basename(args.data_path)), index=False)
        return None
