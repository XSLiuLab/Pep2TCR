import pandas as pd
import numpy as np

def paddind(string: str, pad_char='X', max_len=20):
    str_len = len(string)
    # middle_len = str_len // 2
    string = string + pad_char*(max_len-str_len)
    return string

def encoding_strategy(string: str, encoding_df, pad_char='X', style="One-hotting"):
    string = string.replace(pad_char, 'X')
    if style in ['Embedding', 'One-hotting']:
        encoding_str = [encoding_df["Index"][encoding_df["AminoAcid"].values == c].values[0] for c in string]
        return np.array(encoding_str)  # shape: (20, )
    elif style in ['BLOSUM50', 'BLOSUM62']:
        encoding_str = [list(encoding_df[c]) for c in string]
        return np.array(encoding_str)  # shape: (20, 20)
    elif style == 'AAindex_11':
        encoding_df["-"] = 0
        encoding_str = [list(encoding_df[c]) for c in string]
        return np.array(encoding_str)  # shape: (20, 11)
    else:
        print("Error Enconding Strategy!!!")
        return None

def encoding(dat, style_path, cdr3_len_max=20, pep_len_max=20, pad_char='X', style="One-hotting", has_label=True):
    cdr3s = dat["CDR3"]
    peps = dat["Epitope"]
    if has_label:
        labels = dat["label"]
    else:
        labels = []
    nsample = len(dat)
    
    style_list = ['Embedding', 'One-hotting', 'BLOSUM50', 'BLOSUM62', 'AAindex_11']
    
    if style in style_list:
        index = style_list.index(style)
        if style in ['Embedding', 'One-hotting', 'AAindex_11']:
            encoding_df = pd.read_csv(style_path[index])
        else:
            encoding_df = pd.read_csv(style_path[index], index_col=0).iloc[:20, :20]
            encoding_df['X'] = 0
            encoding_df['-'] = 0
    else:
        print("Error Enconding Strategy!!!")
        return None
    
    cdr3_arr = []
    pep_arr = []
    label_arr = []
    count = 0
    Amino_acid_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                       'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-', pad_char]
    for i in range(nsample):
        cdr3 = paddind(cdr3s[i], pad_char=pad_char, max_len=cdr3_len_max)
        pep = paddind(peps[i], pad_char=pad_char, max_len=pep_len_max)
        
        cdr3_mask = [a in Amino_acid_list for a in cdr3]
        pep_mask = [a in Amino_acid_list for a in pep]
        if not (all(cdr3_mask) & all(pep_mask)):
            count += 1
            print("The number {} sample exists anomalous amino acids".format(count))
            continue
        
        encoding_cdr3 = encoding_strategy(cdr3, encoding_df, pad_char=pad_char, style=style)
        encoding_pep = encoding_strategy(pep, encoding_df, pad_char=pad_char, style=style)
        
        cdr3_arr.append(encoding_cdr3)
        pep_arr.append(encoding_pep)
        if has_label:
            label_arr.append(labels[i])
        count += 1
    
    cdr3_arr = np.array(cdr3_arr)
    pep_arr = np.array(pep_arr)
    if has_label:
        label_arr = np.array(label_arr)
    
    msg = f'Encoding {cdr3_arr.shape[0]} sample(s)'
    print(msg + '\n' + '-'*len(msg))
    
    return (cdr3_arr, pep_arr, label_arr) if has_label else (cdr3_arr, pep_arr)


def encoding_single(cdr3, pep, style_path, cdr3_len_max=20, pep_len_max=20, pad_char='X', style="One-hotting"):
    
    style_list = ['Embedding', 'One-hotting', 'BLOSUM50', 'BLOSUM62', 'AAindex_11']
    
    if style in style_list:
        index = style_list.index(style)
        if style in ['Embedding', 'One-hotting', 'AAindex_11']:
            encoding_df = pd.read_csv(style_path[index])
        else:
            encoding_df = pd.read_csv(style_path[index], index_col=0).iloc[:20, :20]
            encoding_df['X'] = 0
            encoding_df['-'] = 0
    else:
        print("Error Enconding Strategy!!!")
        return [0] * 2

    Amino_acid_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                       'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-', pad_char]
    
    cdr3 = paddind(cdr3, pad_char=pad_char, max_len=cdr3_len_max)
    pep = paddind(pep, pad_char=pad_char, max_len=pep_len_max)
    
    cdr3_mask = [a in Amino_acid_list for a in cdr3]
    pep_mask = [a in Amino_acid_list for a in pep]
    if not (all(cdr3_mask) & all(pep_mask)):
        print("Presence of anomalous amino acids, prediction terminated.")
        return [0] * 2
    
    encoding_cdr3 = encoding_strategy(cdr3, encoding_df, pad_char=pad_char, style=style)
    encoding_pep = encoding_strategy(pep, encoding_df, pad_char=pad_char, style=style)
    
    return encoding_cdr3, encoding_pep
