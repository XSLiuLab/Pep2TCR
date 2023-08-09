import pandas as pd
import numpy as np

# 对数据进行编码，padding字符 X(padding在最后), index为 0
def paddind(string: str, pad_char='X', max_len=20):
    str_len = len(string)
    # middle_len = str_len // 2
    string = string + pad_char*(max_len-str_len)
    return string

def encoding_strategy(string: str, encoding_df, pad_char='X', style="One-hotting"):
    string = string.replace(pad_char, 'X')  # 将填充字符转换成 X
    if style in ['Embedding', 'One-hotting']:
        # 转换成数字即可, 后面调包
        encoding_str = [encoding_df["Index"][encoding_df["AminoAcid"].values == c].values[0] for c in string]
        return np.array(encoding_str)  # shape: (20, )
    elif style in ['BLOSUM50', 'BLOSUM62']:
        # 按氨基酸转化
        encoding_str = [list(encoding_df[c]) for c in string]
        return np.array(encoding_str)  # shape: (20, 20)
    elif style == 'AAindex_11':
        # 按氨基酸转化
        encoding_df["-"] = 0 # 增加一列，用于zero-setting
        encoding_str = [list(encoding_df[c]) for c in string]
        return np.array(encoding_str)  # shape: (20, 11)
    else:
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
            encoding_df = pd.read_csv(style_path[index], index_col=0).iloc[:20, :20]  # 丢弃后5列和后5行
            encoding_df['X'] = 0  # 增加一列 X
            encoding_df['-'] = 0  # 增加一列，用于zero-setting
    else:
        return None
    
    cdr3_arr = []
    pep_arr = []
    label_arr = []
    count = 0 # 对样本进行计数
    Amino_acid_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                       'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-', pad_char]
    for i in range(nsample):
        cdr3 = paddind(cdr3s[i], pad_char=pad_char, max_len=cdr3_len_max)
        pep = paddind(peps[i], pad_char=pad_char, max_len=pep_len_max)
        
        # 判断是否存在不常见氨基酸，剔除这些样本
        cdr3_mask = [a in Amino_acid_list for a in cdr3]
        pep_mask = [a in Amino_acid_list for a in pep]
        if not (all(cdr3_mask) & all(pep_mask)):
            count += 1
            continue  # 终止本次循环
        
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
    
    msg = f'共编码{cdr3_arr.shape[0]}个样本'
    # print(msg + '\n' + '-'*len(msg))
    
    return (cdr3_arr, pep_arr, label_arr) if has_label else (cdr3_arr, pep_arr)


def encoding_single(cdr3, pep, style_path, cdr3_len_max=20, pep_len_max=20, pad_char='X', style="One-hotting"):
    
    style_list = ['Embedding', 'One-hotting', 'BLOSUM50', 'BLOSUM62', 'AAindex_11']
    
    if style in style_list:
        index = style_list.index(style)
        if style in ['Embedding', 'One-hotting', 'AAindex_11']:
            encoding_df = pd.read_csv(style_path[index])
        else:
            encoding_df = pd.read_csv(style_path[index], index_col=0).iloc[:20, :20]  # 丢弃后5列和后5行
            encoding_df['X'] = 0  # 增加一列 X
            encoding_df['-'] = 0  # 增加一列，用于zero-setting
    else:
        # print("Error Enconding Strategy!!!")
        return [0] * 2

    Amino_acid_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                       'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-', pad_char]
    
    cdr3 = paddind(cdr3, pad_char=pad_char, max_len=cdr3_len_max)
    pep = paddind(pep, pad_char=pad_char, max_len=pep_len_max)
    
    # 判断是否存在不常见氨基酸
    cdr3_mask = [a in Amino_acid_list for a in cdr3]
    pep_mask = [a in Amino_acid_list for a in pep]
    if not (all(cdr3_mask) & all(pep_mask)):
        # print("存在异常氨基酸, 终止预测！")
        return [0] * 2
    
    encoding_cdr3 = encoding_strategy(cdr3, encoding_df, pad_char=pad_char, style=style)
    encoding_pep = encoding_strategy(pep, encoding_df, pad_char=pad_char, style=style)
    
    return encoding_cdr3, encoding_pep
