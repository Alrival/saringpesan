import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from utils.data import clean_up_text

import pandas as pd
import numpy as np
from tabulate import tabulate

use_encoding = 'ISO-8859-1'

###
# Calculate Metrics Function
###

def get_metrics(list_hyp, list_label):
    metrics = {}
    metrics["ACC"] = accuracy_score(list_label, list_hyp)
    metrics["F1"] = f1_score(list_label, list_hyp, average='macro', zero_division=0)
    metrics["REC"] = recall_score(list_label, list_hyp, average='macro', zero_division=0)
    metrics["PRE"] = precision_score(list_label, list_hyp, average='macro', zero_division=0)
    return metrics

def metrics_to_string(metric_dict):
    string_list = []
    for key, value in metric_dict.items():
        string_list.append('{}:{:.2f}'.format(key, value))
    return ' '.join(string_list)

def predict_string(string, i2w, model, tokenizer):
    subwords = clean_up_text(string)
    subwords = tokenizer.encode(subwords)
    subwords = torch.LongTensor(subwords).view(1, -1)

    logits = model(subwords)[0]
    label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()

    return {'label': i2w[label],
            'prediction': '{:.2f}%'.format(F.softmax(logits, dim=-1).squeeze()[label] * 100),
            'sentence': string}

def _calculate_percentage(value, total):
    return (value/total)*100

def _infer_CMatrix(dataframe):
    data = []
    col_name = []
    """
    [ TP FN FN ] [ TN FP TN ] [ TN TN FP ]
    [ FP TN TN ] [ FN TP FN ] [ TN TN FP ]
    [ FP TN TN ] [ TN FP TN ] [ FN FN TP ]
    """
    for col in dataframe.columns:
        TP = dataframe.loc[col,col]
        FN = dataframe.loc[col,:].sum() - TP
        FP = dataframe.loc[:,col].sum() - TP
        TN = dataframe.values.sum() - (TP+FN+FP)
        data.append([_calculate_percentage(TP, dataframe.values.sum()),
                     _calculate_percentage(FN, dataframe.values.sum()),
                     _calculate_percentage(FP, dataframe.values.sum()),
                     _calculate_percentage(TN, dataframe.values.sum())])
        col_name.append(col)
    
    table = pd.DataFrame(data, index=col_name, columns=['TP','FN','FP','TN'])
    table = table.append(pd.DataFrame([[table['TP'].mean().round(4),
                                        table['FN'].mean().round(4),
                                        table['FP'].mean().round(4),
                                        table['TN'].mean().round(4)]],
                                      index=['Rata-rata'], columns=['TP','FN','FP','TN']))
    return table

def confusion_matrix(list_actual, list_predict):
    data = {
        'actual': list_actual,
        'predict': list_predict
    }
    df = pd.DataFrame(data, columns=['actual','predict'])
    
    matrix = pd.crosstab(df['actual'], df['predict'], rownames=['Actual'], colnames=['Predicted'])
    cmatrix = _infer_CMatrix(matrix)
    
    return {'CM': matrix,
            'inferred': cmatrix}
            
def print_CMatrix(filepath, i2w, model, tokenizer):
    a_label = []
    p_label = []
    with open(filepath, "r+", encoding=use_encoding) as fs:
        raw_content = fs.read()
        content_list = raw_content.split('\n\n')
        """
            Data format:
                sentence label
        """
        for content in content_list:
            try:
                if content:
                    text,flag = content.split('\t')
                    predict = predict_string(text, i2w, model, tokenizer)
                    a_label.append(flag)
                    p_label.append(predict['label'])
            except:
                print('[WARN] something went wrong when parsing data')
                print('[DEBUG] {}'.format(content))
                continue
        
        cm = confusion_matrix(a_label,p_label)        
        print(tabulate(cm['CM'], headers='keys', tablefmt='psql'))
        print(tabulate(cm['inferred'], headers='keys', tablefmt='psql'))