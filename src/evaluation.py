import csv
import numpy as np
import pandas as pd
from metrics import *

def evaluate(predict_path, answer_path, average=False, with_header=False):
    """
    Parameters:
    ----------
    predict_path: path for predictions in csv format
    answer_path:  path for answer in csv format
    Format:
        prediction: n*(2+k) 2d-array
            n: number of test instances
            k: number of items recommended
            user_id, event_time, rec_1, rec_2, ... , rec_k
        answer: n*k 2d-array
            n: number of test instances
            k: columns = [user_id, ans_id,..., event_time]
    Return:
    ----------
    Dictionary: 
        type(key): str,
        type(value): float/[float]
        if average == True:
            {'hr':score,'mrr':score,'ndcg':score}
        else #average == False:
            {'hr':[score...],'mrr':[score...],'ndcg':[score...]}
    """
    
    df_pre = pd.read_csv(predict_path, header=None)
    df_ans = pd.read_csv(answer_path, header=None)

    # drop [user_id, event_time]
    if with_header:
        targets = df_ans.loc[1:,1].values.astype(int) # drop header and choose the column of targets
        ranked_lists = df_pre.loc[:,:].values.astype(int) # drop header and choose columns of ranked items
    else:
        targets = df_ans.loc[:,1].values.astype(int) # drop header and choose the column of targets
        ranked_lists = df_pre.loc[:,:].values.astype(int) # drop header and choose columns of ranked items

    # get scores of list/average
    hr = get_hit_ratio(ranked_lists,targets,average)
    mrr = get_MRR(ranked_lists,targets,average)
    ndcg = get_NDCG(ranked_lists,targets,average)
    return {'hr':hr,'mrr':mrr,'ndcg':ndcg}