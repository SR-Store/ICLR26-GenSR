import numpy as np


def RMSE_score(true, pred):
    true, pred = np.array(true), np.array(pred)
    return np.sqrt(np.mean((true - pred) ** 2))


def R2_score(true, pred):
    true, pred = np.array(true), np.array(pred)
    return 1 - np.sum((true - pred) ** 2) / np.sum((true - np.mean(true)) ** 2)


def kendall_rank_score(true, pred):
    true, pred = np.array(true), np.array(pred)
    N = len(true)
    tau = sum(np.sign((true[i] - true[j]) * (pred[i] - pred[j])) for i in range(N) for j in range(N))
    return tau / (N * (N - 1))


def spearman_rank_score(true, pred):
    true, pred = np.array(true), np.array(pred)
    N = len(true)
    true_rank = np.argsort(np.argsort(true)[::-1])
    pred_rank = np.argsort(np.argsort(pred)[::-1])
    d = sum((true_rank[i] - pred_rank[i]) ** 2 for i in range(N))
    return 1 - 6 * d / (N * (N ** 2 - 1))


def pearson_score(true, pred):
    true, pred = np.array(true), np.array(pred)
    return np.corrcoef(true, pred)[0, 1]


def AUC_score(true, pred):
    true, pred = np.array(true), np.array(pred)
    N = len(true)
    tmp = sum(int(pred[i] > pred[j]) for i in range(N) for j in range(N) if true[i] > true[j])
    return tmp / (N * (N - 1)) * 2


def NDCG_score(true, pred, k=5):
    true, pred = np.array(true), np.array(pred)
    DCG  = np.sum((2 ** true[np.argsort(pred)[::-1]][:k] - 1) / np.log2(np.arange(k) + 2))
    IDCG = np.sum((2 ** true[np.argsort(true)[::-1]][:k] - 1) / np.log2(np.arange(k) + 2))
    return DCG / IDCG

