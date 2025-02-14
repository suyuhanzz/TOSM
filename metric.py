from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd
import torch.nn as nn
import torch



def get_precision(mlb, predict, targets, topk):
    """
    使用宏观（macro-average）方式计算 Precision@k、Recall@k 和 F1@k。

    参数:
        mlb: 一个 MultiLabelBinarizer 对象，用于将 targets 转换为二值矩阵。
        predict: 可为列表、NumPy 数组或 Tensor，形状为 [batch_size, topk]，
                 每行存放预测的 topk 标签 id。
        targets: 长度为 batch_size 的列表，每个元素为该样本真实标签 id 的集合或列表。
        topk: int，预测中取前 k 个标签。

    返回:
        macro_precision, macro_recall, macro_f1（均为 torch.Tensor 标量）
    """
    # 将真实标签转换为二值矩阵，形状为 [batch_size, num_classes]
    # mlb.transform 返回的是 NumPy 数组，因此再转成 torch.Tensor
    targets2 = torch.from_numpy(mlb.transform(targets)).float()
    
    # 将预测标签构造成二值矩阵
    # predict 若不是 torch.Tensor，则先转换；确保类型为 long 用于 scatter_
    if not torch.is_tensor(predict):
        predict = torch.tensor(predict).long()
    else:
        predict = predict.long()
        
    batch_size = predict.shape[0]
    num_classes = len(mlb.classes_)
    # 构造一个全零矩阵，然后用 scatter_ 在每个样本中将预测的 topk 标签位置置为 1
    predict2 = torch.zeros(batch_size, num_classes)
    predict2 = predict2.scatter_(1, predict, 1).float()
    
    # 计算每个样本中预测正确的标签个数
    correct_per_sample = (targets2 * predict2).sum(dim=1).float()
    
    # 计算每个样本的 Precision@k：正确预测数除以 topk
    precision_per_sample = correct_per_sample / topk
    
    # 计算每个样本的 Recall@k：正确预测数除以真实标签个数
    true_counts = targets2.sum(dim=1).float()
    # 为避免除零，当某个样本没有真实标签时，定义 recall 为 0
    recall_per_sample = torch.where(true_counts > 0, correct_per_sample / true_counts, torch.zeros_like(true_counts))
    
    # 计算每个样本的 F1@k：若 precision + recall 为 0，则 F1 定为 0，否则按调和平均计算
    f1_per_sample = torch.where((precision_per_sample + recall_per_sample) > 0,
                                2 * precision_per_sample * recall_per_sample / (precision_per_sample + recall_per_sample),
                                torch.zeros_like(precision_per_sample))
    
    # 对所有样本取平均，得到宏观指标
    macro_precision = precision_per_sample.mean()
    macro_recall = recall_per_sample.mean()
    macro_f1 = f1_per_sample.mean()
    
    return macro_precision, macro_recall, macro_f1

def get_ndcg(mlb, predict, targets, topk):
    log = 1.0 / np.log2(np.arange(topk) + 2)
    dcg = np.zeros((targets.shape[0], 1))
    targets_bin = mlb.transform(targets)  # 保持 targets 二值矩阵
    # 假设 predict 是一个 NumPy 数组，形状为 (n_samples, topk)
    for i in range(topk):
        # 不重写原始的 predict，而是取当前 rank 的预测
        pred_slice = predict[:, i:i+1]
        
        # 将 pred_slice 转换为对应的二值表示 p
        p = torch.zeros(len(pred_slice), len(mlb.classes_)).scatter_(1, torch.tensor(pred_slice), 1).cpu().numpy()
        
        a1 = np.multiply(p, targets_bin)
        a2 = np.sum(a1, axis=-1, keepdims=True)
        dcg += a2 * log[i]
        
    # 计算理想累计折扣（IDCG）
    # targets_bin.sum(axis=-1) 得到每个样本真实标签数
    ideal_ranks = np.minimum(targets_bin.sum(axis=-1), topk)  # 每个样本实际能获得的最大命中数
    # 注意：索引从0开始，所以 ideal_index = ideal_ranks - 1
    ideal_index = ideal_ranks.astype(int) - 1
    # 使用 log.cumsum() 计算 IDCG，对每个样本取对应的值
    idcg_all = log.cumsum()
    # 避免 ideal_index 负值（若样本真实标签为空，则设为1，结果为0）
    idcg = np.array([idcg_all[idx] if idx >= 0 else 1 for idx in ideal_index])
    
    ndcg = dcg.flatten() / idcg
    answer = np.average(ndcg)
    
    return answer
