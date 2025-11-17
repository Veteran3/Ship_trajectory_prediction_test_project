import numpy as np

def RSE(pred, true, mask=None):
    """
    计算 RSE (Root Relative Squared Error)，支持掩码。
    """
    if mask is None:
        pred_valid = pred
        true_valid = true
    else:
        # 1. 扩展掩码以匹配 pred/true 的维度
        mask_expanded = np.expand_dims(mask, axis=-1)
        # 2. [修正] 显式重复 (repeat) 最后一个维度
        mask_full = mask_expanded.repeat(pred.shape[-1], axis=-1) # (B,T,N,1) -> (B,T,N,2)
        
        # 3. 只选择有效的数据点
        pred_valid = pred[mask_full]
        true_valid = true[mask_full]
    
    # 仅在有效数据上计算
    true_mean = np.mean(true_valid)
    numerator = np.sqrt(np.sum((true_valid - pred_valid) ** 2))
    denominator = np.sqrt(np.sum((true_valid - true_mean) ** 2))
    
    if denominator == 0:
        return np.inf  # 避免除以零
    return numerator / denominator


def CORR(pred, true, mask=None):
    """
    计算 CORR (Correlation)，支持掩码。
    在所有有效点上，按特征维度(D)计算相关性，然后平均。
    """
    if mask is None:
        # 如果没有掩码，仍然 reshape 以便 axis=0 操作
        pred_flat = pred.reshape(-1, pred.shape[-1])
        true_flat = true.reshape(-1, true.shape[-1])
    else:
        # 1. 扩展并重复掩码
        mask_expanded = np.expand_dims(mask, axis=-1)
        mask_full = mask_expanded.repeat(pred.shape[-1], axis=-1) # (B,T,N,2)
        
        # 2. 过滤数据
        pred_valid = pred[mask_full] # (N_valid_points * D,)
        true_valid = true[mask_full] # (N_valid_points * D,)
        
        # 3. Reshape 为 (N_valid_points, D)
        D = pred.shape[-1]
        pred_flat = pred_valid.reshape(-1, D)
        true_flat = true_valid.reshape(-1, D)

    # --- 您的原始 CORR 逻辑 ---
    # (现在在 (N_points, D) 形状的数据上正确运行)
    u = ((true_flat - true_flat.mean(0)) * (pred_flat - pred_flat.mean(0))).sum(0)
    d = np.sqrt(((true_flat - true_flat.mean(0)) ** 2 * (pred_flat - pred_flat.mean(0)) ** 2).sum(0))
    d = np.where(d == 0, 1e-5, d) # 避免除以零
    return (u / d).mean(-1)


def MAE(pred, true, mask=None):
    """
    计算 MAE (Mean Absolute Error)，支持掩码。
    """
    error = np.abs(pred - true) # (B, T, N, 2)
    if mask is None:
        return np.mean(error)
    
    # [修正]
    mask_expanded = np.expand_dims(mask, axis=-1) # (B, T, N, 1)
    mask_full = mask_expanded.repeat(pred.shape[-1], axis=-1) # (B, T, N, 2)
    
    valid_errors = error[mask_full] # (N_valid_points * D,)
    return np.mean(valid_errors)


def MSE(pred, true, mask=None):
    """
    计算 MSE (Mean Squared Error)，支持掩码。
    """
    error = (pred - true) ** 2 # (B, T, N, 2)
    if mask is None:
        return np.mean(error)
    
    # [修正]
    mask_expanded = np.expand_dims(mask, axis=-1)
    mask_full = mask_expanded.repeat(pred.shape[-1], axis=-1)
    
    valid_errors = error[mask_full]
    return np.mean(valid_errors)


def RMSE(pred, true, mask=None):
    """
    计算 RMSE (Root Mean Squared Error)，支持掩码。
    (通过调用已掩码的 MSE)
    """
    return np.sqrt(MSE(pred, true, mask))


def MAPE(pred, true, mask=None):
    """
    计算 MAPE (Mean Absolute Percentage Error)，支持掩码。
    """
    true_safe = true + 1e-5 # 避免除以零
    error = np.abs((pred - true) / true_safe)
    
    if mask is None:
        return np.mean(error)
        
    # [修正]
    mask_expanded = np.expand_dims(mask, axis=-1)
    mask_full = mask_expanded.repeat(pred.shape[-1], axis=-1)
    
    valid_errors = error[mask_full]
    return np.mean(valid_errors)


def MSPE(pred, true, mask=None):
    """
    计算 MSPE (Mean Squared Percentage Error)，支持掩码。
    """
    true_safe = true + 1e-5 # 避免除以零
    error = np.square((pred - true) / true_safe)
    
    if mask is None:
        return np.mean(error)
        
    # [修正]
    mask_expanded = np.expand_dims(mask, axis=-1)
    mask_full = mask_expanded.repeat(pred.shape[-1], axis=-1)
    
    valid_errors = error[mask_full]
    return np.mean(valid_errors)


def metric(pred, true, mask=None):
    """
    计算所有指标 (已更新为传递掩码)
    
    Args:
        pred: [B, T, N, D] - 预测值
        true: [B, T, N, D] - 真实值
        mask: [B, T, N] - 掩码 (bool)
    
    Returns:
        mae, mse, rmse, mape, mspe
    """
    mae = MAE(pred, true, mask)
    mse = MSE(pred, true, mask)
    rmse = RMSE(pred, true, mask)
    mape = MAPE(pred, true, mask)
    mspe = MSPE(pred, true, mask)
    
    return mae, mse, rmse, mape, mspe

def ADE(pred, true, mask=None):
    """
    计算平均位移误差 (Average Displacement Error)
    
    Args:
        pred: [B, T, N, 2] - 预测的经纬度坐标
        true: [B, T, N, 2] - 真实的经纬度坐标
    
    Returns:
        ade: 平均位移误差
    """
    # 计算每个时间步和每个船只的欧氏距离

    distance = np.linalg.norm(pred - true, axis=-1)
    if mask is not None:
        if mask.dtype != bool:
            mask = mask.astype(bool)
        distance = distance[mask]
    
    return np.mean(distance)

def FDE(pred, true, mask=None):
    """
    计算最终位移误差 (Final Displacement Error)
    
    Args:
        pred: [B, T, N, 2] - 预测的经纬度坐标
        true: [B, T, N, 2] - 真实的经纬度坐标
    
    Returns:
        fde: 最终位移误差
    """
    # 获取最后一个时间步的预测和真实值
    pred_final = pred[:, -1, :, :2]  # [B, N, 2]
    true_final = true[:, -1, :, :2]  # [B, N, 2]

    # 计算最终位移误差
    distance = np.linalg.norm(pred_final - true_final, axis=-1)  # [B, N]
    if mask is not None:
        mask = mask[:, -1, :]  # 只考虑最后一个时间步的掩码
        if mask.dtype != bool:
            mask = mask.astype(bool)
        distance = distance[mask]
    
    return np.mean(distance)