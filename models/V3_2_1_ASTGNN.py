import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
from utils.get_loss_function import get_loss_function

"""
v3.2.0版本修改 

更换损失，

添加课程学习的预定采样衰减策略



"""
# ============================================================================
# [新增loss函数] Lane Alignment Loss (航道对齐损失) 让预测向量和意图向量的夹角趋近于 0
# ============================================================================

def compute_lane_alignment_loss(pred_deltas, intent_vectors, mask_y):
    """
    pred_deltas:    [B, T, N, 2] (模型的输出)
    intent_vectors: [B, T, N, 2] (Intent Module 计算出的加权航道方向)
    """
    # 1. 计算余弦相似度
    # dim=-1 表示在 (lat, lon) 维度上计算
    # 结果范围 [-1, 1], 1 表示完全同向
    cos_sim = F.cosine_similarity(pred_deltas, intent_vectors, dim=-1, eps=1e-6)

    # 2. 构造 Loss
    # 我们希望 sim = 1, 所以 Loss = 1 - sim
    loss_align = 1.0 - cos_sim

    # 3. Mask 处理
    if mask_y.dtype != torch.bool: mask_y = mask_y.bool()
    num_valid = mask_y.sum() + 1e-6
    
    loss_align = (loss_align * mask_y).sum() / num_valid
    
    return loss_align

# ============================================================================
# [新增loss函数] FDE Loss (Final Displacement Error) —— “终点锚定”
# ============================================================================
def compute_fde_loss(pred_absolute, y_truth_abs, mask_y):
    """
    只计算序列最后一个点的误差
    pred_absolute: [B, T, N, 2]
    """
    # 1. 取出最后一个时间步 (-1)
    pred_final = pred_absolute[:, -1, :, :]   # [B, N, 2]
    truth_final = y_truth_abs[:, -1, :, :]    # [B, N, 2]
    mask_final = mask_y[:, -1, :]             # [B, N] (只看终点是否有效)

    # 2. 计算平方距离 (MSE)
    # sum(dim=-1) 把 (lat, lon) 的平方差加起来
    loss_fde = ((pred_final - truth_final) ** 2).sum(dim=-1) # [B, N]

    # 3. Mask 处理 & 求平均
    num_valid = mask_final.sum() + 1e-6
    loss_fde = (loss_fde * mask_final).sum() / num_valid
    
    return loss_fde

# ============================================================================
# [新增模块] 物理意图感知模块
# ============================================================================
class PhysicsIntentModule(nn.Module):
    def __init__(self, temperature=0.1, learnable_temp=True, allow_reverse=False):
        """
        物理意图感知模块: 通过计算 COG 和 候选航道 的几何相似度, 
        动态选择最匹配的航道方向。
        
        Args:
            temperature: Softmax温度系数。越小越"果断"。
            learnable_temp: 是否学习温度系数。
            allow_reverse: 是否允许逆行匹配 (取绝对值)。
        """
        super().__init__()
        self.allow_reverse = allow_reverse
        
        if learnable_temp:
            self.temperature = nn.Parameter(torch.tensor([temperature]))
        else:
            self.register_buffer('temperature', torch.tensor([temperature]))

    def forward(self, ship_cog, lane_feats):
        """
        Args:
            ship_cog: [B, N, T, 2] -> (cos, sin)
            lane_feats: [B, N, T, 4] -> (cos1, sin1, cos2, sin2)
        
        Returns:
            intent_vector: [B, N, T, 2] -> 加权融合后的环境指引方向
            attn_weights:  [B, N, T, 2] -> 注意力权重
        """
        B, N, T, _ = ship_cog.shape
        
        # 1. 拆解候选向量 [B, N, T, 2(候选数), 2(维度)]
        candidates = lane_feats.view(B, N, T, 2, 2)
        
        # 2. Mask (处理 Padding: 模长接近0的向量)
        cand_norms = torch.norm(candidates, dim=-1) # [B, N, T, 2]
        valid_mask = cand_norms > 1e-4             
        
        # 3. 物理交互 (Dot Product)
        # ship_cog: [B, N, T, 1, 2]
        if self.allow_reverse:
            scores = torch.abs(torch.sum(ship_cog.unsqueeze(3) * candidates, dim=-1))
        else:
            scores = torch.sum(ship_cog.unsqueeze(3) * candidates, dim=-1)
        
        # 4. Mask 无效航道 (-1e9)
        scores = scores.masked_fill(~valid_mask, -1e9)
        
        # 5. Softmax 决策
        temp = torch.clamp(self.temperature, min=1e-3)
        attn_weights = F.softmax(scores / temp, dim=-1) # [B, N, T, 2]
        
        # 6. 加权融合
        # weights: [B, N, T, 2, 1] * candidates: [B, N, T, 2, 2] -> sum dim 3
        intent_vector = (attn_weights.unsqueeze(-1) * candidates).sum(dim=3)
        
        return intent_vector, attn_weights


#  曲率损失
def direction_loss(y_pred_abs, y_truth_abs, y_mask=None, speed_threshold=1e-3, eps=1e-6):
    """
    计算预测轨迹与真实轨迹的方向一致性 Loss (1 - CosineSimilarity)。
    """
    # 0. 基础校验
    B, T, N, D = y_pred_abs.shape
    if T < 2: return y_pred_abs.new_tensor(0.0)

    # 1. 计算增量向量
    v_pred = y_pred_abs[:, 1:, :, :] - y_pred_abs[:, :-1, :, :]
    v_true = y_truth_abs[:, 1:, :, :] - y_truth_abs[:, :-1, :, :]

    # 2. 速度阈值掩码
    v_true_norm = torch.norm(v_true, dim=-1)
    m_speed = (v_true_norm > speed_threshold).float() 

    # 3. 计算余弦相似度
    cos_sim = F.cosine_similarity(v_pred, v_true, dim=-1, eps=eps)

    # 4. 构造 Loss
    loss_dir = 1.0 - cos_sim

    # 5. 综合 Mask 处理
    final_mask = m_speed
    if y_mask is not None:
        m_valid_seq = y_mask[:, :-1, :] * y_mask[:, 1:, :]
        final_mask = final_mask * m_valid_seq

    # 6. 计算加权平均
    num_valid = final_mask.sum()
    if num_valid < 1.0:
        return y_pred_abs.new_tensor(0.0)

    loss = (loss_dir * final_mask).sum() / num_valid
    return loss

# 矩阵稀疏性测试
def analyze_sparsity(matrix, name="Matrix"):
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    total_elements = matrix.size
    non_zero_elements = np.count_nonzero(matrix)
    zero_elements = total_elements - non_zero_elements
    sparsity = zero_elements / total_elements
    density = non_zero_elements / total_elements
    print(f"--- {name} Sparsity Analysis ---")
    print(f"Non-Zero: {non_zero_elements} ({density*100:.4f}%)")
    if matrix.ndim > 2 and matrix.shape[-1] <= 10:
        num_channels = matrix.shape[-1]
        print(f"--- Per-Channel Breakdown ({num_channels} channels) ---")
        for k in range(num_channels):
            channel_data = matrix[..., k]
            c_nz = np.count_nonzero(channel_data)
            print(f"  Ch{k}: Non-zeros = {c_nz}")
    print("-" * 30 + "\n")


# ----------------------------------------------------------------------
# 模块 1: 帮助函数
# ----------------------------------------------------------------------
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size, device):
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return (torch.from_numpy(mask) == 0).to(device)

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

# ----------------------------------------------------------------------
# 模块 2: 时间趋势感知注意力
# ----------------------------------------------------------------------
class TrendAwareAttention(nn.Module):
    def __init__(self, d_model, num_heads, kernel_size=3, mode='1d', dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.mode = mode
        self.kernel_size = kernel_size

        if self.mode == 'causal':
            self.causal_padding = (self.kernel_size - 1)
            self.conv_q = nn.Conv1d(d_model, d_model, kernel_size)
            self.conv_k = nn.Conv1d(d_model, d_model, kernel_size)
        else:
            self.conv_q = nn.Conv1d(d_model, d_model, kernel_size, padding='same')
            self.conv_k = nn.Conv1d(d_model, d_model, kernel_size, padding='same')

        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        B, N, T_q, D = query.shape
        _, _, T_k, _ = key.shape
        
        q_conv_in = query.reshape(B*N, T_q, D).transpose(1, 2)
        k_conv_in = key.reshape(B*N, T_k, D).transpose(1, 2)
        
        if self.mode == 'causal':
            q_conv_in = F.pad(q_conv_in, (self.causal_padding, 0))
            k_conv_in = F.pad(k_conv_in, (self.causal_padding, 0))
        
        q_conv_out = self.conv_q(q_conv_in).transpose(1, 2)
        k_conv_out = self.conv_k(k_conv_in).transpose(1, 2)
        
        Q = q_conv_out.reshape(B, N, T_q, self.num_heads, self.d_k).transpose(2, 3)
        K = k_conv_out.reshape(B, N, T_k, self.num_heads, self.d_k).transpose(2, 3)
        V = self.linear_v(value).view(B, N, T_k, self.num_heads, self.d_k).transpose(2, 3)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(2).unsqueeze(3)
            scores = scores.masked_fill(mask, -1e9)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0) == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        x = torch.matmul(attn, V)
        x = x.transpose(2, 3).contiguous().view(B, N, T_q, self.d_model)

        return self.linear_out(x)

# ----------------------------------------------------------------------
# 模块 3: 动态空间 GNN
# ----------------------------------------------------------------------
class DynamicSpatialGNN(nn.Module):
    def __init__(self, d_model, num_nodes, edge_dim=4, hidden_edge=32, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.dropout = nn.Dropout(dropout)
        d_attn = d_model

        self.A_prior_fusion = nn.Linear(5, 1, bias=False)
        self.W_q = nn.Linear(d_model, d_attn, bias=False)
        self.W_k = nn.Linear(d_model, d_attn, bias=False)
        self.W_v = nn.Linear(d_model, d_attn, bias=False)

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * d_attn + edge_dim, hidden_edge),
            nn.ReLU(),
            nn.Linear(hidden_edge, 1),
        )

        self.Theta = nn.Linear(d_attn, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.phys_weight = nn.Parameter(torch.tensor(1.0))
        self.prior_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, edge_features, A_prior=None, entity_padding_mask=None):
        B, N, T, D = x.shape
        x_btnd = x.permute(0, 2, 1, 3).contiguous()
        Z = x_btnd.view(B * T, N, D)

        Q = self.W_q(Z)
        K = self.W_k(Z)
        V = self.W_v(Z)
        d_attn = Q.size(-1)

        content_logits = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(d_attn)

        Qi = Q.unsqueeze(2).expand(-1, -1, N, -1)
        Kj = K.unsqueeze(1).expand(-1, N, -1, -1)
        edge_input = torch.cat([Qi, Kj, edge_features], dim=-1)
        phys_logits = self.edge_mlp(edge_input).squeeze(-1)

        logits = content_logits + self.phys_weight * phys_logits

        if A_prior is not None:
            A_prior = self.A_prior_fusion(A_prior).squeeze(-1)
            if A_prior.dim() == 4:
                A_prior = A_prior.contiguous().view(B * T, N, N)
            A_prior = torch.nan_to_num(A_prior, nan=0.0)
            A_prior = torch.clamp(A_prior, min=0.0)
            prior_logits = torch.log(A_prior + 1e-6)
            logits = logits + self.prior_weight * prior_logits

        if entity_padding_mask is not None:
            if entity_padding_mask.dim() == 2:
                mask_bt = entity_padding_mask.unsqueeze(1).expand(B, T, N)
            else:
                mask_bt = entity_padding_mask
            mask_bt = mask_bt.contiguous().view(B * T, N)
            mask_row = mask_bt.unsqueeze(2).expand(B * T, N, N)
            mask_col = mask_bt.unsqueeze(1).expand(B * T, N, N)
            invalid = mask_row | mask_col
            logits = logits.masked_fill(invalid, -1e9)

        alpha = F.softmax(logits, dim=-1)
        spatial_features = torch.matmul(alpha, V)
        output_features = self.Theta(spatial_features)

        output_features = output_features.view(B, T, N, D)
        Z_reshaped = Z.view(B, T, N, D)
        out = self.norm(Z_reshaped + output_features)
        out = self.dropout(out)
        out = out.permute(0, 2, 1, 3).contiguous()

        if entity_padding_mask is not None:
            if entity_padding_mask.dim() == 2:
                mask_bn = entity_padding_mask.unsqueeze(-1).unsqueeze(-1)
            else:
                mask_bn = entity_padding_mask.permute(0, 2, 1).unsqueeze(-1)
            out = out.masked_fill(mask_bn.bool(), 0.0)

        return out

# ----------------------------------------------------------------------
# 模块 4: 嵌入层
# ----------------------------------------------------------------------
class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(TemporalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :, :x.size(2), :]
        return self.dropout(x)

class SpatialPositionalEncoding(nn.Module):
    def __init__(self, num_nodes, d_model, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(x + self.embedding.weight.unsqueeze(0).unsqueeze(2))

# ----------------------------------------------------------------------
# 模块 5: Encoder / Decoder 层
# ----------------------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, temporal_attn, spatial_gnn, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.temporal_attn = temporal_attn
        self.spatial_gnn = spatial_gnn
        self.sublayer_temporal = SublayerConnection(d_model, dropout)
        self.sublayer_spatial = SublayerConnection(d_model, dropout)

    def forward(self, x, temporal_mask, entity_mask, A_social_t=None, edge_features=None):
        B, N, T, D = x.shape
        C = A_social_t.shape[-1]
        E = edge_features.shape[-1]
        A_social_t = A_social_t.reshape(B*T, N, N, C)
        edge_features = edge_features.reshape(B*T, N, N, E)
        
        x = self.sublayer_temporal(x, lambda x: self.temporal_attn(
            x, x, x, key_padding_mask=temporal_mask))
        
        x = self.sublayer_spatial(x, lambda x: self.spatial_gnn(
            x, A_prior=A_social_t, edge_features=edge_features, entity_padding_mask=entity_mask))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, cross_attn, spatial_gnn, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.spatial_gnn = spatial_gnn
        self.sublayer_self_attn = SublayerConnection(d_model, dropout)
        self.sublayer_spatial_attn = SublayerConnection(d_model, dropout)
        self.sublayer_cross_attn = SublayerConnection(d_model, dropout)

    def forward(self, x, memory, temporal_mask_self, temporal_mask_cross, entity_mask, attn_mask_self, A_social_t=None, edge_features_t=None):
        B, N, L, D = x.shape
        C = A_social_t.shape[-1]
        E = edge_features_t.shape[-1]
        
        A_social_t_for_GNN = A_social_t.unsqueeze(1).repeat(1, L, 1, 1, 1).reshape(B * L, N, N, C)
        edge_features_t_for_GNN = edge_features_t.unsqueeze(1).repeat(1, L, 1, 1, 1).reshape(B * L, N, N, E)

        x = self.sublayer_self_attn(x, lambda x: self.self_attn(
            x, x, x, key_padding_mask=temporal_mask_self, attn_mask=attn_mask_self))
        
        x = self.sublayer_spatial_attn(x, lambda x: self.spatial_gnn(
            x, A_prior=A_social_t_for_GNN, edge_features=edge_features_t_for_GNN, entity_padding_mask=entity_mask))
        
        x = self.sublayer_cross_attn(x, lambda x: self.cross_attn(
            x, memory, memory, key_padding_mask=temporal_mask_cross))
        return x

# ----------------------------------------------------------------------
# 模块 6: 完整模型
# ----------------------------------------------------------------------
class Model(nn.Module):
    
    def __init__(self, args):
        super(Model, self).__init__()
        
        self.num_nodes = args.num_ships
        self.in_features = 7        
        self.out_features = 2       
        self.d_model = args.d_model
        self.num_heads = args.n_heads
        self.num_layers = args.e_layers
        self.dropout = args.dropout
        self.kernel_size = getattr(args, 'kernel_size', 3)
        self.pred_len = args.pred_len
        self.sampling_prob = getattr(args, 'sampling_prob', 0.7)

        # -- 核心模块 --
        c = copy.deepcopy
        
        temporal_attn = TrendAwareAttention(
            self.d_model, self.num_heads, self.kernel_size, mode='1d', dropout=self.dropout)
        
        temporal_attn_causal = TrendAwareAttention(
            self.d_model, self.num_heads, self.kernel_size, mode='causal', dropout=self.dropout)
        
        spatial_gnn = DynamicSpatialGNN(
            self.d_model, self.num_nodes, dropout=self.dropout)

        # -- 嵌入层 --
        self.src_input_proj = nn.Linear(self.in_features, self.d_model)
        self.trg_input_proj = nn.Linear(self.out_features, self.d_model)
        
        # -- [修改] 意图感知与环境嵌入 --
        # 1. 初始化意图感知模块
        self.intent_module = PhysicsIntentModule(temperature=0.1, allow_reverse=False)
        
        # 2. 意图向量(2维)的投影层
        self.intent_proj = nn.Linear(2, self.d_model)
        
        # 3. 航道 One-Hot (8维) 的投影层
        self.next_lane_proj = nn.Sequential(
                                nn.Linear(8, self.d_model),
                                nn.ReLU(),
                            )
        
        self.pos_encoder = TemporalPositionalEncoding(self.d_model, self.dropout)
        self.node_encoder = SpatialPositionalEncoding(self.num_nodes, self.d_model, self.dropout)

        # -- Encoder --
        encoder_layer = EncoderLayer(
            self.d_model, c(temporal_attn), c(spatial_gnn), self.dropout)
        self.encoder_layers = clones(encoder_layer, self.num_layers)
        self.encoder_norm = nn.LayerNorm(self.d_model)

        # -- Decoder --
        decoder_layer = DecoderLayer(
            self.d_model, c(temporal_attn_causal), c(temporal_attn), c(spatial_gnn), self.dropout)
        self.decoder_layers = clones(decoder_layer, self.num_layers)
        self.decoder_norm = nn.LayerNorm(self.d_model)

        self.output_proj = nn.Linear(self.d_model, self.out_features)
        
        self.criterion = get_loss_function(args.loss)


    def _compute_truth_deltas(self, x_enc, y_truth_abs):
        last_known_pos = x_enc[:, -1:, :, :2] 
        prev_future_pos = y_truth_abs[:, :-1, :, :2]
        all_previous_positions = torch.cat([last_known_pos, prev_future_pos], dim=1)
        y_truth_deltas = y_truth_abs[..., :2] - all_previous_positions
        return y_truth_deltas

    def forward(self, x_enc, y_truth_abs, mask_x, mask_y, A_social_t=None, edge_features=None):
        """
        Args:
            x_enc: [B, T_in, N, D_in]
        """
        device = x_enc.device

        # 1. 准备掩码
        mask_x_permuted = mask_x.permute(0, 2, 1) 
        mask_y_permuted = mask_y.permute(0, 2, 1) 
        entity_mask = mask_x.any(dim=1) 
        entity_padding_mask = ~entity_mask 
        temporal_padding_mask_enc = ~mask_x_permuted
        temporal_padding_mask_dec = ~mask_y_permuted
        attn_mask_self = subsequent_mask(self.pred_len, device)

        # 分割输入
        seq_x = x_enc[..., :7].to(device)  # [B, T_in, N, 7]
        next_lane_onehot = x_enc[..., 7:15].to(device).float()   # [B, T_in, N, 8]
        lane_dir_feats = x_enc[..., 15:].to(device).float()      # [B, T_in, N, 4]
        
        # 2. Encoder 前向
        x_enc_permuted = seq_x.permute(0, 2, 1, 3) 
        enc_in = self.src_input_proj(x_enc_permuted)
        
        # [关键修改] Encoder 端的意图增强
        if lane_dir_feats is not None:
            # 获取历史 COG (假设在最后两维: cos, sin)
            # x_enc_permuted shape: [B, N, T, 7]
            # 特征顺序: 0:2(pos), 2(speed), 3(course), 4(lane_s), 5(lane_d), 6(cog_cos), 7(cog_sin) ??
            # 根据你之前的 transform_features, 应该是在最后两维
            hist_cog = x_enc_permuted[..., -2:] 
            hist_lane_feats = lane_dir_feats.permute(0, 2, 1, 3) # [B, N, T, 4]
            
            # 计算历史意图向量
            hist_intent, _ = self.intent_module(hist_cog, hist_lane_feats)
            
            # 投影并叠加
            enc_in = enc_in + self.intent_proj(hist_intent)
            
        if next_lane_onehot is not None:
            next_lane_perm = next_lane_onehot.permute(0, 2, 1, 3)
            enc_in = enc_in + self.next_lane_proj(next_lane_perm)

        enc_in = self.node_encoder(enc_in) 
        enc_in = self.pos_encoder(enc_in)  

        memory = enc_in
        for layer in self.encoder_layers:
            memory = layer(
                memory, 
                temporal_mask=temporal_padding_mask_enc, 
                entity_mask=entity_padding_mask,
                A_social_t=A_social_t,
                edge_features=edge_features
            )
        memory = self.encoder_norm(memory)

        # 3. Decoder 前向 (自回归 + 动态意图交互)
        
        # 3.1 准备起始 Token
        y_input_abs = x_enc_permuted[:, :, -1:, :2] # [B, N, 1, 2]
        
        # 3.2 准备初始 COG (用于第一步的意图计算)
        # 假设最后一帧的历史 COG 是合理的初始猜测
        current_cog = hist_cog[:, :, -1:, :] # [B, N, 1, 2]
        
        outputs_deltas = [] 
        
        if self.training:
            y_truth_abs_permuted = y_truth_abs.permute(0, 2, 1, 3) 

        # 3.3 准备环境特征 (Encoder 最后一帧的特征，复制给 Decoder)
        # [B, T_in, N, 4] -> [B, N, 1, 4]
        static_lane_feats = lane_dir_feats[:, -1:, :, :].permute(0, 2, 1, 3) 
        static_next_lane = next_lane_onehot[:, -1:, :, :].permute(0, 2, 1, 3)
        static_next_lane_embed = self.next_lane_proj(static_next_lane)

        intent_vectors_list = []
        for t in range(self.pred_len):
            
            # (A) 嵌入 "绝对坐标"
            dec_in = self.trg_input_proj(y_input_abs) # [B, N, 1, D]
            
            # (B) [核心创新] 动态意图交互
            # 使用 "当前的 COG" 和 "环境特征" 计算意图
            # current_cog: [B, N, 1, 2]
            # static_lane_feats: [B, N, 1, 4]
            step_intent, _ = self.intent_module(current_cog, static_lane_feats)
            intent_vectors_list.append(step_intent)
            # (C) 叠加环境信息
            # 叠加意图向量
            dec_in = dec_in + self.intent_proj(step_intent)
            # 叠加 One-Hot
            dec_in = dec_in + static_next_lane_embed
            
            # (D) 加上位置编码
            dec_in = self.node_encoder(dec_in)
            dec_in = self.pos_encoder(dec_in)
            
            # (E) 准备掩码
            L = dec_in.size(2)
            temporal_mask_self = temporal_padding_mask_dec[:, :, :L]
            attn_mask_self_step = attn_mask_self[:, :L, :L]
            
            # (F) 运行 Decoder
            dec_out = dec_in
            A_social_t_last = A_social_t[:, -1, :, :]
            edge_features_t = edge_features[:, -1, :, :]
            
            for layer in self.decoder_layers:
                dec_out = layer(
                    dec_out, memory,
                    temporal_mask_self=temporal_mask_self,
                    temporal_mask_cross=temporal_padding_mask_enc,
                    entity_mask=entity_padding_mask,
                    attn_mask_self=attn_mask_self_step, 
                    A_social_t=A_social_t_last,
                    edge_features_t=edge_features_t
                )
            dec_out = self.decoder_norm(dec_out)
            
            # (G) 预测 "增量"
            pred_step_delta = self.output_proj(dec_out[:, :, -1:, :]) # [B, N, 1, 2]
            outputs_deltas.append(pred_step_delta)
            
            # (H) 预定采样逻辑
            use_truth = self.training and (torch.rand(1) < self.sampling_prob)
            
            if use_truth:
                next_abs_pos = y_truth_abs_permuted[:, :, t:t+1, :]
                
                # 如果使用了真值，我们需要更新 current_cog 为真值的方向
                # 简单计算：当前真值 - 上一步真值 (即真值 delta)
                # 注意：这里需要归一化得到 cos, sin
                true_delta = next_abs_pos - y_input_abs[:, :, -1:, :]
                current_cog = F.normalize(true_delta, dim=-1) # 更新 COG 用于下一步
            else:
                last_abs_pos = y_input_abs[:, :, -1:, :]
                next_abs_pos = last_abs_pos + pred_step_delta.detach()
                
                # 更新 current_cog 为预测的方向
                current_cog = F.normalize(pred_step_delta.detach(), dim=-1)

            # (I) 回填
            y_input_abs = torch.cat([y_input_abs, next_abs_pos], dim=2)

        output = torch.cat(outputs_deltas, dim=2) 
        output = output.permute(0, 2, 1, 3)

        # [新增] 处理意图向量集合
        # 1. 拼接: [B, N, T, 2]
        intent_vectors = torch.cat(intent_vectors_list, dim=2)
        # 2. 维度变换: [B, T, N, 2] (与 output 对齐)
        intent_vectors = intent_vectors.permute(0, 2, 1, 3)

        final_mask = mask_y.unsqueeze(-1).float()
        
        return output * final_mask, intent_vectors

    def loss(self, pred_deltas, y_truth_abs, x_enc, mask_y, intent_vectors):
        """
        Args:
            pred_deltas: [B, T, N, 2]
            y_truth_abs: [B, T, N, 2]
            x_enc: [B, T_in, N, D]
            mask_y: [B, T, N]
            intent_vectors: [B, T, N, 2]
        """
        y_truth_abs = y_truth_abs[..., :2]
        if mask_y.dtype == torch.float:
            mask_y_bool = mask_y.bool()
        else:
            mask_y_bool = mask_y

        # 0. 计算基础的增量 MSE Loss
        y_truth_deltas = self._compute_truth_deltas(x_enc, y_truth_abs)
        
        loss_delta = self.criterion(
            pred_deltas[mask_y_bool], 
            y_truth_deltas[mask_y_bool]
        )
        
        # 1. 计算绝对坐标轨迹
        pred_absolute = self.integrate(pred_deltas, x_enc)
        
        # 计算基础 diff
        diff = pred_absolute - y_truth_abs
        
        # -----------------------------------------------------------
        # [监控项] 原始 ADE (Average Displacement Error)
        # 这就是你原本的 loss_absolute，数值代表真实的物理均方误差
        # 不乘时间权重，仅用于日志监控
        # -----------------------------------------------------------
        # mse_raw = (diff ** 2).sum(dim=-1) # [B, T, N]
        num_valid = mask_y.sum() + 1e-6
        loss_absolute = self.criterion(
            pred_absolute[mask_y_bool],
            y_truth_abs[mask_y_bool]
        )

        # -----------------------------------------------------------
        # [优化项] 加权绝对 Loss (Weighted Absolute Loss)
        # 赋予序列末尾更大的权重，强迫模型"对齐终点"
        # -----------------------------------------------------------
        T = pred_absolute.shape[1]
        # 权重: [1.0, 1.4, 1.8, ..., 5.0]
        steps = torch.linspace(1.0, 5.0, steps=T, device=pred_absolute.device)
        time_weights = steps.view(1, T, 1, 1) # [1, T, 1, 1]
        
        # 均方误差 * 时间权重
        weighted_mse = (diff ** 2 * time_weights).sum(dim=-1) # [B, T, N]
        loss_weighted_abs = (weighted_mse * mask_y).sum() / num_valid
        
        # -----------------------------------------------------------
        # 4. 意图对齐损失 (Lane Alignment Loss)
        # -----------------------------------------------------------
        # 计算余弦相似度 (dim=-1 在 lat/lon 维度计算)
        cos_sim = F.cosine_similarity(pred_deltas, intent_vectors, dim=-1, eps=1e-6)
        loss_align_map = 1.0 - cos_sim
        loss_align = (loss_align_map * mask_y).sum() / num_valid

        # -----------------------------------------------------------
        # 5. 最终组合 (用于反向传播)
        # -----------------------------------------------------------
        # 注意：这里我们用 loss_weighted_abs 进入 total_loss，而不是 loss_ade
        total_loss = loss_delta + loss_weighted_abs + 2.0 * loss_align

        # 6. 计算方向一致性 Loss (仅监控)
        loss_direction = direction_loss(pred_absolute, y_truth_abs, y_mask=mask_y)
        
        # 返回值增加了 loss_ade
        return loss_delta, total_loss, loss_direction, loss_absolute

    def integrate(self, pred_deltas, x_enc_history):
        last_known_pos = x_enc_history[:, -1:, :, :2].to(pred_deltas.device)
        cumulative_deltas = torch.cumsum(pred_deltas, dim=1)
        outputs_absolute = last_known_pos + cumulative_deltas
        return outputs_absolute