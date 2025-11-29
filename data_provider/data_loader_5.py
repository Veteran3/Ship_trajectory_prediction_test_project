from shapely.geometry import LineString, Point
from shapely import distance, line_interpolate_point, line_locate_point
import shapely
import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
from functools import lru_cache

warnings.filterwarnings('ignore')
"""
优化版本 - 基于版本4
优化：
1. 使用 Shapely 2.0 向量化操作
2. 预计算航道几何信息
3. 使用 NumPy 向量化替代 Python 循环
4.  缓存航道查找结果
"""


# ===== 航道文件读取 =====
def load_lane_table(path: str):
    ext = os.path. splitext(path)[1]. lower()
    print(path)
    if ext in [". xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd. read_csv(path, encoding="utf-8")

    df = df. rename(columns={
        df.columns[0]: "lane_name",
        df. columns[1]: "lane_dir",
        df. columns[2]: "lane_role",
        df.columns[3]: "seq",
        df.columns[4]: "lon",
        df. columns[5]: "lat",
    })
    print("[Lane] load_lane_table OK, type(df) =", type(df))
    print(df.head())
    return df


# ===== 构建航道几何 =====
def build_lanes(df: pd.DataFrame):
    lanes = {}
    for (lane_name, lane_dir, lane_role), g in df.groupby(["lane_name", "lane_dir", "lane_role"]):
        if str(lane_dir). upper() in ("EW", "WE"):
            g = g.sort_values("lon")
        else:
            g = g.sort_values("lat")
        coords = list(zip(g["lon"]. values, g["lat"].values))
        if len(coords) < 2:
            continue
        line = LineString(coords)
        entry = lanes.setdefault(lane_name, {"dir": lane_dir})
        entry[lane_role] = line
    print("[Lane] build_lanes OK, lanes keys:", list(lanes.keys()))
    return lanes


def next_lanes_to_onehot(next_lanes, all_lanes=["EW1", "NS1", "NS2", "EW2", "WE1", "WE2", "SN1", "SN2"]):
    """
    将多个可能的下一个航道转换为 one-hot 编码。
    """
    onehot = np.zeros(len(all_lanes), dtype=np.float32)
    
    if isinstance(next_lanes, str):
        next_lanes = [next_lanes]
    
    for lane in next_lanes:
        if lane in all_lanes:
            idx = all_lanes. index(lane)
            onehot[idx] = 1
    
    return onehot


class LaneGeometryCache:
    """预计算并缓存航道几何信息，加速查找"""
    
    def __init__(self, lanes: dict, df_lanes: pd.DataFrame):
        self. lanes = lanes
        self.df_lanes = df_lanes
        self.lane_names = list(lanes. keys())
        self.num_lanes = len(self.lane_names)
        
        # 预提取所有中心线
        self.center_lines = []
        self.center_line_coords = []
        self.lane_name_to_idx = {}
        
        for idx, name in enumerate(self. lane_names):
            self.lane_name_to_idx[name] = idx
            center = lanes[name]. get("center", None)
            if center is not None:
                self.center_lines.append(center)
                self.center_line_coords. append(np.array(center. coords))
            else:
                self.center_lines. append(None)
                self.center_line_coords.append(None)
        
        # 预构建 next_lane 查找表
        self._build_next_lane_lookup()
        
        # 预计算航道序号映射
        self._build_seq_lookup()
    
    def _build_next_lane_lookup(self):
        """预构建 (lane_name, seq) -> next_lanes 查找表"""
        self.next_lane_lookup = {}
        for _, row in self.df_lanes.iterrows():
            key = (row['lane_name'], row['seq'])
            next_lanes_str = row. get('next_lane', '')
            if pd.isna(next_lanes_str) or next_lanes_str == '':
                next_lanes = []
            else:
                next_lanes = [l.strip(). strip("'\"") for l in str(next_lanes_str).strip("[]").split(',')]
            self.next_lane_lookup[key] = next_lanes
    
    def _build_seq_lookup(self):
        """预构建航道点坐标到序号的映射"""
        self.seq_coords = {}  # lane_name -> array of coords
        self.seq_values = {}  # lane_name -> array of seq values
        
        for lane_name in self.lane_names:
            lane_df = self.df_lanes[self.df_lanes['lane_name'] == lane_name]
            if not lane_df.empty:
                self.seq_coords[lane_name] = lane_df[['lon', 'lat']].values
                self.seq_values[lane_name] = lane_df['seq'].values
    
    def find_nearest_lanes_batch(self, lons: np.ndarray, lats: np. ndarray):
        """
        批量查找最近航道
        
        Args:
            lons: [N] 经度数组
            lats: [N] 纬度数组
        
        Returns:
            lane_indices: [N] 最近航道索引
            lane_names: list of lane names
            lane_entries: list of lane entries
        """
        N = len(lons)
        
        # 创建点数组 (使用 Shapely 2.0 向量化)
        points = shapely.points(lons, lats)
        
        # 计算到每条航道的距离
        min_dists = np.full(N, np.inf)
        best_lane_idx = np.zeros(N, dtype=np.int32)
        
        for idx, center in enumerate(self.center_lines):
            if center is None:
                continue
            # 向量化距离计算
            dists = shapely.distance(points, center)
            update_mask = dists < min_dists
            min_dists[update_mask] = dists[update_mask]
            best_lane_idx[update_mask] = idx
        
        lane_names = [self.lane_names[i] for i in best_lane_idx]
        lane_entries = [self.lanes[name] for name in lane_names]
        
        return best_lane_idx, lane_names, lane_entries, min_dists
    
    def compute_lane_features_batch(self, lons: np.ndarray, lats: np.ndarray, 
                                     lane_indices: np.ndarray, valid_mask: np. ndarray):
        """
        批量计算航道特征 (s_norm, d_signed)
        
        Args:
            lons, lats: [N] 坐标数组
            lane_indices: [N] 航道索引
            valid_mask: [N] 有效掩码
        
        Returns:
            s_norm: [N] 归一化弧长
            d_signed: [N] 带符号横向距离
        """
        N = len(lons)
        s_norm = np.zeros(N, dtype=np.float32)
        d_signed = np. zeros(N, dtype=np.float32)
        
        # 按航道分组处理
        for lane_idx in range(self.num_lanes):
            center = self.center_lines[lane_idx]
            if center is None:
                continue
            
            # 找到属于这条航道的点
            mask = (lane_indices == lane_idx) & valid_mask
            if not np.any(mask):
                continue
            
            # 提取坐标
            pts_lon = lons[mask]
            pts_lat = lats[mask]
            points = shapely.points(pts_lon, pts_lat)
            
            # 批量计算弧长
            s_values = shapely.line_locate_point(center, points)
            s_norm_batch = s_values / max(center.length, 1e-6)
            
            # 批量计算投影点
            proj_points = shapely.line_interpolate_point(center, s_values)
            
            # 批量计算距离
            d_values = shapely.distance(points, proj_points)
            
            # 计算符号 (需要向量化)
            # 获取投影点和前进方向点
            s2_values = np.minimum(s_values + 1.0, center.length)
            ahead_points = shapely.line_interpolate_point(center, s2_values)
            
            # 提取坐标
            proj_coords = shapely.get_coordinates(proj_points)
            ahead_coords = shapely.get_coordinates(ahead_points)
            
            # 计算方向向量和相对位置向量
            dx = ahead_coords[:, 0] - proj_coords[:, 0]
            dy = ahead_coords[:, 1] - proj_coords[:, 1]
            vx = pts_lon - proj_coords[:, 0]
            vy = pts_lat - proj_coords[:, 1]
            
            # 叉积符号
            sign = np.sign(-dy * vx + dx * vy)
            d_signed_batch = d_values * sign
            
            # 填回结果
            s_norm[mask] = s_norm_batch. astype(np.float32)
            d_signed[mask] = d_signed_batch.astype(np.float32)
        
        return s_norm, d_signed
    
    def compute_lane_dir_features_batch(self, lons: np.ndarray, lats: np.ndarray,
                                         lane_indices: np.ndarray, valid_mask: np. ndarray):
        """
        批量计算航道方向特征 (cos, sin)
        """
        N = len(lons)
        lane_cos = np.zeros(N, dtype=np.float32)
        lane_sin = np.zeros(N, dtype=np. float32)
        
        for lane_idx in range(self.num_lanes):
            center = self.center_lines[lane_idx]
            if center is None:
                continue
            
            mask = (lane_indices == lane_idx) & valid_mask
            if not np. any(mask):
                continue
            
            pts_lon = lons[mask]
            pts_lat = lats[mask]
            points = shapely.points(pts_lon, pts_lat)
            
            # 计算弧长
            s_values = shapely.line_locate_point(center, points)
            
            # 计算切线方向
            ds = 1.0
            s1 = np.maximum(0.0, s_values - ds)
            s2 = np.minimum(center.length, s_values + ds)
            
            p1 = shapely.line_interpolate_point(center, s1)
            p2 = shapely.line_interpolate_point(center, s2)
            
            p1_coords = shapely.get_coordinates(p1)
            p2_coords = shapely.get_coordinates(p2)
            
            dx = p2_coords[:, 0] - p1_coords[:, 0]
            dy = p2_coords[:, 1] - p1_coords[:, 1]
            norm = np.sqrt(dx**2 + dy**2)
            norm = np.maximum(norm, 1e-6)
            
            lane_cos[mask] = (dx / norm).astype(np.float32)
            lane_sin[mask] = (dy / norm).astype(np.float32)
        
        return lane_cos, lane_sin
    
    def get_seq_batch(self, lane_names: list, lons: np.ndarray, lats: np.ndarray):
        """批量获取航道序号"""
        N = len(lane_names)
        seqs = np.zeros(N, dtype=np.int32)
        
        for i, (lane_name, lon, lat) in enumerate(zip(lane_names, lons, lats)):
            if lane_name not in self.seq_coords:
                seqs[i] = -1
                continue
            
            coords = self.seq_coords[lane_name]
            dists = np.sqrt((coords[:, 0] - lon)**2 + (coords[:, 1] - lat)**2)
            seqs[i] = self.seq_values[lane_name][np.argmin(dists)]
        
        return seqs
    
    def get_next_lanes_batch(self, lane_names: list, seqs: np.ndarray, num_lanes: int = 8):
        """批量获取 next_lanes 的 one-hot 编码"""
        N = len(lane_names)
        onehots = np. zeros((N, num_lanes), dtype=np.float32)
        
        for i, (lane_name, seq) in enumerate(zip(lane_names, seqs)):
            key = (lane_name, int(seq))
            next_lanes = self.next_lane_lookup.get(key, [])
            onehots[i] = next_lanes_to_onehot(next_lanes)
        
        return onehots


class ShipTrajectoryDataset(Dataset):
    """
    船舶轨迹预测数据集 (优化版本)
    
    主要优化：
    1. 使用批量向量化操作替代逐点循环
    2.  预计算航道几何信息
    3. 缓存归一化参数
    """
    
    def __init__(
        self,
        root_path,
        data_path,
        flag='train',
        size=None,
        num_ships=17,
        num_features=4,
        scale=True,
        scale_type='standard',
        predict_position_only=True,
        lane_table_path=None
    ):
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self. num_ships = num_ships
        self. num_features = num_features
        self.scale = scale
        self.scale_type = scale_type
        self.predict_position_only = predict_position_only
        self.social_sigma = 1.0
        self.tcpa_threshold = 300.0
        self.dcpa_threshold = 1000.0

        # 航道几何特征
        self. lanes = None
        self.lane_cache = None
        if lane_table_path is not None:
            self.df_lanes = load_lane_table(lane_table_path)
            self.lanes = build_lanes(self. df_lanes)
            # 创建优化的航道缓存
            self.lane_cache = LaneGeometryCache(self.lanes, self.df_lanes)

        # 序列长度
        if size is None:
            self. seq_len = 8
            self.pred_len = 12
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        
        self.__read_data__()
    
    def __read_data__(self):
        """从npz文件读取数据"""
        full_path = os. path.join(self.root_path, self.data_path)
        print(f'Loading {self.flag} data from: {full_path}')
        
        data = np.load(full_path)
        
        self.data_x = data['X']
        self. data_y = data['y']
        self. mask_x = data['X_mask']
        self.mask_y = data['y_mask']
        self.ship_counts = data['ship_counts']
        self.global_ids = data['global_ids']
        
        print(f'{self.flag} dataset loaded:')
        print(f'  X shape: {self. data_x.shape}')
        print(f'  y shape: {self. data_y.shape}')
        print(f'  Number of samples: {len(self.data_x)}')
        
        # 验证数据维度
        assert self.data_x.shape[1] == self.seq_len
        assert self.data_y.shape[1] == self. pred_len
        assert self.data_x.shape[2] == self.num_ships
        assert self. data_y.shape[2] == self.num_ships
        assert self.data_x.shape[3] == self. num_features
        
        if self.scale:
            self.__normalize_data__()
    
    def _transform_features(self, x, mask):
        """将原始的 4维特征 [Lat, Lon, SOG, COG] 转换为 5维特征"""
        assert mask is not None, "Mask must be provided"
        
        feats_others = x[..., :3]
        cog_scaled = x[..., 3]

        if self.scale_type == 'standard':
            cog_phys = cog_scaled * self.std[3] + self.mean[3]
        elif self.scale_type == 'minmax':
            cog_phys = cog_scaled * (self.max_val[3] - self.min_val[3]) + self.min_val[3]
        
        cog_rad = np.deg2rad(cog_phys)
        cog_cos = np.cos(cog_rad)
        cog_sin = np.sin(cog_rad)

        if mask is not None and mask.dtype != bool:
            mask = mask. astype(bool)

        new_feats = np.concatenate([feats_others, cog_cos[..., np.newaxis], cog_sin[..., np.newaxis]], axis=-1)
        new_feats = new_feats * mask[..., np.newaxis]

        return new_feats. astype(np.float32)

    def __normalize_data__(self):
        """数据归一化"""
        if self.flag == 'train':
            self.mean = np. zeros(self.num_features)
            self. std = np.zeros(self.num_features)
            self.min_val = np.zeros(self.num_features)
            self.max_val = np.zeros(self.num_features)
            
            for feat_idx in range(self.num_features):
                valid_data = self.data_x[:, :, :, feat_idx][self.mask_x == 1]
                
                if self.scale_type == 'standard':
                    self.mean[feat_idx] = valid_data.mean()
                    self. std[feat_idx] = valid_data.std()
                elif self.scale_type == 'minmax':
                    self.min_val[feat_idx] = valid_data. min()
                    self.max_val[feat_idx] = valid_data. max()

            if self.scale_type == 'standard':
                self. std = np.where(self.std < 1e-8, 1.0, self.std)
            elif self. scale_type == 'minmax':
                span = self.max_val - self.min_val
                span = np.where(span < 1e-8, 1.0, span)
                self.max_val = self. min_val + span

            self.scaler_params = {
                'mean': self.mean,
                'std': self.std,
                'min_val': self.min_val,
                'max_val': self. max_val,
                'scale_type': self.scale_type
            }
            
            np.save(os.path.join(self.root_path, 'scaler_params.npy'), self.scaler_params)

            print(f'\nNormalization parameters ({self.scale_type}):')
            feature_names = ['Longitude', 'Latitude', 'SOG', 'COG']
            for i, name in enumerate(feature_names):
                if self.scale_type == 'standard':
                    print(f'  {name}: mean={self.mean[i]:.4f}, std={self.std[i]:.4f}')
                else:
                    print(f'  {name}: min={self.min_val[i]:.4f}, max={self. max_val[i]:.4f}')
            
            self._apply_normalization(self.data_x, mask=self.mask_x)
            self._apply_normalization(self.data_y, mask=self.mask_y)
        else:
            scaler_path = os.path.join(self.root_path, 'scaler_params.npy')
            if os.path.exists(scaler_path):
                self.scaler_params = np.load(scaler_path, allow_pickle=True).item()
                self.mean = self.scaler_params['mean']
                self.std = self.scaler_params['std']
                self.min_val = self.scaler_params['min_val']
                self.max_val = self.scaler_params['max_val']
                self.scale_type = self.scaler_params['scale_type']
                
                print(f'\nLoaded normalization parameters from: {scaler_path}')
                
                self._apply_normalization(self.data_x, mask=self.mask_x)
                self._apply_normalization(self.data_y, mask=self.mask_y)
            else:
                print(f'\nWarning: Scaler parameters not found at {scaler_path}')

    def _apply_normalization(self, data, mask):
        """应用归一化（向量化版本）"""
        B, T, N, Dp = data.shape
        F = min(self.num_features, Dp)
        m = mask.astype(np.bool_)
        
        if self.scale_type == 'standard':
            for f in range(F):
                data[..., f] = np.where(
                    m,
                    (data[..., f] - self.mean[f]) / (self.std[f] + 1e-8),
                    data[..., f]
                )
        elif self. scale_type == 'minmax':
            for f in range(F):
                data[..., f] = np.where(
                    m,
                    (data[..., f] - self.min_val[f]) / (self.max_val[f] - self.min_val[f] + 1e-8),
                    data[..., f]
                )
    
    def _inverse_normalize_coords(self, x_data):
        """反归一化坐标（仅前两维）- 向量化版本"""
        D = x_data. shape[-1]
        x_phys = x_data. copy()
        
        if self.scale_type == 'standard':
            for f in range(min(D, 2)):  # 只反归一化经纬度
                x_phys[..., f] = x_phys[..., f] * self.std[f] + self.mean[f]
        elif self.scale_type == 'minmax':
            for f in range(min(D, 2)):
                x_phys[..., f] = x_phys[..., f] * (self.max_val[f] - self.min_val[f]) + self.min_val[f]
        
        return x_phys
    
    def inverse_transform(self, data):
        """反归一化"""
        if not self.scale:
            return data
        
        data_copy = data.copy()
        num_features_to_inverse = data. shape[-1]
        
        for feat_idx in range(num_features_to_inverse):
            if self.scale_type == 'standard':
                data_copy[..., feat_idx] = data_copy[..., feat_idx] * self.std[feat_idx] + self.mean[feat_idx]
            elif self.scale_type == 'minmax':
                data_copy[..., feat_idx] = data_copy[..., feat_idx] * (self.max_val[feat_idx] - self.min_val[feat_idx]) + self. min_val[feat_idx]
        
        return data_copy
    
    def save_scaler_params(self, save_path=None):
        """保存归一化参数"""
        if self.flag == 'train' and self.scale:
            if save_path is None:
                save_path = os.path.join(self. root_path, 'scaler_params.npy')
            np.save(save_path, self.scaler_params)
            print(f'Scaler parameters saved to: {save_path}')
    
    def __getitem__(self, index):
        """获取单个样本（优化版本）"""
        seq_x = self.data_x[index]  # [T_in, N, D]
        seq_y = self.data_y[index]  # [T_out, N, D]
        seq_x_mask = self. mask_x[index]  # [T_in, N]
        seq_y_mask = self. mask_y[index]  # [T_out, N]
        ship_count = self.ship_counts[index]
        global_id = self.global_ids[index]
        
        T_in, N, D = seq_x.shape
        
        # 计算社会力矩阵
        A_social = self._build_semantic_social_fusion_matrix(seq_x)
        edge_features = self._build_edge_features(seq_x, mask=seq_x_mask)

        # ===== 航道特征（优化版本）=====
        if self.lane_cache is not None:
            lane_feats, lane_dir_feats, next_lanes_feats = self._build_lane_features_optimized(
                seq_x, seq_x_mask, index
            )
        else:
            lane_feats = np.zeros((T_in, N, 2), dtype=np.float32)
            lane_dir_feats = np.zeros((T_in, N, 2), dtype=np.float32)
            next_lanes_feats = np.zeros((T_in, N, 8), dtype=np. float32)
        
        # 转换特征
        seq_x = self._transform_features(seq_x, seq_x_mask)
        seq_y = self._transform_features(seq_y, seq_y_mask)
        
        # 拼接航道特征
        seq_x_final = np.concatenate([seq_x, lane_feats, next_lanes_feats, lane_dir_feats], axis=-1)

        return seq_x_final, seq_y, seq_x_mask, seq_y_mask, ship_count, global_id, A_social, edge_features
    
    def _build_lane_features_optimized(self, seq_x, mask, index):
        """
        优化的航道特征计算 - 使用批量向量化操作
        
        返回:
            lane_feats: [T, N, 2] (s_norm, d_signed)
            lane_dir_feats: [T, N, 2] (cos, sin)
            next_lanes_feats: [T, N, 8] (one-hot)
        """
        T, N, D = seq_x.shape
        
        # 反归一化获取物理坐标
        x_phys = self._inverse_normalize_coords(seq_x)
        lons_flat = x_phys[..., 0]. flatten()  # [T*N]
        lats_flat = x_phys[..., 1].flatten()  # [T*N]
        mask_flat = mask. astype(bool).flatten()  # [T*N]
        
        # 批量查找最近航道
        lane_indices, lane_names, lane_entries, _ = self.lane_cache.find_nearest_lanes_batch(
            lons_flat, lats_flat
        )
        
        # 批量计算 s_norm, d_signed
        s_norm_flat, d_signed_flat = self. lane_cache.compute_lane_features_batch(
            lons_flat, lats_flat, lane_indices, mask_flat
        )
        
        # 批量计算航道方向
        lane_cos_flat, lane_sin_flat = self.lane_cache.compute_lane_dir_features_batch(
            lons_flat, lats_flat, lane_indices, mask_flat
        )
        
        # 批量获取序号和 next_lanes
        seqs_flat = self.lane_cache. get_seq_batch(lane_names, lons_flat, lats_flat)
        next_lanes_flat = self.lane_cache.get_next_lanes_batch(lane_names, seqs_flat)
        
        # reshape 回 [T, N, ...]
        lane_feats = np. stack([
            s_norm_flat.reshape(T, N),
            d_signed_flat.reshape(T, N)
        ], axis=-1)
        
        lane_dir_feats = np.stack([
            lane_cos_flat.reshape(T, N),
            lane_sin_flat.reshape(T, N)
        ], axis=-1)
        
        next_lanes_feats = next_lanes_flat. reshape(T, N, -1)
        
        return lane_feats, lane_dir_feats, next_lanes_feats
    
    def __len__(self):
        return len(self.data_x)

    def _build_social_graph(self, x_data, mask=None):
        """构建社交图（基于船舶之间的距离）- 向量化版本"""
        T_seq, N, D_in = x_data. shape
        pos_data = x_data[:, :, :2]
        speed_data = x_data[:, :, 2]
        heading_data = x_data[:, :, 3]

        heading_rad = np.deg2rad(heading_data)
        v_x = speed_data * np.sin(heading_rad)
        v_y = speed_data * np.cos(heading_rad)
        vel_vectors = np.stack((v_x, v_y), axis=-1)

        pos_i = pos_data[:, :, np.newaxis, :]
        vel_i = vel_vectors[:, :, np.newaxis, :]
        pos_j = pos_data[:, np.newaxis, :, :]
        vel_j = vel_vectors[:, np.newaxis, :, :]

        dist_matrix = np.linalg.norm(pos_i - pos_j, axis=-1) + 1e-8
        vel_diff_sq = np.sum((vel_i - vel_j)**2, axis=-1)

        vel_term = np.exp(-vel_diff_sq / (2 * (self.social_sigma ** 2)))
        dist_term = 1.0 / (dist_matrix + 1e-9)

        A_social = vel_term * dist_term

        if mask is not None:
            m = mask.astype(bool) if mask. dtype != bool else mask
            mask_i = m[:, :, np.newaxis]
            mask_j = m[:, np.newaxis, :]
            combined_mask = mask_i & mask_j
            A_social = A_social * combined_mask

        diag_idx = np.arange(N)
        A_social[:, diag_idx, diag_idx] = 0.0

        return A_social

    def _build_semantic_social_fusion_matrix(self, x_data, mask=None):
        """构建语义社会力矩阵 - 优化向量化版本"""
        T, N, D = x_data.shape
        
        x_phys = x_data * self.scaler_params['std'][:D] + self.scaler_params['mean'][:D]
        
        pos_phys = x_phys[..., :2]
        speed_phys = x_phys[..., 2]
        course_phys = x_phys[..., 3]

        course_rad = np.deg2rad(90 - course_phys)
        vx = speed_phys * np.cos(course_rad)
        vy = speed_phys * np.sin(course_rad)
        vel_phys = np.stack([vx, vy], axis=-1)

        pos_i = pos_phys[:, :, np.newaxis, :]
        pos_j = pos_phys[:, np.newaxis, :, :]
        vel_i = vel_phys[:, :, np.newaxis, :]
        vel_j = vel_phys[:, np.newaxis, :, :]
        course_i = course_phys[:, :, np.newaxis]

        delta_p = pos_j - pos_i
        dist = np.linalg.norm(delta_p, axis=-1)
        
        delta_v = vel_j - vel_i
        rel_speed_sq = np.sum(np.square(delta_v), axis=-1)
        
        angle_vec = np.rad2deg(np.arctan2(delta_p[..., 1], delta_p[..., 0]))
        heading_math = 90 - course_i
        beta = heading_math - angle_vec
        beta = (beta + 180) % 360 - 180
        
        mask_head_on = (np.abs(beta) <= 15)
        mask_cross_starboard = (beta > 15) & (beta <= 112.5)
        mask_cross_port = (beta >= -112.5) & (beta < -15)
        mask_overtaking = (np.abs(beta) > 112.5)

        vel_term = np. exp(-rel_speed_sq / (2 * self.social_sigma**2))
        dist_term = 1.0 / (dist + 1e-9)
        base_weight = dist_term * vel_term

        A_anchor = base_weight[..., np.newaxis]
        
        A_rules = np.stack([
            base_weight * mask_head_on,
            base_weight * mask_cross_starboard,
            base_weight * mask_cross_port,
            base_weight * mask_overtaking
        ], axis=-1)
        
        if mask is not None:
            m = mask.astype(bool) if mask.dtype != bool else mask
            mask_i = m[:, :, np.newaxis, np.newaxis]
            mask_j = m[:, np. newaxis, :, np.newaxis]
            combined_mask = mask_i & mask_j
            A_rules = A_rules * combined_mask

        A_final = np.concatenate([A_anchor, A_rules], axis=-1). astype(np. float32)
        
        diag_idx = np.arange(N)
        A_final[:, diag_idx, diag_idx, :] = 0.0
        
        return A_final
        
    def _build_edge_features(self, x_data, mask=None):
        """构建边特征 - 向量化版本"""
        T, N, D = x_data.shape
        
        x_phys = x_data * self. scaler_params['std'][:D] + self.scaler_params['mean'][:D]
        pos_phys = x_phys[..., :2]
        speed_phys = x_phys[..., 2]
        course_phys = x_phys[..., 3]

        course_rad = np.deg2rad(90 - course_phys)
        vx = speed_phys * np. cos(course_rad)
        vy = speed_phys * np.sin(course_rad)
        vel_phys = np.stack([vx, vy], axis=-1)

        pos_i = pos_phys[:, :, np.newaxis, :]
        pos_j = pos_phys[:, np. newaxis, :, :]
        vel_i = vel_phys[:, :, np.newaxis, :]
        vel_j = vel_phys[:, np.newaxis, :, :]
        course_i = course_phys[:, :, np. newaxis]

        delta_p = pos_j - pos_i
        dist = np.linalg.norm(delta_p, axis=-1)
        inv_dist = 1.0 / (dist + 1.0)
        
        delta_v = vel_j - vel_i
        rel_speed = np.linalg.norm(delta_v, axis=-1)
        rel_speed = np.log1p(rel_speed)
        
        angle_vec = np.rad2deg(np.arctan2(delta_p[..., 1], delta_p[..., 0]))
        heading_math = 90 - course_i
        beta_deg = heading_math - angle_vec
        beta_rad = np.deg2rad(beta_deg)
        
        cos_beta = np. cos(beta_rad)
        sin_beta = np.sin(beta_rad)
        
        edge_features = np. stack([inv_dist, rel_speed, cos_beta, sin_beta], axis=-1)
        
        if mask is not None:
            m = mask.astype(bool) if mask. dtype != bool else mask
            mask_i = m[:, :, np.newaxis, np.newaxis]
            mask_j = m[:, np.newaxis, :, np.newaxis]
            combined_mask = mask_i & mask_j
            edge_features = edge_features * combined_mask
            
        return edge_features. astype(np. float32)


def ship_collate_fn(batch):
    """自定义 collate 函数"""
    seq_x_list, seq_y_list, mask_x_list, mask_y_list, count_list, id_list, A_social_list, edge_feat_list = zip(*batch)
    
    seq_x = torch.from_numpy(np. array(seq_x_list)). float()
    seq_y = torch. from_numpy(np.array(seq_y_list)).float()
    mask_x = torch.from_numpy(np. array(mask_x_list)).bool()
    mask_y = torch.from_numpy(np. array(mask_y_list)).bool()
    ship_count = torch. from_numpy(np.array(count_list)).long()
    global_id = torch.from_numpy(np. array(id_list)).long()
    A_social = torch. from_numpy(np.array(A_social_list)).float()
    edge_features = torch.from_numpy(np. array(edge_feat_list)).float()
    
    return seq_x, seq_y, mask_x, mask_y, ship_count, global_id, A_social, edge_features


class ShipTrajectoryDataLoader:
    """数据加载器工厂类"""
    
    def __init__(self, args):
        self.args = args
    
    def get_data_loader(self, flag):
        """获取数据加载器"""
        data_dict = {
            'train': 'train. npz',
            'val': 'val.npz',
            'test': 'test.npz'
        }
        
        shuffle_flag = (flag == 'train')
        drop_last = (flag == 'train')
        
        dataset = ShipTrajectoryDataset(
            root_path=self.args.root_path,
            data_path=data_dict[flag],
            flag=flag,
            size=[self.args.seq_len, self. args.pred_len],
            num_ships=self. args.num_ships,
            num_features=self.args. num_features,
            scale=self. args.scale,
            scale_type=self. args.scale_type,
            predict_position_only=self.args. predict_position_only,
            lane_table_path=getattr(self.args, 'lane_table_path', None)
        )
        
        if flag == 'train':
            dataset.save_scaler_params()
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args.num_workers,
            drop_last=drop_last,
            pin_memory=True,
            collate_fn=ship_collate_fn
        )
        
        return dataset, dataloader


if __name__ == '__main__':
    """测试数据加载"""
    
    class Args:
        root_path = './data/'
        seq_len = 8
        pred_len = 12
        num_ships = 17
        num_features = 4
        batch_size = 32
        num_workers = 4
        scale = True
        scale_type = 'standard'
        predict_position_only = True
        lane_table_path = None  # 设置为你的航道表路径
    
    args = Args()
    
    loader_factory = ShipTrajectoryDataLoader(args)
    
    print('=' * 50)
    print('Testing train dataset...')
    print('=' * 50)
    train_dataset, train_loader = loader_factory.get_data_loader('train')
    
    import time
    start = time.time()
    for i, batch in enumerate(train_loader):
        if i >= 10:
            break
    print(f'Time for 10 batches: {time.time() - start:.2f}s')
    
    print('\nAll tests passed!')