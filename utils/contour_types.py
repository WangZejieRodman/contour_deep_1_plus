"""
Contour Context Loop Closure Detection - Basic Data Structures
基础数据结构和配置类
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import yaml

# 常量定义
BITS_PER_LAYER = 20 #BCI分bin数  1 3 5 10 15 20 30 40

# 10层配置时
# DIST_BIN_LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 用于生成距离键和形成星座的层
# LAYER_AREA_WEIGHTS = [0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1]  # 计算归一化"使用区域百分比"时每层的权重

# 5层配置时
# DIST_BIN_LAYERS = [0, 1, 2, 3, 4]  # 所有层级
# LAYER_AREA_WEIGHTS = [0.1, 0.2, 0.4, 0.2, 0.1]  # 5个权重

# 15层配置时
# DIST_BIN_LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # 所有层级
# LAYER_AREA_WEIGHTS = [0.03, 0.03, 0.05, 0.05, 0.07, 0.07, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.03]  # 15个权重，中间层级权重更高

# 8层配置时
DIST_BIN_LAYERS = [0, 1, 2, 3, 4, 5, 6, 7]  # 所有层级
LAYER_AREA_WEIGHTS = [0.08, 0.1, 0.12, 0.15, 0.2, 0.15, 0.12, 0.08]  # 8个权重

NUM_BIN_KEY_LAYER = len(DIST_BIN_LAYERS)
RET_KEY_DIM = 10  # 检索键维度

@dataclass
class ContourViewStatConfig:
    """轮廓视图统计配置"""
    min_cell_cov: int = 4
    point_sigma: float = 0.01
    com_bias_thres: float = 0.5

@dataclass
class ContourSimThresConfig:
    """轮廓相似性阈值配置"""
    ta_cell_cnt: float = 20.0      # 放宽到20 (原来6.0)
    tp_cell_cnt: float = 0.5       # 放宽到50% (原来20%)
    tp_eigval: float = 0.5         # 放宽到50% (原来20%)
    ta_h_bar: float = 1.0          # 放宽到1.0 (原来0.3)
    ta_rcom: float = 2.0           # 放宽到2.0 (原来0.4)
    tp_rcom: float = 0.8           # 放宽到80% (原来25%)

@dataclass
class TreeBucketConfig:
    """树桶配置"""
    max_elapse: float = 25.0
    min_elapse: float = 15.0

@dataclass
class ContourManagerConfig:
    """轮廓管理器配置"""
    lv_grads: List[float] = field(default_factory=lambda: [0.0, 1.25, 2.5, 3.75, 5.0])
    reso_row: float = 1.0
    reso_col: float = 1.0
    n_row: int = 150
    n_col: int = 150
    lidar_height: float = 2.0
    blind_sq: float = 9.0
    min_cont_key_cnt: int = 9
    min_cont_cell_cnt: int = 3
    piv_firsts: int = 6
    dist_firsts: int = 10
    roi_radius: float = 10.0
    use_vertical_complexity: bool = True # 垂直结构复杂度开关
    neighbor_layer_range: int = 7  # 8层配置时，邻居搜索的层级范围（±N层）0=仅本层, 1=本层±1, 2=本层±2, ..., 7=本层±7
    angular_consistency_threshold: float = np.pi / 16  # 角度一致性阈值（弧度）默认 π/16 ≈ 0.196 rad ≈ 11.25°


@dataclass
class ContourDBConfig:
    """轮廓数据库配置"""
    nnk: int = 50
    max_fine_opt: int = 10
    q_levels: List[int] = field(default_factory=lambda: [0, 1, 2, 3])  # 改为全部10层
    cont_sim_cfg: ContourSimThresConfig = field(default_factory=ContourSimThresConfig)
    tb_cfg: TreeBucketConfig = field(default_factory=TreeBucketConfig)

@dataclass
class GMMOptConfig:
    """GMM优化配置"""
    min_area_perc: float = 0.95
    levels: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    cov_dilate_scale: float = 2.0

class PredictionOutcome(Enum):
    """预测结果类型"""
    TP = 0  # True Positive
    FP = 1  # False Positive
    TN = 2  # True Negative
    FN = 3  # False negative

@dataclass
class ScoreConstellSim:
    """星座相似性分数"""
    i_ovlp_sum: int = 0
    i_ovlp_max_one: int = 0
    i_in_ang_rng: int = 0

    def overall(self) -> int:
        return self.i_in_ang_rng

    def cnt(self) -> int:
        return self.i_in_ang_rng

    def strict_smaller(self, other: 'ScoreConstellSim') -> bool:
        return (self.i_ovlp_sum < other.i_ovlp_sum and
                self.i_ovlp_max_one < other.i_ovlp_max_one and
                self.i_in_ang_rng < other.i_in_ang_rng)

@dataclass
class ScorePairwiseSim:
    """成对相似性分数"""
    i_indiv_sim: int = 0
    i_orie_sim: int = 0

    def overall(self) -> int:
        return self.i_orie_sim

    def cnt(self) -> int:
        return self.i_orie_sim

    def strict_smaller(self, other: 'ScorePairwiseSim') -> bool:
        return (self.i_indiv_sim < other.i_indiv_sim and
                self.i_orie_sim < other.i_orie_sim)

@dataclass
class ScorePostProc:
    """后处理分数"""
    correlation: float = 0.0
    area_perc: float = 0.0
    neg_est_dist: float = 0.0  # 负距离（因为越大越好）

    def overall(self) -> float:
        return self.correlation

    def strict_smaller(self, other: 'ScorePostProc') -> bool:
        return (self.correlation < other.correlation and
                self.area_perc < other.area_perc and
                self.neg_est_dist < other.neg_est_dist)

@dataclass
class CandidateScoreEnsemble:
    """候选分数集合"""
    sim_constell: ScoreConstellSim = field(default_factory=ScoreConstellSim)
    sim_pair: ScorePairwiseSim = field(default_factory=ScorePairwiseSim)
    sim_post: ScorePostProc = field(default_factory=ScorePostProc)

@dataclass
class ConstellationPair:
    """星座对"""
    level: int
    seq_src: int
    seq_tgt: int

    def __lt__(self, other):
        return (self.level, self.seq_src, self.seq_tgt) < (other.level, other.seq_src, other.seq_tgt)

    def __eq__(self, other):
        return (self.level == other.level and
                self.seq_src == other.seq_src and
                self.seq_tgt == other.seq_tgt)

    def __hash__(self):
        return hash((self.level, self.seq_src, self.seq_tgt))

@dataclass
class RelativePoint:
    """BCI中的相对点"""
    level: int# 邻居轮廓所在层级
    seq: int# 邻居轮廓序号
    bit_pos: int# 在距离位串中的位置
    r: float# 到中心轮廓的距离
    theta: float# 相对于中心轮廓的角度

@dataclass
class DistSimPair:
    """距离相似性对"""
    level: int
    seq_src: int
    seq_tgt: int
    orie_diff: float


class BCI:
    """二进制星座标识"""

    def __init__(self, seq: int, level: int):
        self.piv_seq = seq# 中心轮廓序号
        self.level = level# 所属层级
        self.dist_bin = np.zeros(BITS_PER_LAYER * NUM_BIN_KEY_LAYER, dtype=bool)# 距离二进制位串 (20位×20层=400位)
        self.nei_pts: List[RelativePoint] = []# 邻居轮廓详细信息列表
        self.nei_idx_segs: List[int] = []# 按距离分组的索引段

    @staticmethod
    def check_constell_sim(src: 'BCI', tgt: 'BCI', lb: 'ScoreConstellSim',
                           constell_res: List['ConstellationPair'],
                           angular_threshold: float = np.pi / 16) -> 'ScoreConstellSim':
        """
        检查两个BCI的星座相似性

        Args:
            src: 源BCI
            tgt: 目标BCI
            lb: 下界阈值
            constell_res: 输出的星座对结果
            angular_threshold: 角度一致性阈值（弧度）

        Returns:
            相似性分数
        """
        assert src.level == tgt.level, "BCI层级必须相同"

        from contour_types import ScoreConstellSim, ConstellationPair, DistSimPair

        # ✅ 统计：总调用次数
        global CONSTELL_CHECK_STATS
        CONSTELL_CHECK_STATS.total_calls += 1

        # 1. 计算位重叠
        and1 = src.dist_bin & tgt.dist_bin
        # 左移和右移操作（模拟C++中的位移）
        src_left = np.roll(src.dist_bin, -1)  # 左移1位
        src_right = np.roll(src.dist_bin, 1)  # 右移1位

        and2 = src_left & tgt.dist_bin
        and3 = src_right & tgt.dist_bin

        ovlp1 = np.sum(and1)
        ovlp2 = np.sum(and2)
        ovlp3 = np.sum(and3)

        ovlp_sum = ovlp1 + ovlp2 + ovlp3
        max_one = max(ovlp1, ovlp2, ovlp3)

        ret = ScoreConstellSim()
        ret.i_ovlp_sum = ovlp_sum
        ret.i_ovlp_max_one = max_one

        # 如果重叠不足，直接返回
        if ovlp_sum < lb.i_ovlp_sum or max_one < lb.i_ovlp_max_one:
            # ✅ 统计：位重叠过滤
            CONSTELL_CHECK_STATS.filtered_by_overlap += 1
            return ret

        # 2. 角度一致性检查 - 生成潜在配对
        potential_pairs: List[DistSimPair] = []

        # 遍历目标的相邻点段
        p11 = 0
        for p2 in range(len(tgt.nei_idx_segs) - 1):
            tgt_bit_pos = tgt.nei_pts[tgt.nei_idx_segs[p2]].bit_pos

            # 找到源中匹配的位段范围
            while (p11 < len(src.nei_idx_segs) - 1 and
                   src.nei_pts[src.nei_idx_segs[p11]].bit_pos < tgt_bit_pos - 1):
                p11 += 1

            p12 = p11
            while (p12 < len(src.nei_idx_segs) - 1 and
                   src.nei_pts[src.nei_idx_segs[p12]].bit_pos <= tgt_bit_pos + 1):
                p12 += 1

            # 生成潜在配对
            for i in range(tgt.nei_idx_segs[p2], tgt.nei_idx_segs[p2 + 1]):
                for j in range(src.nei_idx_segs[p11], src.nei_idx_segs[p12]):
                    if j >= len(src.nei_pts) or i >= len(tgt.nei_pts):
                        continue

                    rp1 = src.nei_pts[j]
                    rp2 = tgt.nei_pts[i]

                    if rp1.level == rp2.level and abs(rp1.bit_pos - rp2.bit_pos) <= 1:
                        orie_diff = rp2.theta - rp1.theta
                        potential_pairs.append(DistSimPair(rp1.level, rp1.seq, rp2.seq, orie_diff))

                        # ✅ 统计：收集角度差异（在归一化之前先记录）
                        # 注意：这里暂不记录，等归一化后再记录

        # 3. 规范化角度差到[-π, π]并收集统计
        for pair in potential_pairs:
            pair.orie_diff = clamp_angle(pair.orie_diff)
            # ✅ 统计：收集角度差异（归一化后）
            angle_deg = abs(np.degrees(pair.orie_diff))
            CONSTELL_CHECK_STATS.angle_diffs.append(angle_deg)

        # 按角度差排序
        potential_pairs.sort(key=lambda x: x.orie_diff)

        # 找到最长的角度一致范围
        angular_range = angular_threshold  # 使用传入的角度阈值
        longest_in_range_beg = 0
        longest_in_range = 1
        pot_sz = len(potential_pairs)

        if pot_sz == 0:
            ret.i_in_ang_rng = 0
            # ✅ 统计：配对数为0（算作角度过滤）
            CONSTELL_CHECK_STATS.filtered_by_angle += 1
            return ret

        p1 = 0
        p2 = 0

        while p1 < pot_sz:
            # 计算角度差，考虑周期性
            angle_diff = potential_pairs[p2 % pot_sz].orie_diff - potential_pairs[p1].orie_diff
            if p2 >= pot_sz:
                angle_diff += 2 * np.pi

            if angle_diff > angular_range:
                p1 += 1
            else:
                current_range = p2 - p1 + 1
                if current_range > longest_in_range:
                    longest_in_range = current_range
                    longest_in_range_beg = p1
                p2 += 1

        ret.i_in_ang_rng = longest_in_range

        # 如果角度范围内的对数不足，返回
        if longest_in_range < lb.i_in_ang_rng:
            # ✅ 统计：角度一致性过滤
            CONSTELL_CHECK_STATS.filtered_by_angle += 1
            return ret

        # ✅ 统计：通过所有检查
        CONSTELL_CHECK_STATS.passed += 1

        # 4. 构建结果星座对
        constell_res.clear()

        # 添加在角度范围内的对
        for i in range(longest_in_range_beg, longest_in_range_beg + longest_in_range):
            idx = i % pot_sz
            if idx < len(potential_pairs):
                pair = potential_pairs[idx]
                constell_res.append(ConstellationPair(pair.level, pair.seq_src, pair.seq_tgt))

        # 添加锚点对
        constell_res.append(ConstellationPair(src.level, src.piv_seq, tgt.piv_seq))

        # 为了人类可读性进行排序
        constell_res.sort(key=lambda x: (x.level, x.seq_src))

        return ret

@dataclass
class RunningStatRecorder:
    """运行统计记录器"""
    cell_cnt: int = 0
    cell_pos_sum: np.ndarray = field(default_factory=lambda: np.zeros(2))
    cell_pos_tss: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    cell_vol3: float = 0.0
    cell_vol3_torq: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def running_stats(self, curr_row: int, curr_col: int, height: float):
        """添加统计数据"""
        self.cell_cnt += 1 #像素数量
        v_rc = np.array([curr_row, curr_col], dtype=float)
        self.cell_pos_sum += v_rc #位置和 [Σx, Σy]
        self.cell_pos_tss += np.outer(v_rc, v_rc) #位置平方和 [[Σx², Σxy], [Σxy, Σy²]]
        self.cell_vol3 += height #高度总和 Σh
        self.cell_vol3_torq += height * v_rc #高度加权位置 [Σ(h·x), Σ(h·y)]

def load_config_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """从YAML文件加载配置"""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def clamp_angle(ang: float) -> float:
    """将角度限制在[-π, π)范围内"""
    return ang - np.floor((ang + np.pi) / (2 * np.pi)) * 2 * np.pi

def diff_perc(num1: float, num2: float, perc: float) -> bool:
    """检查两个数的百分比差异是否超过阈值"""
    if max(num1, num2) == 0:
        return num1 != num2
    diff_ratio = abs((num1 - num2) / max(num1, num2))
    result = diff_ratio > perc
    return result

def diff_delt(num1: float, num2: float, delta: float) -> bool:
    """检查两个数的绝对差异是否超过阈值"""
    abs_diff = abs(num1 - num2)
    result = abs_diff > delta
    return result

def gauss_pdf(x: float, mean: float, sd: float) -> float:
    """高斯概率密度函数"""
    return np.exp(-0.5 * ((x - mean) / sd) ** 2) / np.sqrt(2 * np.pi * sd * sd)


class ConstellationCheckStats:
    """星座相似性检查统计"""

    def __init__(self):
        self.total_calls = 0  # 总调用次数
        self.filtered_by_overlap = 0  # 位重叠过滤次数
        self.filtered_by_angle = 0  # 角度一致性过滤次数
        self.passed = 0  # 通过次数
        self.angle_diffs = []  # 角度差异列表（度）

    def reset(self):
        """重置统计"""
        self.total_calls = 0
        self.filtered_by_overlap = 0
        self.filtered_by_angle = 0
        self.passed = 0
        self.angle_diffs = []

    def get_angle_distribution(self):
        """获取角度差异分布"""
        if not self.angle_diffs:
            return {}

        import numpy as np
        diffs = np.array(self.angle_diffs)

        return {
            'total_pairs': len(diffs),
            'mean': np.mean(diffs),
            'std': np.std(diffs),
            'median': np.median(diffs),
            'less_1deg': np.sum(diffs < 1.0) / len(diffs) * 100,
            'less_3deg': np.sum(diffs < 3.0) / len(diffs) * 100,
            'less_5deg': np.sum(diffs < 5.0) / len(diffs) * 100,
            'less_10deg': np.sum(diffs < 10.0) / len(diffs) * 100,
            'less_20deg': np.sum(diffs < 20.0) / len(diffs) * 100,
            'greater_20deg': np.sum(diffs >= 20.0) / len(diffs) * 100,
        }


# 全局统计实例
CONSTELL_CHECK_STATS = ConstellationCheckStats()