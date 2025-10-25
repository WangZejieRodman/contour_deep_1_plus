"""
BEV Generator: 从原始点云生成多层BEV表示
功能：
1. 读取原始点云（Chilean格式：.bin文件）
2. 生成多层BEV二值图（8层）
3. 生成垂直复杂度图（VCD）
4. 保存为缓存文件（.npz格式）
"""

import numpy as np
import os
import sys
from typing import Tuple, List, Optional
import yaml

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.contour_types import ContourManagerConfig


class BEVGenerator:
    """BEV生成器类"""

    def __init__(self, config: ContourManagerConfig):
        """
        初始化BEV生成器

        Args:
            config: 轮廓管理器配置
        """
        self.cfg = config

        # 验证配置
        assert config.n_col % 2 == 0
        assert config.n_row % 2 == 0
        assert len(config.lv_grads) > 0

        # 坐标范围
        self.x_min = -(config.n_row // 2) * config.reso_row
        self.x_max = -self.x_min
        self.y_min = -(config.n_col // 2) * config.reso_col
        self.y_max = -self.y_min

        # BEV数据存储
        self.bev = None
        self.layer_masks = None  # 垂直结构复杂度
        self.bev_pixfs = []  # BEV像素信息

        # 初始化BEV
        self._init_bev()

    def _init_bev(self):
        """初始化BEV图像"""
        self.bev = np.full((self.cfg.n_row, self.cfg.n_col), -1000.0, dtype=np.float32)

    def hash_point_to_image(self, pt: np.ndarray) -> Tuple[int, int]:
        """
        将点映射到图像坐标

        Args:
            pt: 点坐标 [x, y, z]

        Returns:
            (row, col) 或 (-1, -1) 如果点在范围外
        """
        padding = 1e-2
        x, y = pt[0], pt[1]

        # 检查范围
        if (x < self.x_min + padding or x > self.x_max - padding or
                y < self.y_min + padding or y > self.y_max - padding or
                (y * y + x * x) < self.cfg.blind_sq):
            return -1, -1

        row = int(np.floor(x / self.cfg.reso_row)) + self.cfg.n_row // 2
        col = int(np.floor(y / self.cfg.reso_col)) + self.cfg.n_col // 2

        # 验证范围
        if not (0 <= row < self.cfg.n_row and 0 <= col < self.cfg.n_col):
            return -1, -1

        return row, col

    def point_to_cont_row_col(self, p_in_l: np.ndarray) -> np.ndarray:
        """
        将激光雷达坐标系中的点转换到连续图像坐标系

        Args:
            p_in_l: 激光雷达坐标系中的点 [x, y]

        Returns:
            连续的行列坐标
        """
        continuous_rc = np.array([
            p_in_l[0] / self.cfg.reso_row + self.cfg.n_row / 2 - 0.5,
            p_in_l[1] / self.cfg.reso_col + self.cfg.n_col / 2 - 0.5
        ], dtype=np.float32)
        return continuous_rc

    def make_bev(self, point_cloud: np.ndarray, str_id: str = "") -> Tuple[np.ndarray, np.ndarray]:
        """
        从点云生成BEV图像（核心函数）

        Args:
            point_cloud: 点云数组，形状为 [N, 3] (x, y, z)
            str_id: 字符串ID（用于调试）

        Returns:
            bev_layers: 多层BEV二值图 [num_layers, H, W]
            vcd: 垂直复杂度图 [H, W]
        """
        assert point_cloud.shape[0] > 10, "点云数量太少"
        assert point_cloud.shape[1] >= 3, "点云必须包含x,y,z坐标"

        # 清空之前的数据
        self.bev_pixfs.clear()
        self._init_bev()

        # 初始化层级掩码（用于垂直结构复杂度）
        lv_grads = self.cfg.lv_grads
        num_layers = len(lv_grads) - 1
        self.layer_masks = np.zeros((self.cfg.n_row, self.cfg.n_col, num_layers), dtype=bool)

        tmp_pillars = {}

        # 处理每个点
        for pt in point_cloud:
            row, col = self.hash_point_to_image(pt)
            if row >= 0:
                height = self.cfg.lidar_height + pt[2]

                # 更新最大高度
                if self.bev[row, col] < height:
                    self.bev[row, col] = height
                    # 计算连续坐标
                    coor_f = self.point_to_cont_row_col(pt[:2])
                    hash_key = row * self.cfg.n_col + col
                    tmp_pillars[hash_key] = (coor_f[0], coor_f[1], height)

                # 判断该点属于哪个层级并记录
                for level in range(num_layers):
                    h_min = lv_grads[level]
                    h_max = lv_grads[level + 1]
                    if h_min <= height < h_max:
                        self.layer_masks[row, col, level] = True

        # 转换为列表格式
        self.bev_pixfs = [(k, v) for k, v in tmp_pillars.items()]
        self.bev_pixfs.sort(key=lambda x: x[0])

        # 生成多层BEV二值图
        bev_layers = self._generate_bev_layers(lv_grads)

        # 生成垂直复杂度图
        vcd = self._generate_vertical_complexity_map()

        return bev_layers, vcd

    def _generate_bev_layers(self, lv_grads: List[float]) -> np.ndarray:
        """
        生成多层BEV二值图

        Args:
            lv_grads: 层级高度阈值列表

        Returns:
            bev_layers: [num_layers, H, W]，二值图（0或255）
        """
        num_layers = len(lv_grads) - 1
        bev_layers = np.zeros((num_layers, self.cfg.n_row, self.cfg.n_col), dtype=np.uint8)

        for level in range(num_layers):
            h_min = lv_grads[level]
            h_max = lv_grads[level + 1]

            # 创建区间掩码：[h_min, h_max)
            mask = ((self.bev >= h_min) & (self.bev < h_max)).astype(np.uint8) * 255
            bev_layers[level] = mask

        return bev_layers

    def _generate_vertical_complexity_map(self) -> np.ndarray:
        """
        生成垂直复杂度图（每个像素位置有多少层存在结构）

        Returns:
            vcd: [H, W]，整数图（0到num_layers）
        """
        # 沿着层级维度求和，得到每个像素位置的层数
        vcd = np.sum(self.layer_masks, axis=2).astype(np.uint8)
        return vcd

    def get_bev_image(self) -> np.ndarray:
        """获取原始BEV图像（高度图）"""
        return self.bev.copy()


def load_chilean_pointcloud(filepath: str) -> np.ndarray:
    """
    加载Chilean点云文件（.bin格式）

    Args:
        filepath: 点云文件路径

    Returns:
        point_cloud: [N, 3] numpy数组
    """
    try:
        # 读取二进制数据
        pc = np.fromfile(filepath, dtype=np.float64)

        # 检查数据长度是否是3的倍数
        if len(pc) % 3 != 0:
            print(f"Warning: 点云数据长度不是3的倍数: {len(pc)}")
            return np.array([])

        # reshape为 [N, 3] 格式
        num_points = len(pc) // 3
        pc = pc.reshape(num_points, 3)

        return pc

    except Exception as e:
        print(f"Error: 加载点云失败 {filepath}: {e}")
        return np.array([])


def generate_bev_from_file(pointcloud_path: str,
                           config: ContourManagerConfig,
                           save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    从点云文件生成BEV（便捷函数）

    Args:
        pointcloud_path: 点云文件路径
        config: BEV配置
        save_path: 保存路径（可选）

    Returns:
        bev_layers: [num_layers, H, W]
        vcd: [H, W]
    """
    # 1. 加载点云
    pointcloud = load_chilean_pointcloud(pointcloud_path)
    if len(pointcloud) == 0:
        raise ValueError(f"无法加载点云: {pointcloud_path}")

    # 2. 生成BEV
    generator = BEVGenerator(config)
    bev_layers, vcd = generator.make_bev(pointcloud)

    # 3. 保存（可选）
    if save_path is not None:
        np.savez_compressed(save_path, bev_layers=bev_layers, vcd=vcd)
        print(f"BEV saved to: {save_path}")

    return bev_layers, vcd


def load_config_from_yaml(yaml_path: str) -> ContourManagerConfig:
    """
    从YAML配置文件加载BEV配置

    Args:
        yaml_path: config_base.yaml路径

    Returns:
        ContourManagerConfig实例
    """
    with open(yaml_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    # 提取BEV相关配置
    bev_cfg = cfg_dict['bev']

    config = ContourManagerConfig()
    config.lv_grads = bev_cfg['lv_grads']
    config.reso_row = bev_cfg['resolution']
    config.reso_col = bev_cfg['resolution']
    config.n_row = bev_cfg['grid_size']['n_row']
    config.n_col = bev_cfg['grid_size']['n_col']
    config.roi_radius = bev_cfg['roi_radius']
    config.lidar_height = bev_cfg['lidar_height']
    config.blind_sq = bev_cfg['blind_zone']

    return config


# ========== 测试代码 ==========
if __name__ == "__main__":
    """
    测试BEV生成器
    用法: python data/bev_generator.py
    """
    print("Testing BEV Generator...")

    # 1. 加载配置
    config_path = "/home/wzj/pan1/contour_deep_1/configs/config_base.yaml"
    if not os.path.exists(config_path):
        print(f"Error: 配置文件不存在 {config_path}")
        print("请先创建 /home/wzj/pan1/contour_deep_1/configs/config_base.yaml")
        sys.exit(1)

    config = load_config_from_yaml(config_path)
    print(f"配置加载成功: {len(config.lv_grads) - 1}层, 分辨率{config.reso_row}m")

    # 2. 测试点云文件（需要修改为你的实际路径）
    test_pointcloud_path = "/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times/chilean_NoRot_NoScale_5cm/100/pointcloud_20m_10overlap/100008.bin"

    if not os.path.exists(test_pointcloud_path):
        print(f"Warning: 测试点云文件不存在 {test_pointcloud_path}")
        print("请修改 test_pointcloud_path 为实际路径")

        # 创建一个模拟点云用于测试
        print("使用模拟点云进行测试...")
        test_pointcloud = np.random.randn(10000, 3) * 5  # 10000个点
    else:
        test_pointcloud = load_chilean_pointcloud(test_pointcloud_path)
        print(f"加载点云成功: {len(test_pointcloud)} 个点")

    # 3. 生成BEV
    generator = BEVGenerator(config)
    bev_layers, vcd = generator.make_bev(test_pointcloud, "test")

    # 4. 输出统计信息
    print(f"\n生成结果:")
    print(f"  BEV layers shape: {bev_layers.shape}")  # 应该是 [8, 200, 200]
    print(f"  VCD shape: {vcd.shape}")  # 应该是 [200, 200]
    print(f"  BEV层数: {bev_layers.shape[0]}")
    print(f"  每层占用像素数:")
    for i in range(bev_layers.shape[0]):
        occupied = np.sum(bev_layers[i] > 0)
        print(f"    Layer {i}: {occupied} pixels")
    print(f"  VCD最大值: {vcd.max()}")  # 应该<=8
    print(f"  VCD平均值: {vcd.mean():.2f}")

    # 5. 保存测试结果
    test_save_path = "test_bev_output.npz"
    np.savez_compressed(test_save_path, bev_layers=bev_layers, vcd=vcd)
    print(f"\n测试结果已保存: {test_save_path}")

    # 6. 验证加载
    loaded = np.load(test_save_path)
    assert np.array_equal(loaded['bev_layers'], bev_layers)
    assert np.array_equal(loaded['vcd'], vcd)
    print("✓ 保存/加载验证成功!")

    print("\n✓ BEV Generator测试完成!")