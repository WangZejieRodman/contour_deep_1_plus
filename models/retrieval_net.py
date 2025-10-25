"""
RetrievalNet: 方向1检索特征提取网络 (完整版)
输入: [B, 9, 200, 200] (8层BEV + 1层VCD)
输出: [B, 128] 全局特征向量

架构设计理念：
1. 输入的9个通道：前8个是8层BEV，第9个是VCD
2. 跨层注意力：学习8层BEV的重要性权重，融合多层信息
"""

import torch
import torch.nn as nn
from models.modules import MultiScaleConv, SpatialAttention


class ResBlock(nn.Module):
    """残差块，用于下采样"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class SimpleCrossLayerAttention(nn.Module):
    """
    简化的跨层注意力模块

    核心思想：
    - 对8层BEV分别处理后的特征，学习一个权重向量
    - 加权融合得到最终特征
    """

    def __init__(self, feature_dim=128, num_bev_layers=8):
        super(SimpleCrossLayerAttention, self).__init__()
        self.num_bev_layers = num_bev_layers
        self.feature_dim = feature_dim

        # 为每层学习一个注意力权重
        # 输入：全局特征 → 输出：8个权重
        self.attention_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, H, W] → [B, C, 1, 1]
            nn.Flatten(),  # [B, C]
            nn.Linear(feature_dim, num_bev_layers),
            nn.Softmax(dim=1)  # [B, 8]，权重和为1
        )

    def forward(self, x, layer_features):
        """
        Args:
            x: [B, C, H, W] 当前主特征
            layer_features: List of [B, C_layer, H, W]，长度为8
                           每个元素是一层BEV处理后的特征

        Returns:
            weighted_features: [B, C_total, H, W]
        """
        # 计算8层的注意力权重
        attention_weights = self.attention_fc(x)  # [B, 8]

        # 确保layer_features有8个元素
        assert len(layer_features) == self.num_bev_layers

        # 堆叠layer_features
        stacked = torch.stack(layer_features, dim=1)  # [B, 8, C_layer, H, W]

        # 加权求和
        attention_weights = attention_weights.view(-1, self.num_bev_layers, 1, 1, 1)  # [B, 8, 1, 1, 1]
        weighted = stacked * attention_weights  # [B, 8, C_layer, H, W]

        # 沿层维度求和
        output = weighted.sum(dim=1)  # [B, C_layer, H, W]

        return output


class RetrievalNet(nn.Module):
    """
    完整版RetrievalNet（包含跨层注意力）

    流程：
    Input [B, 9, 200, 200]
      ↓ 分离8层BEV + 1层VCD
    → 对每层BEV独立处理（共享权重的卷积）
    → 拼接得到 [B, 128, 200, 200]
    → MultiScaleConv (增强特征)
    → SpatialAttention
    → ResBlocks下采样
    → CrossLayerAttention (融合8层)
    → GlobalPooling
    → FC → [B, 128]
    """

    def __init__(self, output_dim=128):
        super(RetrievalNet, self).__init__()

        # === 阶段0: 每层BEV独立编码 ===
        # 为8层BEV + 1层VCD分别编码
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )  # 每层：1 → 16通道，输入1个通道（二值图：0或255），输出16个通道（16种不同的特征响应），每个通道是一个特征检测器（类似"边缘检测器"、"角点检测器"）

        # === 阶段1: 多尺度卷积 ===
        # 输入：9×16=144通道(层级信息隐式编码在了特征的激活模式中) → 输出：128通道
        self.multiscale_conv = MultiScaleConv(in_channels=144, out_channels=128)

        # === 阶段2: 空间注意力 ===
        self.spatial_attention = SpatialAttention()

        # === 阶段3: 残差下采样 ===
        self.res_block1 = ResBlock(128, 128, stride=2)  # 200→100
        self.res_block2 = ResBlock(128, 128, stride=2)  # 100→50
        self.res_block3 = ResBlock(128, 128, stride=2)  # 50→25

        # === 阶段4: 跨层注意力 ===
        # 注意：此时每层特征是16维，所以输出也是16维
        self.cross_layer_attention = SimpleCrossLayerAttention(
            feature_dim=128,
            num_bev_layers=8
        )

        # 下采样模块（用于layer_features）
        self.downsample_for_layers = nn.Sequential(
            ResBlock(16, 16, stride=2),
            ResBlock(16, 16, stride=2),
            ResBlock(16, 16, stride=2),
        )  # 200→100→50→25

        # === 阶段5: 全局池化 ===
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        # === 阶段6: 全连接 ===
        # 跨层注意力输出16维 → GAP+GMP拼接 → 32维
        self.fc = nn.Sequential(
            nn.Linear(32, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim)
        )

    def forward(self, x):
        """
        Args:
            x: [B, 9, 200, 200]
               - 前8个通道：8层BEV
               - 第9个通道：VCD

        Returns:
            features: [B, 128]
        """
        batch_size = x.size(0)

        # === 阶段0: 分离并编码每层BEV ===
        # 分离9个通道
        individual_channels = []
        for i in range(9):
            channel = x[:, i:i+1, :, :]  # [B, 1, 200, 200]
            encoded = self.bev_encoder(channel)  # [B, 16, 200, 200]
            individual_channels.append(encoded)

        # 拼接：9×16 = 144通道
        x_concat = torch.cat(individual_channels, dim=1)  # [B, 144, 200, 200]

        # 保存前8层（BEV）的特征用于跨层注意力
        layer_features_original = individual_channels[:8]  # List of [B, 16, 200, 200]

        # === 阶段1: 多尺度卷积 ===
        x = self.multiscale_conv(x_concat)  # [B, 128, 200, 200]

        # === 阶段2: 空间注意力 ===
        x = self.spatial_attention(x)  # [B, 128, 200, 200]

        # === 阶段3: 残差下采样 ===
        x = self.res_block1(x)  # [B, 128, 100, 100]
        x = self.res_block2(x)  # [B, 128, 50, 50]
        x = self.res_block3(x)  # [B, 128, 25, 25]

        # === 阶段4: 跨层注意力 ===
        # 将layer_features也下采样到25×25
        layer_features_downsampled = []
        for lf in layer_features_original:
            lf_down = self.downsample_for_layers(lf)  # [B, 16, 25, 25]
            layer_features_downsampled.append(lf_down)

        # 应用跨层注意力
        x_attended = self.cross_layer_attention(x, layer_features_downsampled)  # [B, 16, 25, 25]

        # === 阶段5: 全局池化 ===
        gap_features = self.gap(x_attended)  # [B, 16, 1, 1]
        gmp_features = self.gmp(x_attended)  # [B, 16, 1, 1]

        gap_features = gap_features.view(batch_size, -1)  # [B, 16]
        gmp_features = gmp_features.view(batch_size, -1)  # [B, 16]

        global_features = torch.cat([gap_features, gmp_features], dim=1)  # [B, 32]

        # === 阶段6: 全连接 ===
        output = self.fc(global_features)  # [B, 128]

        # L2归一化
        output = torch.nn.functional.normalize(output, p=2, dim=1)

        return output

    def get_embedding_dim(self):
        return 128


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("Testing RetrievalNet (Full Version with CrossLayerAttention)...")

    # 创建网络
    model = RetrievalNet(output_dim=128)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n网络参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  参数量: {total_params / 1e6:.2f}M")
    print(f"  目标: <10M ({'✓' if total_params < 10e6 else '✗'})")

    # 测试前向传播
    print("\n测试前向传播:")
    batch_size = 4
    x = torch.randn(batch_size, 9, 200, 200)

    print(f"  输入: {x.shape}")

    with torch.no_grad():
        output = model(x)

    print(f"  输出: {output.shape}")
    print(f"  预期: [{batch_size}, 128]")

    assert output.shape == (batch_size, 128), "输出维度错误!"

    # 验证L2归一化
    norms = torch.norm(output, p=2, dim=1)
    print(f"\n  特征向量L2范数: {norms}")
    print(f"  范数均值: {norms.mean().item():.6f} (应接近1.0)")

    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "L2归一化失败!"

    # 测试梯度
    print("\n测试梯度反向传播:")
    x.requires_grad = True
    output = model(x)
    loss = output.sum()
    loss.backward()

    print(f"  梯度检查: {'✓ 通过' if x.grad is not None else '✗ 失败'}")

    # 推理速度测试
    print("\n推理速度测试:")
    model.eval()
    import time

    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}")

    # 将模型和数据移到GPU
    model = model.to(device)
    x_test = torch.randn(batch_size, 9, 200, 200).to(device)

    with torch.no_grad():
        # 预热
        for _ in range(10):
            _ = model(x_test)

        # GPU同步
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # 计时
        start = time.time()
        num_iterations = 100
        for _ in range(num_iterations):
            _ = model(x_test)

        # GPU同步
        if device.type == 'cuda':
            torch.cuda.synchronize()

        end = time.time()

    avg_time = (end - start) / num_iterations * 1000
    fps = 1000 / avg_time * batch_size

    print(f"  平均推理时间: {avg_time:.2f}ms")
    print(f"  吞吐量: {fps:.1f} samples/sec")
    print(f"  目标: >30 FPS ({'✓ 达标' if fps > 30 else '✗ 未达标'})")

    print("\n✓ RetrievalNet (Full Version) 所有测试通过!")
    print("\n架构说明:")
    print("  - 包含完整的CrossLayerAttention")
    print("  - 每层BEV独立编码后再融合")
    print("  - 跨层注意力学习8层BEV的重要性权重")