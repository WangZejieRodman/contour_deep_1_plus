"""
Day 9: 训练集自查询验证
用法: 在PyCharm中运行
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm


def evaluate_recall_on_trainset():
    """在训练集上自查询评估（排除自身）"""
    print("=" * 60)
    print("训练集自查询验证")
    print("=" * 60)
    print("目的: 验证模型是否学到了东西")
    print("方法: 在训练集内部进行KNN检索（排除自身）")
    print("=" * 60)

    from models.retrieval_net import RetrievalNet
    from data.dataset_retrieval import RetrievalDataset
    import yaml
    from sklearn.neighbors import NearestNeighbors

    # 1. 加载模型
    print("\n[1/5] 加载模型...")
    model = RetrievalNet(output_dim=128)
    checkpoint = torch.load('checkpoints/retrieval_baseline_day8/latest.pth',
                           map_location='cuda',
                           weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    model.eval()
    print(f"  ✓ 加载Epoch {checkpoint['epoch']+1}的模型")
    print(f"  ✓ Val Loss: {checkpoint['metric']:.4f}")

    # 2. 加载配置
    with open('/home/wzj/pan1/contour_deep_1/configs/config_base.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 3. 创建训练集（作为数据库和查询）
    print("\n[2/5] 加载训练集...")
    train_dataset = RetrievalDataset(
        queries_pickle='/home/wzj/pan1/contour_deep_1/data/training_queries_chilean_period.pickle',
        cache_root='/home/wzj/pan1/contour_deep_1/data/Chilean_BEV_Cache',
        split='train',
        num_negatives=10,
        augmentation_config=None,  # 评估时不增强
        resolution=config['bev']['resolution'],
        use_cache=True
    )
    print(f"  ✓ 训练集大小: {len(train_dataset.query_keys)}")

    # 4. 提取所有训练集特征
    print("\n[3/5] 提取训练集特征...")
    all_features = []
    all_keys = []
    all_ground_truths = []

    with torch.no_grad():
        for query_key in tqdm(train_dataset.query_keys, desc="提取特征"):
            # 加载BEV
            bev_data = train_dataset._load_bev_from_cache(query_key)
            if bev_data is None:
                continue

            # 预处理
            bev_tensor = train_dataset._preprocess_bev(*bev_data, apply_aug=False)
            bev_tensor = bev_tensor.unsqueeze(0).cuda()

            # 提取特征
            feat = model(bev_tensor)
            all_features.append(feat.cpu().numpy())
            all_keys.append(query_key)

            # 获取ground truth（训练集内的positives）
            query_data = train_dataset.queries[query_key]
            all_ground_truths.append(set(query_data['positives']))#排除自身
            # all_ground_truths.append(set(query_data['positives']) | {query_key})# 如果要"不排除自身"，需要将自己也加入正样本集合

    all_features = np.vstack(all_features)  # [N, 128]
    print(f"  ✓ 提取了 {len(all_features)} 个特征")

    # 5. 构建KNN索引
    print("\n[4/5] 构建KNN索引...")
    knn = NearestNeighbors(n_neighbors=26,  # Top-25 + 自身
                          metric='euclidean',
                          n_jobs=-1)
    knn.fit(all_features)
    print(f"  ✓ 索引构建完成")

    # 6. 计算Recall@1, @5, @10, @25（排除自身）
    print("\n[5/5] 计算Recall@N（排除自身）...")
    recalls = {1: 0, 5: 0, 10: 0, 25: 0}
    valid_queries = 0

    # 统计信息
    total_positives = []
    rank_of_first_positive = []

    for i in tqdm(range(len(all_features)), desc="KNN检索"):
        if len(all_ground_truths[i]) == 0:
            continue

        valid_queries += 1
        total_positives.append(len(all_ground_truths[i]))

        # KNN搜索（返回26个，包括自身）
        distances, indices = knn.kneighbors([all_features[i]])

        # 排除自身（第一个结果一定是自己）
        retrieved_indices = indices[0][1:]  # 去掉第一个（自己）
        # retrieved_indices = indices[0] # 不去掉第一个（自己）
        retrieved_keys = [all_keys[idx] for idx in retrieved_indices]

        # 找到第一个正样本的排名
        first_positive_rank = None
        for rank, key in enumerate(retrieved_keys, start=1):
            if key in all_ground_truths[i]:
                first_positive_rank = rank
                break

        if first_positive_rank:
            rank_of_first_positive.append(first_positive_rank)

        # 检查Recall@K
        for k in [1, 5, 10, 25]:
            if any(key in all_ground_truths[i] for key in retrieved_keys[:k]):
                recalls[k] += 1

    # 归一化
    for k in recalls:
        recalls[k] = recalls[k] / valid_queries * 100

    # 7. 输出结果
    print("\n" + "=" * 60)
    print("训练集自查询结果:")
    print("=" * 60)
    print(f"有效查询数: {valid_queries}")
    print(f"平均正样本数: {np.mean(total_positives):.1f}")
    print(f"\nRecall性能:")
    print(f"  Recall@1:  {recalls[1]:.2f}%")
    print(f"  Recall@5:  {recalls[5]:.2f}%")
    print(f"  Recall@10: {recalls[10]:.2f}%")
    print(f"  Recall@25: {recalls[25]:.2f}%")

    # 统计第一个正样本的平均排名
    if rank_of_first_positive:
        avg_rank = np.mean(rank_of_first_positive)
        median_rank = np.median(rank_of_first_positive)
        print(f"\n第一个正样本排名统计:")
        print(f"  平均排名: {avg_rank:.1f}")
        print(f"  中位数排名: {median_rank:.1f}")

    # 8. 诊断结论
    print("\n" + "=" * 60)
    print("诊断结论:")
    print("=" * 60)

    if recalls[1] > 80:
        print(f"  ✅ 模型学习正常 (Recall@1 = {recalls[1]:.2f}%)")
        print(f"  ✅ 特征空间良好，正样本被拉近")
        print(f"  → 问题出在评估逻辑，需要修复跨时间段评估")
    elif recalls[1] > 50:
        print(f"  ⚠️  模型学习一般 (Recall@1 = {recalls[1]:.2f}%)")
        print(f"  ⚠️  特征有一定区分能力，但还不够强")
        print(f"  → 建议: 调整损失函数参数或训练更多epochs")
    elif recalls[1] > 20:
        print(f"  ❌ 模型学习较差 (Recall@1 = {recalls[1]:.2f}%)")
        print(f"  ❌ 特征空间混乱，正负样本未分离")
        print(f"  → 建议: 检查Triplet Loss的margin参数")
    else:
        print(f"  ❌ 模型几乎没学到东西 (Recall@1 = {recalls[1]:.2f}%)")
        print(f"  ❌ 特征可能是随机的")
        print(f"  → 建议: 检查数据、模型架构、损失函数")

    return recalls


def quick_feature_analysis():
    """快速特征分析（可选）"""
    print("\n" + "=" * 60)
    print("特征空间分析")
    print("=" * 60)

    from models.retrieval_net import RetrievalNet
    from data.dataset_retrieval import RetrievalDataset
    import yaml

    # 加载模型
    model = RetrievalNet(output_dim=128)
    checkpoint = torch.load('checkpoints/retrieval_baseline_day8/latest.pth',
                           map_location='cuda',
                           weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    model.eval()

    # 加载配置
    with open('/home/wzj/pan1/contour_deep_1/configs/config_base.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 加载训练集（只取前100个样本快速分析）
    train_dataset = RetrievalDataset(
        queries_pickle='/home/wzj/pan1/contour_deep_1/data/training_queries_chilean_period.pickle',
        cache_root='/home/wzj/pan1/contour_deep_1/data/Chilean_BEV_Cache',
        split='train',
        num_negatives=10,
        augmentation_config=None,
        resolution=config['bev']['resolution'],
        use_cache=True
    )

    # 只取前100个
    sample_keys = train_dataset.query_keys[:100]

    # 提取特征
    features = []
    labels = []  # 0=有正样本, 1=无正样本

    with torch.no_grad():
        for query_key in tqdm(sample_keys, desc="提取特征（前100个）"):
            bev_data = train_dataset._load_bev_from_cache(query_key)
            if bev_data is None:
                continue

            bev_tensor = train_dataset._preprocess_bev(*bev_data, apply_aug=False)
            bev_tensor = bev_tensor.unsqueeze(0).cuda()

            feat = model(bev_tensor)
            features.append(feat.cpu().numpy()[0])

            # 标签
            query_data = train_dataset.queries[query_key]
            label = 0 if len(query_data['positives']) > 0 else 1
            labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    # 统计特征范数
    norms = np.linalg.norm(features, axis=1)
    print(f"\n特征向量统计:")
    print(f"  L2范数: {norms.mean():.4f} ± {norms.std():.4f}")
    print(f"  范数范围: [{norms.min():.4f}, {norms.max():.4f}]")

    # 计算特征间距离
    from sklearn.metrics.pairwise import euclidean_distances
    dist_matrix = euclidean_distances(features)

    # 排除对角线
    mask = np.ones_like(dist_matrix, dtype=bool)
    np.fill_diagonal(mask, False)
    distances = dist_matrix[mask]

    print(f"\n特征间距离统计:")
    print(f"  平均距离: {distances.mean():.4f}")
    print(f"  标准差: {distances.std():.4f}")
    print(f"  最小距离: {distances.min():.4f}")
    print(f"  最大距离: {distances.max():.4f}")

    # 分析：正样本对 vs 负样本对的距离
    print(f"\n正负样本距离分析:")
    print(f"  如果模型学习正常:")
    print(f"    - 正样本对距离应该 < 平均距离")
    print(f"    - 负样本对距离应该 > 平均距离")


def main():
    """主函数"""
    print("=" * 60)
    print("Day 9: 训练集自查询验证")
    print("=" * 60)

    # 1. 检查checkpoint
    checkpoint_path = "checkpoints/retrieval_baseline_day8/latest.pth"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint不存在: {checkpoint_path}")
        return

    # 2. 训练集自查询评估
    try:
        recalls = evaluate_recall_on_trainset()

        # 3. 保存结果
        import json
        results = {
            'evaluation_type': 'train_self_query',
            'recalls': recalls,
            'checkpoint': checkpoint_path,
        }

        os.makedirs('/home/wzj/pan1/contour_deep_1/scripts/logs/day9_evaluation_self', exist_ok=True)
        with open('/home/wzj/pan1/contour_deep_1/scripts/logs/day9_evaluation_self/train_self_query-4.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n结果已保存: /home/wzj/pan1/contour_deep_1/scripts/logs/day9_evaluation_self/train_self_query-4.json")

    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 可选：特征空间分析
    print("\n是否进行特征空间分析？")
    print("（这会额外花费约1分钟）")

    try:
        quick_feature_analysis()
    except Exception as e:
        print(f"特征分析出错（可忽略）: {e}")

    # 5. 下一步建议
    print("\n" + "=" * 60)
    print("下一步建议:")
    print("=" * 60)

    if 'recalls' in locals():
        if recalls[1] > 80:
            print("  ✅ 模型没问题！")
            print("  → 现在需要修复跨时间段评估逻辑")
            print("  → 使用GPS坐标计算True Positive")
        elif recalls[1] > 50:
            print("  ⚠️  模型有一定能力，但还需优化")
            print("  → Day 10: 调整超参数")
            print("  → 尝试: 减小margin (0.5→0.3)")
            print("  → 尝试: 增加训练epochs")
        else:
            print("  ❌ 模型学习效果差")
            print("  → 检查损失函数参数")
            print("  → 检查数据增强")
            print("  → 考虑调整模型结构")


if __name__ == "__main__":
    main()