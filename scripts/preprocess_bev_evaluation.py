"""
BEV预处理脚本：为评估数据集生成缓存
用法: python scripts/preprocess_bev_evaluation.py --split database
      python scripts/preprocess_bev_evaluation.py --split query
      python scripts/preprocess_bev_evaluation.py --split all
"""

import numpy as np
import os
import sys
import argparse
import yaml
from tqdm import tqdm
from pathlib import Path
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.bev_generator import BEVGenerator, load_chilean_pointcloud, load_config_from_yaml
import pickle


def setup_logging(log_dir: str):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"preprocess_eval_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


def load_pickle(filepath: str):
    """加载pickle文件"""
    with open(filepath, 'rb') as handle:
        data = pickle.load(handle)
    logging.info(f"加载pickle: {filepath}")
    return data


def process_single_pointcloud(pointcloud_path: str,
                              generator: BEVGenerator,
                              base_path: str):
    """
    处理单个点云

    Returns:
        (bev_layers, vcd, success) 或 (None, None, False)
    """
    # 构建完整路径
    full_path = os.path.join(base_path, pointcloud_path)

    # 检查文件是否存在
    if not os.path.exists(full_path):
        logging.warning(f"点云文件不存在: {full_path}")
        return None, None, False

    try:
        # 加载点云
        pointcloud = load_chilean_pointcloud(full_path)

        # 检查点云大小
        if len(pointcloud) < 100:
            logging.warning(f"点云太小 ({len(pointcloud)} points): {pointcloud_path}")
            return None, None, False

        # 生成BEV
        bev_layers, vcd = generator.make_bev(pointcloud)

        return bev_layers, vcd, True

    except Exception as e:
        logging.error(f"处理失败 {pointcloud_path}: {e}")
        return None, None, False


def preprocess_database(config_path: str,
                       base_path: str,
                       cache_root: str,
                       database_pickle: str,
                       overwrite: bool = False):
    """
    预处理数据库集（Session 190-194）

    Args:
        config_path: config_base.yaml路径
        base_path: 数据集根目录
        cache_root: 缓存根目录
        database_pickle: 数据库pickle文件路径
        overwrite: 是否覆盖已存在的缓存
    """
    logging.info(f"{'='*60}")
    logging.info(f"开始处理数据库集...")
    logging.info(f"{'='*60}")

    # 1. 加载配置
    config = load_config_from_yaml(config_path)
    generator = BEVGenerator(config)
    logging.info(f"BEV配置: {len(config.lv_grads) - 1}层, 分辨率{config.reso_row}m")

    # 2. 加载数据库pickle
    database_sets = load_pickle(database_pickle)
    logging.info(f"加载 {len(database_sets)} 个数据库session")

    # 3. 为每个session创建缓存目录
    total_entries = sum(len(db_set) for db_set in database_sets)
    logging.info(f"总数据库条目: {total_entries}")

    # 4. 批量处理
    success_count = 0
    skip_count = 0
    fail_count = 0

    for session_idx, database in enumerate(database_sets):
        if len(database) == 0:
            logging.info(f"Session {session_idx}: 空，跳过")
            continue

        # 创建该session的缓存目录
        session_cache_dir = os.path.join(cache_root, 'evaluation_database', f'session_{session_idx}')
        os.makedirs(session_cache_dir, exist_ok=True)

        logging.info(f"\n处理数据库 Session {session_idx} ({len(database)} 个条目)...")

        for local_idx in tqdm(sorted(database.keys()), desc=f"  Database Session {session_idx}"):
            entry = database[local_idx]

            # 缓存文件路径
            cache_filename = f"{local_idx:06d}.npz"
            cache_path = os.path.join(session_cache_dir, cache_filename)

            # 检查是否已存在（断点续传）
            if os.path.exists(cache_path) and not overwrite:
                skip_count += 1
                continue

            # 处理点云
            pointcloud_path = entry['query']
            bev_layers, vcd, success = process_single_pointcloud(
                pointcloud_path, generator, base_path
            )

            if success:
                # 保存缓存
                np.savez_compressed(
                    cache_path,
                    bev_layers=bev_layers,
                    vcd=vcd,
                    local_idx=local_idx,
                    session_idx=session_idx,
                    pointcloud_path=pointcloud_path
                )
                success_count += 1
            else:
                fail_count += 1

    # 5. 统计报告
    logging.info(f"\n{'=' * 60}")
    logging.info(f"数据库集处理完成!")
    logging.info(f"  成功: {success_count}")
    logging.info(f"  跳过: {skip_count} (已存在)")
    logging.info(f"  失败: {fail_count}")
    logging.info(f"  总计: {total_entries}")
    logging.info(f"{'=' * 60}\n")

    return success_count, skip_count, fail_count


def preprocess_query(config_path: str,
                    base_path: str,
                    cache_root: str,
                    query_pickle: str,
                    overwrite: bool = False):
    """
    预处理查询集（Session 195-199）

    Args:
        config_path: config_base.yaml路径
        base_path: 数据集根目录
        cache_root: 缓存根目录
        query_pickle: 查询pickle文件路径
        overwrite: 是否覆盖已存在的缓存
    """
    logging.info(f"{'='*60}")
    logging.info(f"开始处理查询集...")
    logging.info(f"{'='*60}")

    # 1. 加载配置
    config = load_config_from_yaml(config_path)
    generator = BEVGenerator(config)
    logging.info(f"BEV配置: {len(config.lv_grads) - 1}层, 分辨率{config.reso_row}m")

    # 2. 加载查询pickle
    query_sets = load_pickle(query_pickle)
    logging.info(f"加载 {len(query_sets)} 个查询session")

    # 3. 为每个session创建缓存目录
    total_entries = sum(len(query_set) for query_set in query_sets)
    logging.info(f"总查询条目: {total_entries}")

    # 4. 批量处理
    success_count = 0
    skip_count = 0
    fail_count = 0

    for session_idx, queries in enumerate(query_sets):
        if len(queries) == 0:
            logging.info(f"Session {session_idx}: 空，跳过")
            continue

        # 创建该session的缓存目录
        session_cache_dir = os.path.join(cache_root, 'evaluation_query', f'session_{session_idx}')
        os.makedirs(session_cache_dir, exist_ok=True)

        logging.info(f"\n处理查询 Session {session_idx} ({len(queries)} 个条目)...")

        for local_idx in tqdm(sorted(queries.keys()), desc=f"  Query Session {session_idx}"):
            entry = queries[local_idx]

            # 缓存文件路径
            cache_filename = f"{local_idx:06d}.npz"
            cache_path = os.path.join(session_cache_dir, cache_filename)

            # 检查是否已存在（断点续传）
            if os.path.exists(cache_path) and not overwrite:
                skip_count += 1
                continue

            # 处理点云
            pointcloud_path = entry['query']
            bev_layers, vcd, success = process_single_pointcloud(
                pointcloud_path, generator, base_path
            )

            if success:
                # 保存缓存
                np.savez_compressed(
                    cache_path,
                    bev_layers=bev_layers,
                    vcd=vcd,
                    local_idx=local_idx,
                    session_idx=session_idx,
                    pointcloud_path=pointcloud_path
                )
                success_count += 1
            else:
                fail_count += 1

    # 5. 统计报告
    logging.info(f"\n{'=' * 60}")
    logging.info(f"查询集处理完成!")
    logging.info(f"  成功: {success_count}")
    logging.info(f"  跳过: {skip_count} (已存在)")
    logging.info(f"  失败: {fail_count}")
    logging.info(f"  总计: {total_entries}")
    logging.info(f"{'=' * 60}\n")

    return success_count, skip_count, fail_count


def verify_cache(cache_dir: str, expected_sessions: int):
    """
    验证缓存完整性

    Args:
        cache_dir: 缓存目录
        expected_sessions: 预期session数量
    """
    logging.info(f"\n验证缓存: {cache_dir}")

    session_dirs = [d for d in Path(cache_dir).iterdir() if d.is_dir() and d.name.startswith('session_')]
    actual_sessions = len(session_dirs)

    logging.info(f"  Session数量: {actual_sessions}/{expected_sessions}")

    if actual_sessions != expected_sessions:
        logging.warning(f"  ⚠️  Session数量不匹配!")
        return False

    # 检查每个session
    total_files = 0
    for session_dir in session_dirs:
        cache_files = list(session_dir.glob("*.npz"))
        total_files += len(cache_files)

        # 随机验证一个文件
        if cache_files:
            sample_file = cache_files[0]
            try:
                data = np.load(sample_file)
                assert 'bev_layers' in data
                assert 'vcd' in data
                assert data['bev_layers'].shape == (8, 200, 200)
                assert data['vcd'].shape == (200, 200)
            except Exception as e:
                logging.error(f"  ✗ 文件损坏: {sample_file} - {e}")
                return False

    logging.info(f"  ✓ 总文件数: {total_files}")

    # 计算磁盘占用
    total_size = 0
    for session_dir in session_dirs:
        for cache_file in session_dir.glob("*.npz"):
            total_size += cache_file.stat().st_size

    size_gb = total_size / (1024 ** 3)
    logging.info(f"  磁盘占用: {size_gb:.2f} GB")

    return True


def main():
    parser = argparse.ArgumentParser(description='评估数据集BEV预处理脚本')
    parser.add_argument('--split', type=str, default='all',
                        choices=['database', 'query', 'all'],
                        help='处理哪个数据集')
    parser.add_argument('--config', type=str,
                        default='/home/wzj/pan1/contour_deep_1/configs/config_base.yaml',
                        help='配置文件路径')
    parser.add_argument('--base_path', type=str,
                        default='/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times/',
                        help='数据集根目录')
    parser.add_argument('--cache_root', type=str,
                        default='/home/wzj/pan1/contour_deep_1/data/Chilean_BEV_Cache/',
                        help='缓存根目录')
    parser.add_argument('--database_pickle', type=str,
                        default='/home/wzj/pan1/contour_deep_1/data/chilean_evaluation_database_190_194.pickle',
                        help='数据库pickle文件路径')
    parser.add_argument('--query_pickle', type=str,
                        default='/home/wzj/pan1/contour_deep_1/data/chilean_evaluation_query_195_199.pickle',
                        help='查询pickle文件路径')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖已存在的缓存')

    args = parser.parse_args()

    # 设置日志
    log_file = setup_logging('logs')
    logging.info(f"日志文件: {log_file}")
    logging.info(f"配置: {args}")

    # 处理数据集
    if args.split in ['database', 'all']:
        database_stats = preprocess_database(
            args.config, args.base_path, args.cache_root,
            args.database_pickle, args.overwrite
        )

    if args.split in ['query', 'all']:
        query_stats = preprocess_query(
            args.config, args.base_path, args.cache_root,
            args.query_pickle, args.overwrite
        )

    # 验证缓存
    if args.split in ['database', 'all']:
        verify_cache(os.path.join(args.cache_root, 'evaluation_database'), expected_sessions=5)

    if args.split in ['query', 'all']:
        verify_cache(os.path.join(args.cache_root, 'evaluation_query'), expected_sessions=5)

    logging.info("\n✓ 预处理完成!")


if __name__ == "__main__":
    main()