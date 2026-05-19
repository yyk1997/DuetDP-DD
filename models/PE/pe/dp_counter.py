import faiss
import logging
import numpy as np
from collections import Counter
import torch


def dp_nn_histogram(public_features, private_features, noise_multiplier,
                    num_packing=1, num_nearest_neighbor=1, mode='L2',
                    threshold=0.0):

    assert public_features.shape[0] % num_packing == 0
    num_true_public_features = public_features.shape[0] // num_packing

    d = public_features.shape[1]

    # -------------------------------------------------
    # 1. 判断是否真的可以使用 GPU Faiss
    # -------------------------------------------------
    use_gpu_faiss = (
        hasattr(faiss, "StandardGpuResources") and
        torch.cuda.is_available()
    )

    if use_gpu_faiss:
        logging.info("[Faiss] GPU version available, using GPU Faiss")
    else:
        logging.info("[Faiss] GPU Faiss NOT available, fallback to CPU Faiss")

    # -------------------------------------------------
    # 2. 构建 CPU index
    # -------------------------------------------------
    if mode == 'L2':
        cpu_index = faiss.IndexFlatL2(d)
    elif mode == 'IP':
        cpu_index = faiss.IndexFlatIP(d)
    else:
        raise Exception(f'Unknown mode {mode}')

    # -------------------------------------------------
    # 3. 如果可以，用 GPU index；否则继续用 CPU
    # -------------------------------------------------
    if use_gpu_faiss:
        try:
            res = faiss.StandardGpuResources()
            # 使用第 0 张 GPU（如果你想用别的卡，可以改这里）
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        except Exception as e:
            logging.warning(
                f"[Faiss] Failed to init GPU index, fallback to CPU. Error: {e}"
            )
            index = cpu_index
            use_gpu_faiss = False
    else:
        index = cpu_index

    # -------------------------------------------------
    # 4. 保证输入是 numpy array（Faiss 只吃 numpy）
    # -------------------------------------------------
    if torch.is_tensor(public_features):
        public_features = public_features.detach().cpu().numpy()
    if torch.is_tensor(private_features):
        private_features = private_features.detach().cpu().numpy()

    # -------------------------------------------------
    # 5. 建索引 + 搜索
    # -------------------------------------------------
    index.add(public_features)
    logging.info(f'Number of samples in index: {index.ntotal}')

    _, ids = index.search(private_features, k=num_nearest_neighbor)
    logging.info('Finished search')

    # -------------------------------------------------
    # 6. 统计直方图
    # -------------------------------------------------
    counter = Counter(list(ids.flatten()))
    count = np.zeros(shape=num_true_public_features)

    for k in counter:
        count[k % num_true_public_features] += counter[k]

    logging.info(f'Clean count sum: {np.sum(count)}')
    logging.info(f'Clean count num>0: {np.sum(count > 0)}')
    logging.info(f'Largest clean counters: {sorted(count)[::-1][:50]}')

    clean_count = count.copy()

    # -------------------------------------------------
    # 7. 加噪声（DP）
    # -------------------------------------------------
    count = count + (
        np.random.normal(size=len(count))
        * np.sqrt(num_nearest_neighbor)
        * noise_multiplier
    )

    logging.info(f'Noisy count sum: {np.sum(count)}')
    logging.info(f'Noisy count num>0: {np.sum(count > 0)}')
    logging.info(f'Largest noisy counters: {sorted(count)[::-1][:50]}')

    # -------------------------------------------------
    # 8. clip
    # -------------------------------------------------
    count = np.clip(count, a_min=threshold, a_max=None)
    count = count - threshold

    logging.info(f'Clipped noisy count sum: {np.sum(count)}')
    logging.info(f'Clipped noisy count num>0: {np.sum(count > 0)}')
    logging.info(f'Clipped largest noisy counters: {sorted(count)[::-1][:50]}')

    return count, clean_count
