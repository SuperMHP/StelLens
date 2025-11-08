import os
import json
import numpy as np
import cv2
from matplotlib.colors import Normalize
from astropy.visualization import ZScaleInterval
from scipy import ndimage
import matplotlib.pyplot as plt

space_mean, space_std = 0.02098092, 0.85422988

def fits_vis(img, method='percentile', lower_percentile=0.5, upper_percentile=99.5, stretch='asinh', max_iter=30):
    """
    图像可视化（支持 ZScale、分位数、混合、细节优化四种方法）
    """
    img = np.asarray(img, dtype=float)
    shape = img.shape  # 保存原始形状
    finite_mask = np.isfinite(img)
    img_valid = img[finite_mask]  # 仅用于统计计算

    if method == 'percentile':
        z1 = np.percentile(img_valid, lower_percentile)
        z2 = np.percentile(img_valid, upper_percentile)

    elif method == 'zscale':
        z = ZScaleInterval()
        z1, z2 = z.get_limits(img_valid)

    elif method == 'hybrid':
        z = ZScaleInterval()
        z1, z2 = z.get_limits(img_valid)
        p1 = np.percentile(img_valid, lower_percentile)
        p2 = np.percentile(img_valid, upper_percentile)
        # z1 = max(z1, p1)
        # z2 = min(z2, p2)
        alpha = 0.9  # 0~1，越大越偏向 zscale
        z1 = alpha * z1 + (1 - alpha) * p1
        z2 = alpha * z2 + (1 - alpha) * p2

    elif method == 'detail':
        # 背景估计
        median = np.median(img_valid)
        std = np.std(img_valid)
        clean_img = img_valid[(img_valid > median - 3*std) & (img_valid < median + 10*std)]

        percentiles = np.linspace(upper_percentile, 99.99, max_iter)
        best_score = -np.inf
        best_params = None

        for up in percentiles:
            vmin = np.percentile(clean_img, lower_percentile)
            vmax = np.percentile(clean_img, up)
            scaled = np.clip((img - vmin) / (vmax - vmin), 0, 1)
            # gamma = 0.5  # <1 强调暗部结构
            # scaled = scaled ** gamma
            # 非线性拉伸
            if stretch == 'asinh':
                scaled = np.arcsinh(10 * scaled) / np.arcsinh(10)

                # stretch_scale = 10 * (vmax - vmin) / (np.percentile(clean_img, 99.5) - np.percentile(clean_img, 0.5))
                # scaled = np.arcsinh(stretch_scale * scaled) / np.arcsinh(stretch_scale)


            elif stretch == 'sqrt':
                scaled = np.sqrt(scaled)

            grad = ndimage.gaussian_gradient_magnitude(scaled, sigma=1)
            score = np.nanmean(grad)

            if score > best_score:
                best_score = score
                best_params = (vmin, vmax)

        z1, z2 = best_params

    else:
        raise ValueError("method 必须是 'percentile'、'zscale'、'hybrid' 或 'detail'")

    # 归一化并保持原形状
    norm = Normalize(vmin=z1, vmax=z2)
    normalized_array = norm(img)  # 直接用原二维数组
    cmap = plt.get_cmap('gray')
    wave_array = cmap(normalized_array)
    wave_array = (wave_array[..., 0]*255).astype(np.uint8)

    return wave_array


def pred_vis(pred_img, save_dir, mode = 'percentile' ): # zscale hybrid
    pred_img = pred_img * space_std + space_mean
    mask = (pred_img == 0)
       
    pred_img_vis = fits_vis(pred_img, mode)
    plt.imshow(pred_img_vis, cmap='gray')  # gray  viridis
    plt.contour(mask, colors=[(237/255, 104/255, 113/255)], linewidths=0.5)
    plt.axis('off')
    plt.savefig(os.path.join(save_dir,'visualize_{}.png'.format(mode)), 
                dpi=500, bbox_inches='tight', pad_inches=0)

    plt.close()
