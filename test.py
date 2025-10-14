import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def simulate_distances(dimensions, n_points=500, seed=42):
    """模拟不同维度下的点间距离分布"""
    np.random.seed(seed)
    results = {}
    for d in dimensions:
        data = np.random.rand(n_points, d)
        q = np.random.rand(1, d)
        distances = np.linalg.norm(data - q, axis=1)
        results[d] = distances
    return results

def plot_all(results):
    """综合绘制：分布直方图 + min/max/mean + 最近邻比值"""
    dims = sorted(results.keys())
    min_d, max_d, mean_d, ratio = [], [], [], []

    for d in dims:
        dist = results[d]
        min_d.append(dist.min())
        max_d.append(dist.max())
        mean_d.append(dist.mean())
        ratio.append(dist.min() / dist.mean())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#17becf"]

    # 子图1：不同维度的距离分布
    for i, (d, dist) in enumerate(results.items()):
        axes[0, 0].hist(dist, bins=30, alpha=0.6, color=colors[i % len(colors)], label=f"{d}维")
    axes[0, 0].set_title("不同维度下点到查询点的距离分布", fontsize=14)
    axes[0, 0].set_xlabel("欧氏距离", fontsize=12)
    axes[0, 0].set_ylabel("频数", fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)

    # 子图2：最小/最大/平均距离随维度变化
    axes[0, 1].plot(dims, min_d, "o-", color="#1f77b4", linewidth=2, markersize=6, label="最近距离")
    axes[0, 1].plot(dims, max_d, "s-", color="#ff7f0e", linewidth=2, markersize=6, label="最远距离")
    axes[0, 1].plot(dims, mean_d, "^-", color="#2ca02c", linewidth=2, markersize=6, label="平均距离")
    axes[0, 1].set_title("维度灾难：距离随维度变化趋势", fontsize=14)
    axes[0, 1].set_xlabel("维度", fontsize=12)
    axes[0, 1].set_ylabel("距离", fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)

    # 子图3：最近邻/平均距离比值
    axes[1, 0].plot(dims, ratio, "d-", color="red", linewidth=2, markersize=6, label="比值")
    axes[1, 0].axhline(1, color="gray", linestyle="--", linewidth=1)  # 参考线 y=1
    axes[1, 0].set_title("维度灾难：最近邻与平均点几乎一样远", fontsize=14)
    axes[1, 0].set_xlabel("维度", fontsize=12)
    axes[1, 0].set_ylabel("最近邻 / 平均距离", fontsize=12)
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)
    axes[1, 0].legend()

    # 子图4：文字解释
    axes[1, 1].axis("off")
    explanation = (
        "直观结论：\n"
        "1. 随着维度升高，所有点到查询点的距离都在增大；\n"
        "2. 最近邻距离逐渐接近平均距离；\n"
        "3. 高维空间里点之间差别越来越小，距离失去区分能力。\n\n"
        " 这就是维度灾难的本质"
    )
    axes[1, 1].text(0.05, 0.5, explanation, fontsize=13, va="center")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dimensions = [2, 10, 50, 100, 500, 1000]
    results = simulate_distances(dimensions, n_points=1000)
    plot_all(results)
