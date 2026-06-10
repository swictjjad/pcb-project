# -*- coding: utf-8 -*-
"""
统计显著性检验工具

所有"提升X%"的结论必须通过统计检验才有意义。
0.8%的mAP提升可能是随机种子不同导致的波动。

本模块提供：
1. Bootstrap检验：对多次实验的mAP进行重采样，估计置信区间和p-value
2. 配对t检验：同一测试集上两个模型的逐类mAP对比
3. Wilcoxon符号秩检验：非参数检验，不假设正态分布

使用方法：
    from utils.statistical_testing import bootstrap_test, paired_t_test

    # 方法1：多次实验的mAP列表
    baseline_maps = [0.845, 0.847, 0.843, 0.846, 0.844]  # 5次实验
    improved_maps = [0.853, 0.855, 0.851, 0.854, 0.852]  # 5次实验
    result = paired_t_test(baseline_maps, improved_maps)
    print(f"p-value: {result['p_value']:.4f}")  # < 0.05 才显著

    # 方法2：单次实验的逐类mAP
    baseline_per_class = {'open': 0.85, 'short': 0.90, ...}
    improved_per_class = {'open': 0.88, 'short': 0.92, ...}
    result = paired_t_test(
        list(baseline_per_class.values()),
        list(improved_per_class.values())
    )
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import stats as scipy_stats


def bootstrap_test(samples_a: List[float], samples_b: List[float],
                   n_bootstrap: int = 10000, confidence: float = 0.95,
                   alternative: str = 'two-sided') -> dict:
    """
    Bootstrap检验：估计两个样本均值差异的置信区间和p-value

    适用场景：多次实验的mAP对比（如5次不同随机种子的训练结果）

    Args:
        samples_a: 基线模型的多次实验结果
        samples_b: 改进模型的多次实验结果
        n_bootstrap: Bootstrap重采样次数
        confidence: 置信水平
        alternative: 'two-sided', 'greater', 'less'

    Returns:
        dict: 包含均值差异、置信区间、p-value、是否显著
    """
    a = np.array(samples_a)
    b = np.array(samples_b)
    observed_diff = b.mean() - a.mean()

    # Bootstrap重采样
    n_a, n_b = len(a), len(b)
    boot_diffs = []

    for _ in range(n_bootstrap):
        boot_a = np.random.choice(a, size=n_a, replace=True)
        boot_b = np.random.choice(b, size=n_b, replace=True)
        boot_diffs.append(boot_b.mean() - boot_a.mean())

    boot_diffs = np.array(boot_diffs)

    # 置信区间
    alpha = 1 - confidence
    if alternative == 'two-sided':
        ci_lower = np.percentile(boot_diffs, alpha / 2 * 100)
        ci_upper = np.percentile(boot_diffs, (1 - alpha / 2) * 100)
    elif alternative == 'greater':
        ci_lower = np.percentile(boot_diffs, alpha * 100)
        ci_upper = np.inf
    else:
        ci_lower = -np.inf
        ci_upper = np.percentile(boot_diffs, (1 - alpha) * 100)

    # p-value
    if alternative == 'two-sided':
        p_value = 2 * min(
            np.mean(boot_diffs <= 0),
            np.mean(boot_diffs >= 0)
        )
    elif alternative == 'greater':
        p_value = np.mean(boot_diffs <= 0)
    else:
        p_value = np.mean(boot_diffs >= 0)

    return {
        'observed_diff': float(observed_diff),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'confidence': confidence,
        'n_bootstrap': n_bootstrap,
        'mean_a': float(a.mean()),
        'mean_b': float(b.mean()),
        'std_a': float(a.std()),
        'std_b': float(b.std()),
    }


def paired_t_test(samples_a: List[float], samples_b: List[float],
                  alpha: float = 0.05) -> dict:
    """
    配对t检验：同一测试集上两个模型的对比

    适用场景：
    1. 多次实验的mAP对比
    2. 同一测试集上逐类mAP的对比

    前提假设：差异服从正态分布（样本量>=30时稳健）

    Args:
        samples_a: 基线模型结果
        samples_b: 改进模型结果
        alpha: 显著性水平

    Returns:
        dict: 包含t统计量、p-value、是否显著、效应量(Cohen's d)
    """
    a = np.array(samples_a)
    b = np.array(samples_b)

    assert len(a) == len(b), "样本量必须相同"

    diff = b - a

    # 配对t检验
    t_stat, p_value = scipy_stats.ttest_rel(b, a)

    # Cohen's d（效应量）
    d_mean = diff.mean()
    d_std = diff.std(ddof=1)
    cohens_d = d_mean / d_std if d_std > 0 else float('inf')

    # 效应量解释
    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"

    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'alpha': alpha,
        'mean_diff': float(d_mean),
        'std_diff': float(d_std),
        'cohens_d': float(cohens_d),
        'effect_size': effect_size,
        'mean_a': float(a.mean()),
        'mean_b': float(b.mean()),
        'n_samples': len(a),
    }


def wilcoxon_test(samples_a: List[float], samples_b: List[float],
                  alpha: float = 0.05) -> dict:
    """
    Wilcoxon符号秩检验：非参数检验

    适用场景：样本量小（<30）或差异不服从正态分布时
    不假设差异服从正态分布，更保守但更稳健。

    Args:
        samples_a: 基线模型结果
        samples_b: 改进模型结果
        alpha: 显著性水平

    Returns:
        dict: 包含W统计量、p-value、是否显著
    """
    a = np.array(samples_a)
    b = np.array(samples_b)
    diff = b - a

    # 去除零差异
    diff_nonzero = diff[diff != 0]
    if len(diff_nonzero) < 1:
        return {
            'W_statistic': 0,
            'p_value': 1.0,
            'significant': False,
            'note': '所有差异为零，无法检验'
        }

    W, p_value = scipy_stats.wilcoxon(b, a, alternative='greater')

    return {
        'W_statistic': float(W),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'alpha': alpha,
        'mean_diff': float(diff.mean()),
        'n_nonzero_diffs': len(diff_nonzero),
    }


def multi_run_comparison(results: Dict[str, List[float]],
                         baseline_name: str = 'baseline',
                         alpha: float = 0.05) -> Dict[str, dict]:
    """
    多模型对比：每个模型与基线进行配对t检验

    Args:
        results: {模型名: [多次实验mAP]}
        baseline_name: 基线模型名称
        alpha: 显著性水平

    Returns:
        {模型名: 检验结果}
    """
    baseline = results[baseline_name]
    comparisons = {}

    for name, values in results.items():
        if name == baseline_name:
            continue
        comparisons[name] = paired_t_test(baseline, values, alpha)

    return comparisons


def format_significance(result: dict) -> str:
    """格式化显著性检验结果为可读字符串"""
    if 't_statistic' in result:
        # t检验
        sig_mark = "*" if result['significant'] else ""
        return (
            f"差异={result['mean_diff']:.4f}, "
            f"t={result['t_statistic']:.3f}, "
            f"p={result['p_value']:.4f}{sig_mark}, "
            f"Cohen's d={result['cohens_d']:.3f}({result['effect_size']})"
        )
    elif 'W_statistic' in result:
        sig_mark = "*" if result['significant'] else ""
        return (
            f"差异={result['mean_diff']:.4f}, "
            f"W={result['W_statistic']:.1f}, "
            f"p={result['p_value']:.4f}{sig_mark}"
        )
    else:
        # Bootstrap
        sig_mark = "*" if result['significant'] else ""
        return (
            f"差异={result['observed_diff']:.4f}, "
            f"CI=[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}], "
            f"p={result['p_value']:.4f}{sig_mark}"
        )
