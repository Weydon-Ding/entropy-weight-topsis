import numpy as np
import pandas as pd


def data_standardization(X, types):
    """
    数据标准化
    :param X: 原始数据矩阵
    :param types: 指标类型列表，1 表示效益型，0 表示成本型
    :return: 标准化后的数据矩阵
    """
    n, m = X.shape
    Y = np.zeros((n, m))
    for j in range(m):
        if types[j] == 1:  # 效益型指标
            Y[:, j] = (X[:, j] - np.min(X[:, j])) / (np.max(X[:, j]) - np.min(X[:, j]))
        else:  # 成本型指标
            Y[:, j] = (np.max(X[:, j]) - X[:, j]) / (np.max(X[:, j]) - np.min(X[:, j]))
    return Y


def non_zero_translation(Y):
    """
    对标准化后的矩阵进行非零平移
    :param Y: 标准化后的矩阵
    :return: 非零平移后的矩阵
    """
    # 计算需要平移的量（最小值的相反数 + epsilon）
    min_val = np.min(Y)
    epsilon = 0.0001  # 平移的最小增量值
    shift = (-min_val + epsilon) if min_val <= 0 else 0.0
    return Y + shift


def calculate_entropy(Y):
    """
    计算熵值
    :param Y: 标准化后的数据矩阵
    :return: 各指标的熵值
    """
    n, m = Y.shape
    p = np.zeros((n, m))
    for j in range(m):
        p[:, j] = Y[:, j] / np.sum(Y[:, j])
    E = -1 / np.log(n) * np.sum(p * np.log(p), axis=0)
    return E, p


def determine_weights(E):
    """
    确定权重
    :param E: 各指标的熵值
    :return: 各指标的权重
    """
    w = (1 - E) / np.sum(1 - E)
    return w


def calculate_weighted_matrix(Y, w):
    """
    计算加权标准化矩阵
    :param Y: 标准化后的数据矩阵
    :param w: 各指标的权重
    :return: 加权标准化矩阵
    """
    Z = Y * w
    return Z


def determine_ideal_solutions(Z):
    """
    确定正理想解和负理想解
    :param Z: 加权标准化矩阵
    :return: 正理想解和负理想解
    """
    Z_plus = np.max(Z, axis=0)
    Z_minus = np.min(Z, axis=0)
    return Z_plus, Z_minus


def calculate_distances(Z, Z_plus, Z_minus):
    """
    计算各方案与正、负理想解的距离
    :param Z: 加权标准化矩阵
    :param Z_plus: 正理想解
    :param Z_minus: 负理想解
    :return: 各方案与正、负理想解的距离
    """
    d_plus = np.sqrt(np.sum((Z - Z_plus) ** 2, axis=1))
    d_minus = np.sqrt(np.sum((Z - Z_minus) ** 2, axis=1))
    return d_plus, d_minus


def calculate_comprehensive_index(d_plus, d_minus):
    """
    计算综合评价指数
    :param d_plus: 各方案与正理想解的距离
    :param d_minus: 各方案与负理想解的距离
    :return: 各方案的综合评价指数
    """
    C = d_minus / (d_plus + d_minus)
    return C


def entropy_weight_topsis(X, types):
    """
    熵权 TOPSIS 法主函数
    :param X: 原始数据矩阵
    :param types: 指标类型列表，1 表示效益型，0 表示成本型
    :return: 各方案的综合评价指数及相关结果
    """
    # 数据标准化
    Y = data_standardization(X, types)
    # 非零平移
    Y = non_zero_translation(Y)

    # 计算熵值和指标所占比重（p）
    E, p = calculate_entropy(Y)

    # 计算差异化系数（1 - 熵值）
    diff_coeff = 1 - E

    # 确定权重
    w = determine_weights(E)

    # 计算加权标准化矩阵
    Z = calculate_weighted_matrix(Y, w)

    # 确定正理想解和负理想解
    Z_plus, Z_minus = determine_ideal_solutions(Z)

    # 计算各方案与正、负理想解的距离（欧氏距离）
    d_plus, d_minus = calculate_distances(Z, Z_plus, Z_minus)

    # 计算综合评价指数（相对贴近度）
    C = calculate_comprehensive_index(d_plus, d_minus)

    return {
        '标准化矩阵': Y,
        '指标比重': p,
        '熵值': E,
        '差异化系数': diff_coeff,
        '权重': w,
        '正理想解': Z_plus,
        '负理想解': Z_minus,
        '正理想解距离': d_plus,
        '负理想解距离': d_minus,
        '综合评价指数': C
    }


# 原始数据
data = """
5.5976 3.8603 1.2053 2.3114 1.2053 2.1375
0.2487 0.2170 0.1788 0.1653 0.0440 -0.0215
0.7435 0.3445 -0.1824 0.2696 -0.6231 -0.4991
0.1852 0.2034 0.6125 0.6053 0.5456 0.6429
3.3684 3.1132 1.9549 1.6250 1.5680 1.6858
0.0025 0.0002 0.4110 0.3172 0.2157 0.3761
0.0334 0.0306 0.0041 0.0106 0.0159 0.0197
0.2428 0.4710 0.4639 0.2683 0.1568 0.1749
-0.0041 0.1094 0.0632 0.0877 0.0737 -0.0617
0.2000 0.1800 0.6000 0.6800 -0.2800 -1.2400
0.1940 -0.1730 0.3667 0.0871 -0.3727 -0.2161
0.2334 0.1822 0.5232 0.6219 0.8591 0.9632
0.0000 0.0000 0.3879 0.3355 0.2998 0.2440
0.0529 0.0320 0.0127 0.0100 0.0002 0.0008
0.0194 0.0476 0.0347 0.0299 0.0504 0.0739
0.2373 0.2625 0.3615 0.1217 0.1334 0.1336
0.0000 0.0000 0.0074 0.0113 0.0119 0.0097
"""

# 将数据转换为 numpy 数组并转置
lines = data.strip().split('\n')
X = np.array([[float(val.replace(',', '')) for val in line.split()] for line in lines]).T

# 指标类型列表，1 表示效益型，0 表示成本型
types = [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
# 检查指标数量是否匹配
if X.shape[1] != len(types):
    raise ValueError(f"指标数量不匹配：数据有{X.shape[1]}个指标，types有{len(types)}个类型")

years = [2018, 2019, 2020, 2021, 2022, 2023]

# 格式：(start_index, end_index)，表示包含从start到end的指标（含两端）
groups = [
    (0, 2),
    (3, 6),
    (7, 9),
    (10, 12),
    (13, 16),
]
# 确保分组覆盖所有指标
all_indices = set()
for start, end in groups:
    all_indices.update(range(start, end+1))
if all_indices != set(range(len(types))):
    raise ValueError("分组未覆盖所有指标，请检查分组索引")


# 创建 ExcelWriter 对象
with pd.ExcelWriter('topsis_results_all.xlsx') as writer:
    # 计算 15 个指标整体的结果
    all_results = entropy_weight_topsis(X, types)
    headers = [f"指标{i + 1}" for i in range(X.shape[1])]

    # 标准化并进行非零平移后的矩阵
    df_Y = pd.DataFrame(all_results['标准化矩阵'].T, index=headers, columns=years)
    df_Y.index.name = '指标'
    df_Y.to_excel(writer, sheet_name='全部指标_标准化矩阵')

    # 指标所占比重（p）
    df_p = pd.DataFrame(all_results['指标比重'].T, index=headers, columns=years)
    df_p.index.name = '指标'
    df_p.to_excel(writer, sheet_name='全部指标_指标比重')

    # 熵值
    df_E = pd.DataFrame(all_results['熵值'], index=headers, columns=['熵值'])
    df_E.index.name = '指标'
    df_E.to_excel(writer, sheet_name='全部指标_熵值')

    # 差异化系数（1 - 熵值）
    df_diff = pd.DataFrame(all_results['差异化系数'], index=headers, columns=['差异化系数'])
    df_diff.index.name = '指标'
    df_diff.to_excel(writer, sheet_name='全部指标_差异化系数')

    # 熵权（指标的权重 w）
    df_w = pd.DataFrame(all_results['权重'], index=headers, columns=['熵权'])
    df_w.index.name = '指标'
    df_w.to_excel(writer, sheet_name='全部指标_熵权')

    # 正理想解
    df_Z_plus = pd.DataFrame(all_results['正理想解'], index=headers, columns=['正理想解'])
    df_Z_plus.index.name = '指标'
    df_Z_plus.to_excel(writer, sheet_name='全部指标_正理想解')

    # 负理想解
    df_Z_minus = pd.DataFrame(all_results['负理想解'], index=headers, columns=['负理想解'])
    df_Z_minus.index.name = '指标'
    df_Z_minus.to_excel(writer, sheet_name='全部指标_负理想解')

    # 各方案与正理想解的欧氏距离
    df_d_plus = pd.DataFrame(all_results['正理想解距离'], index=years, columns=['欧氏距离'])
    df_d_plus.index.name = '年份'
    df_d_plus.to_excel(writer, sheet_name='全部指标_正理想解距离')

    # 各方案与负理想解的欧氏距离
    df_d_minus = pd.DataFrame(all_results['负理想解距离'], index=years, columns=['欧氏距离'])
    df_d_minus.index.name = '年份'
    df_d_minus.to_excel(writer, sheet_name='全部指标_负理想解距离')

    # 相对贴近度
    df_C = pd.DataFrame(all_results['综合评价指数'], index=years, columns=['相对贴近度'])
    df_C.index.name = '年份'
    df_C.to_excel(writer, sheet_name='全部指标_相对贴近度')

    # 对综合评价指数进行排名
    sorted_indices = np.argsort(all_results['综合评价指数'])[::-1]
    ranked_years = [years[i] for i in sorted_indices]
    ranked_scores = all_results['综合评价指数'][sorted_indices]
    ranking_df = pd.DataFrame({
        '排名': range(1, len(ranked_years) + 1),
        '年份': ranked_years,
        '综合评价指数': ranked_scores
    })
    ranking_df.to_excel(writer, sheet_name='全部指标_排名', index=False)

    # 按自定义分组处理每组数据
    for group_idx, (start, end) in enumerate(groups, 1):  # group_idx从1开始计数
        # 截取当前组的指标数据和类型（注意切片是左闭右开，需end+1）
        X_group = X[:, start:end+1]
        types_group = types[start:end+1]
        # 生成当前组的指标名称（如指标1-2、指标3-7等）
        headers = [f"指标{k + 1}" for k in range(start, end+1)]

        # 调用熵权TOPSIS算法计算组内结果
        group_results = entropy_weight_topsis(X_group, types_group)

        # 标准化并进行非零平移后的矩阵
        df_Y = pd.DataFrame(group_results['标准化矩阵'].T, index=headers, columns=years)
        df_Y.index.name = '指标'
        df_Y.to_excel(writer, sheet_name=f'组{group_idx}_标准化矩阵')

        # 指标所占比重（p）
        df_p = pd.DataFrame(group_results['指标比重'].T, index=headers, columns=years)
        df_p.index.name = '指标'
        df_p.to_excel(writer, sheet_name=f'组{group_idx}_指标比重')

        # 熵值
        df_E = pd.DataFrame(group_results['熵值'], index=headers, columns=['熵值'])
        df_E.index.name = '指标'
        df_E.to_excel(writer, sheet_name=f'组{group_idx}_熵值')

        # 差异化系数（1 - 熵值）
        df_diff = pd.DataFrame(group_results['差异化系数'], index=headers, columns=['差异化系数'])
        df_diff.index.name = '指标'
        df_diff.to_excel(writer, sheet_name=f'组{group_idx}_差异化系数')

        # 熵权（指标的权重 w）
        df_w = pd.DataFrame(group_results['权重'], index=headers, columns=['熵权'])
        df_w.index.name = '指标'
        df_w.to_excel(writer, sheet_name=f'组{group_idx}_熵权')

        # 正理想解
        df_Z_plus = pd.DataFrame(group_results['正理想解'], index=headers, columns=['正理想解'])
        df_Z_plus.index.name = '指标'
        df_Z_plus.to_excel(writer, sheet_name=f'组{group_idx}_正理想解')

        # 负理想解
        df_Z_minus = pd.DataFrame(group_results['负理想解'], index=headers, columns=['负理想解'])
        df_Z_minus.index.name = '指标'
        df_Z_minus.to_excel(writer, sheet_name=f'组{group_idx}_负理想解')

        # 各方案与正理想解的欧氏距离
        df_d_plus = pd.DataFrame(group_results['正理想解距离'], index=years, columns=['欧氏距离'])
        df_d_plus.index.name = '年份'
        df_d_plus.to_excel(writer, sheet_name=f'组{group_idx}_正理想解距离')

        # 各方案与负理想解的欧氏距离
        df_d_minus = pd.DataFrame(group_results['负理想解距离'], index=years, columns=['欧氏距离'])
        df_d_minus.index.name = '年份'
        df_d_minus.to_excel(writer, sheet_name=f'组{group_idx}_负理想解距离')

        # 相对贴近度
        df_C = pd.DataFrame(group_results['综合评价指数'], index=years, columns=['相对贴近度'])
        df_C.index.name = '年份'
        df_C.to_excel(writer, sheet_name=f'组{group_idx}_相对贴近度')

        # 对综合评价指数进行排名
        sorted_indices = np.argsort(group_results['综合评价指数'])[::-1]
        ranked_years = [years[i] for i in sorted_indices]
        ranked_scores = group_results['综合评价指数'][sorted_indices]
        ranking_df = pd.DataFrame({
            '排名': range(1, len(ranked_years) + 1),
            '年份': ranked_years,
            '综合评价指数': ranked_scores
        })
        ranking_df.to_excel(writer, sheet_name=f'组{group_idx}_排名', index=False)

print("结果已成功保存为 topsis_results_all.xlsx。")
