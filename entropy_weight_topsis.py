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
0.024	0.0205	0.0871	0.0779	-0.0173	-0.1092
0.0289	0.0254	0.1459	0.1633	-0.0658	-0.309
0.0263	0.0225	0.1061	0.1217	-0.0374	-0.2565
3.3684	3.1132	1.9549	1.625	1.568	1.6858
2.2791	2.2145	1.6351	1.3475	1.3887	1.502
0.1852	0.2034	0.6125	0.6053	0.5456	0.6429
0.9136	0.9088	0.8216	0.6401	0.4637	0.4259
3.9912	4.2404	4.8013	4.2197	4.0514	4.9715
4.9615	4.9259	3.046	1.7867	1.144	1.0163
0.199	0.0427	0.3655	0.0517	-0.3005	-0.145
0.0541	0.0427	0.9591	0.039	-0.1052	-0.0289
-0.4089	-0.108	5.433	0.2071	-1.2148	-4.8869
0.1864	0.0531	0.2267	0.0013	0.0696	-0.1393
0.05291	0.03201	0.01273	0.00997	0.0024	0.00075
0.2373	0.2625	0.3615	0.1217	0.1334	0.1336
"""

# 将数据转换为 numpy 数组并转置
lines = data.strip().split('\n')
X = np.array([[float(val.replace(',', '')) for val in line.split('\t')] for line in lines]).T

# 指标类型列表，1 表示效益型，0 表示成本型
types = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]

# 每三行作为一组进行处理
num_groups = len(types) // 3
years = [2018, 2019, 2020, 2021, 2022, 2023]

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

    # 计算每组（一级指标）的结果
    for i in range(num_groups):
        start_index = i * 3
        end_index = start_index + 3
        X_group = X[:, start_index:end_index]
        types_group = types[start_index:end_index]
        headers = [f"指标{start_index + k + 1}" for k in range(3)]

        group_results = entropy_weight_topsis(X_group, types_group)

        # 标准化并进行非零平移后的矩阵
        df_Y = pd.DataFrame(group_results['标准化矩阵'].T, index=headers, columns=years)
        df_Y.index.name = '指标'
        df_Y.to_excel(writer, sheet_name=f'组{i + 1}_标准化矩阵')

        # 指标所占比重（p）
        df_p = pd.DataFrame(group_results['指标比重'].T, index=headers, columns=years)
        df_p.index.name = '指标'
        df_p.to_excel(writer, sheet_name=f'组{i + 1}_指标比重')

        # 熵值
        df_E = pd.DataFrame(group_results['熵值'], index=headers, columns=['熵值'])
        df_E.index.name = '指标'
        df_E.to_excel(writer, sheet_name=f'组{i + 1}_熵值')

        # 差异化系数（1 - 熵值）
        df_diff = pd.DataFrame(group_results['差异化系数'], index=headers, columns=['差异化系数'])
        df_diff.index.name = '指标'
        df_diff.to_excel(writer, sheet_name=f'组{i + 1}_差异化系数')

        # 熵权（指标的权重 w）
        df_w = pd.DataFrame(group_results['权重'], index=headers, columns=['熵权'])
        df_w.index.name = '指标'
        df_w.to_excel(writer, sheet_name=f'组{i + 1}_熵权')

        # 正理想解
        df_Z_plus = pd.DataFrame(group_results['正理想解'], index=headers, columns=['正理想解'])
        df_Z_plus.index.name = '指标'
        df_Z_plus.to_excel(writer, sheet_name=f'组{i + 1}_正理想解')

        # 负理想解
        df_Z_minus = pd.DataFrame(group_results['负理想解'], index=headers, columns=['负理想解'])
        df_Z_minus.index.name = '指标'
        df_Z_minus.to_excel(writer, sheet_name=f'组{i + 1}_负理想解')

        # 各方案与正理想解的欧氏距离
        df_d_plus = pd.DataFrame(group_results['正理想解距离'], index=years, columns=['欧氏距离'])
        df_d_plus.index.name = '年份'
        df_d_plus.to_excel(writer, sheet_name=f'组{i + 1}_正理想解距离')

        # 各方案与负理想解的欧氏距离
        df_d_minus = pd.DataFrame(group_results['负理想解距离'], index=years, columns=['欧氏距离'])
        df_d_minus.index.name = '年份'
        df_d_minus.to_excel(writer, sheet_name=f'组{i + 1}_负理想解距离')

        # 相对贴近度
        df_C = pd.DataFrame(group_results['综合评价指数'], index=years, columns=['相对贴近度'])
        df_C.index.name = '年份'
        df_C.to_excel(writer, sheet_name=f'组{i + 1}_相对贴近度')

        # 对综合评价指数进行排名
        sorted_indices = np.argsort(group_results['综合评价指数'])[::-1]
        ranked_years = [years[i] for i in sorted_indices]
        ranked_scores = group_results['综合评价指数'][sorted_indices]
        ranking_df = pd.DataFrame({
            '排名': range(1, len(ranked_years) + 1),
            '年份': ranked_years,
            '综合评价指数': ranked_scores
        })
        ranking_df.to_excel(writer, sheet_name=f'组{i + 1}_排名', index=False)

print("结果已成功保存为 topsis_results_all.xlsx。")
