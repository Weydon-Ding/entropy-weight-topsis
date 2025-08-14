import numpy as np
import pandas as pd
import yaml  # 需要安装pyyaml：pip install pyyaml


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


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # 加载配置
    config = load_config('config.yaml')

    # 解析配置数据
    data_str = config['raw_data'].strip()
    lines = data_str.split('\n')
    X = np.array([[float(val.replace(',', '')) for val in line.split()] for line in lines]).T

    types = config['indicator_types']
    years = config['years']
    groups = config['groups']
    output_file = config['output_file']

    # 校验数据
    if X.shape[1] != len(types):
        raise ValueError(f"指标数量不匹配：数据有{X.shape[1]}个指标，types有{len(types)}个类型")

    all_indices = set()
    for start, end in groups:
        all_indices.update(range(start, end+1))
    if all_indices != set(range(len(types))):
        raise ValueError("分组未覆盖所有指标，请检查分组索引")

    # 生成结果
    with pd.ExcelWriter(output_file) as writer:
        # 全部指标结果
        all_results = entropy_weight_topsis(X, types)
        headers = [f"指标{i + 1}" for i in range(X.shape[1])]

        # 写入全部指标相关表格（省略重复代码，保持与原逻辑一致）
        pd.DataFrame(all_results['标准化矩阵'].T, index=headers, columns=years).to_excel(writer, sheet_name='全部指标_标准化矩阵')
        pd.DataFrame(all_results['指标比重'].T, index=headers, columns=years).to_excel(writer, sheet_name='全部指标_指标比重')
        pd.DataFrame(all_results['熵值'], index=headers, columns=['熵值']).to_excel(writer, sheet_name='全部指标_熵值')
        pd.DataFrame(all_results['差异化系数'], index=headers, columns=['差异化系数']).to_excel(writer, sheet_name='全部指标_差异化系数')
        pd.DataFrame(all_results['权重'], index=headers, columns=['熵权']).to_excel(writer, sheet_name='全部指标_熵权')
        pd.DataFrame(all_results['正理想解'], index=headers, columns=['正理想解']).to_excel(writer, sheet_name='全部指标_正理想解')
        pd.DataFrame(all_results['负理想解'], index=headers, columns=['负理想解']).to_excel(writer, sheet_name='全部指标_负理想解')
        pd.DataFrame(all_results['正理想解距离'], index=years, columns=['欧氏距离']).to_excel(writer, sheet_name='全部指标_正理想解距离')
        pd.DataFrame(all_results['负理想解距离'], index=years, columns=['欧氏距离']).to_excel(writer, sheet_name='全部指标_负理想解距离')
        pd.DataFrame(all_results['综合评价指数'], index=years, columns=['相对贴近度']).to_excel(writer, sheet_name='全部指标_相对贴近度')

        # 排名
        sorted_indices = np.argsort(all_results['综合评价指数'])[::-1]
        ranking_df = pd.DataFrame({
            '排名': range(1, len(years) + 1),
            '年份': [years[i] for i in sorted_indices],
            '综合评价指数': all_results['综合评价指数'][sorted_indices]
        })
        ranking_df.to_excel(writer, sheet_name='全部指标_排名', index=False)

        # 分组指标结果
        for group_idx, (start, end) in enumerate(groups, 1):
            X_group = X[:, start:end+1]
            types_group = types[start:end+1]
            headers_group = [f"指标{k + 1}" for k in range(start, end+1)]
            group_results = entropy_weight_topsis(X_group, types_group)

            # 写入分组相关表格（省略重复代码）
            pd.DataFrame(group_results['标准化矩阵'].T, index=headers_group, columns=years).to_excel(writer, sheet_name=f'组{group_idx}_标准化矩阵')
            pd.DataFrame(group_results['指标比重'].T, index=headers_group, columns=years).to_excel(writer, sheet_name=f'组{group_idx}_指标比重')
            pd.DataFrame(group_results['熵值'], index=headers_group, columns=['熵值']).to_excel(writer, sheet_name=f'组{group_idx}_熵值')
            pd.DataFrame(group_results['差异化系数'], index=headers_group, columns=['差异化系数']).to_excel(writer, sheet_name=f'组{group_idx}_差异化系数')
            pd.DataFrame(group_results['权重'], index=headers_group, columns=['熵权']).to_excel(writer, sheet_name=f'组{group_idx}_熵权')
            pd.DataFrame(group_results['正理想解'], index=headers_group, columns=['正理想解']).to_excel(writer, sheet_name=f'组{group_idx}_正理想解')
            pd.DataFrame(group_results['负理想解'], index=headers_group, columns=['负理想解']).to_excel(writer, sheet_name=f'组{group_idx}_负理想解')
            pd.DataFrame(group_results['正理想解距离'], index=years, columns=['欧氏距离']).to_excel(writer, sheet_name=f'组{group_idx}_正理想解距离')
            pd.DataFrame(group_results['负理想解距离'], index=years, columns=['欧氏距离']).to_excel(writer, sheet_name=f'组{group_idx}_负理想解距离')
            pd.DataFrame(group_results['综合评价指数'], index=years, columns=['相对贴近度']).to_excel(writer, sheet_name=f'组{group_idx}_相对贴近度')

            # 分组排名
            sorted_group_indices = np.argsort(group_results['综合评价指数'])[::-1]
            group_ranking_df = pd.DataFrame({
                '排名': range(1, len(years) + 1),
                '年份': [years[i] for i in sorted_group_indices],
                '综合评价指数': group_results['综合评价指数'][sorted_group_indices]
            })
            group_ranking_df.to_excel(writer, sheet_name=f'组{group_idx}_排名', index=False)

    print(f"结果已成功保存为 {output_file}。")


if __name__ == "__main__":
    main()
