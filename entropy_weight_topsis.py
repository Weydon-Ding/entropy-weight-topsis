import numpy as np


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
    epsilon = 1e-6
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
    :return: 各方案的综合评价指数
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

    # 生成 Markdown 内容
    years = [2018, 2019, 2020, 2021, 2022, 2023]
    headers = ["年份"] + [f"指标{i + 1}" for i in range(Y.shape[1])]

    markdown_content = ""

    # 标准化并进行非零平移后的矩阵
    markdown_content += "### 标准化并进行非零平移后的矩阵\n"
    markdown_content += "| " + " | ".join(headers) + " |\n"
    markdown_content += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for i, year in enumerate(years):
        row = [str(year)] + [f"{val:.4f}" for val in Y[i]]
        markdown_content += "| " + " | ".join(row) + " |\n"

    # 指标所占比重（p）
    markdown_content += "\n### 指标所占比重（p）\n"
    markdown_content += "| " + " | ".join(headers) + " |\n"
    markdown_content += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for i, year in enumerate(years):
        row = [str(year)] + [f"{val:.4f}" for val in p[i]]
        markdown_content += "| " + " | ".join(row) + " |\n"

    # 熵值
    markdown_content += "\n### 熵值\n"
    markdown_content += "| 统计指标 | " + " | ".join([f"指标{i + 1}" for i in range(len(E))]) + " |\n"
    markdown_content += "| --- | " + " | ".join(["---"] * len(E)) + " |\n"
    markdown_content += "| 熵值 | " + " | ".join([f"{val:.4f}" for val in E]) + " |\n"

    # 差异化系数（1 - 熵值）
    markdown_content += "\n### 差异化系数（1 - 熵值）\n"
    markdown_content += "| 统计指标 | " + " | ".join([f"指标{i + 1}" for i in range(len(diff_coeff))]) + " |\n"
    markdown_content += "| --- | " + " | ".join(["---"] * len(diff_coeff)) + " |\n"
    markdown_content += "| 差异化系数 | " + " | ".join([f"{val:.4f}" for val in diff_coeff]) + " |\n"

    # 熵权（指标的权重 w）
    markdown_content += "\n### 熵权（指标的权重 w）\n"
    markdown_content += "| 统计指标 | " + " | ".join([f"指标{i + 1}" for i in range(len(w))]) + " |\n"
    markdown_content += "| --- | " + " | ".join(["---"] * len(w)) + " |\n"
    markdown_content += "| 熵权 | " + " | ".join([f"{val:.4f}" for val in w]) + " |\n"

    # 正理想解
    markdown_content += "\n### 正理想解\n"
    markdown_content += "| 统计指标 | " + " | ".join([f"指标{i + 1}" for i in range(len(Z_plus))]) + " |\n"
    markdown_content += "| --- | " + " | ".join(["---"] * len(Z_plus)) + " |\n"
    markdown_content += "| 正理想解 | " + " | ".join([f"{val:.4f}" for val in Z_plus]) + " |\n"

    # 负理想解
    markdown_content += "\n### 负理想解\n"
    markdown_content += "| 统计指标 | " + " | ".join([f"指标{i + 1}" for i in range(len(Z_minus))]) + " |\n"
    markdown_content += "| --- | " + " | ".join(["---"] * len(Z_minus)) + " |\n"
    markdown_content += "| 负理想解 | " + " | ".join([f"{val:.4f}" for val in Z_minus]) + " |\n"

    # 各方案与正理想解的欧氏距离
    markdown_content += "\n### 各方案与正理想解的欧氏距离\n"
    markdown_content += "| 年份 | " + " | ".join([str(year) for year in years]) + " |\n"
    markdown_content += "| --- | " + " | ".join(["---"] * len(years)) + " |\n"
    markdown_content += "| 欧氏距离 | " + " | ".join([f"{val:.4f}" for val in d_plus]) + " |\n"

    # 各方案与负理想解的欧氏距离
    markdown_content += "\n### 各方案与负理想解的欧氏距离\n"
    markdown_content += "| 年份 | " + " | ".join([str(year) for year in years]) + " |\n"
    markdown_content += "| --- | " + " | ".join(["---"] * len(years)) + " |\n"
    markdown_content += "| 欧氏距离 | " + " | ".join([f"{val:.4f}" for val in d_minus]) + " |\n"

    # 相对贴近度
    markdown_content += "\n### 相对贴近度\n"
    markdown_content += "| 年份 | " + " | ".join([str(year) for year in years]) + " |\n"
    markdown_content += "| --- | " + " | ".join(["---"] * len(years)) + " |\n"
    markdown_content += "| 相对贴近度 | " + " | ".join([f"{val:.4f}" for val in C]) + " |\n"

    # 对综合评价指数进行排名
    sorted_indices = np.argsort(C)[::-1]
    ranked_years = [years[i] for i in sorted_indices]
    ranked_scores = C[sorted_indices]

    # 输出排名结果
    markdown_content += "\n### 各年份综合评价指数排名\n"
    markdown_content += "| 排名 | 年份 | 综合评价指数 |\n"
    markdown_content += "| --- | --- | --- |\n"
    for i, (year, score) in enumerate(zip(ranked_years, ranked_scores), start=1):
        markdown_content += f"| {i} | {year} | {score:.4f} |\n"

    return markdown_content


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

# 调用主函数生成 Markdown 内容
markdown_content = entropy_weight_topsis(X, types)

# 保存 Markdown 文件
try:
    with open('topsis_result.md', 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    print("Markdown 文件已成功保存为 topsis_result.md")
except Exception as e:
    print(f"保存文件时出现错误: {e}")
