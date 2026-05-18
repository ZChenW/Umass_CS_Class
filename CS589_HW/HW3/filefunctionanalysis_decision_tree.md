# HW3 `code` 文件夹函数逐一解析

## 目录说明

- 代码目录：`HW3_CMPSCI_589_Spring2026_Supporting_Files/code`
- 文件列表：
  - `decision_tree.py`
  - `random_forest.py`
  - `train.py`

---

## `decision_tree.py`

### 文件职责

- 实现一个基于信息增益的分类决策树。
- 支持数值特征与离散特征两种分裂方式。
- 提供训练、单样本预测、批量预测、准确率评估功能。

### 类：`DecisionTree`

### 1) `__init__(self, max_lenght=None, min_samples_split=2, min_gain=1e-6, node_chose_limit=None, random_seed=None)`

- 作用：初始化决策树的超参数、随机数生成器和树结构容器。
- 逻辑：
  1. 保存最大深度、最小分裂样本数、最小信息增益、候选特征数量上限。
  2. 使用 `np.random.RandomState(random_seed)` 建立可复现随机源。
  3. 初始化 `self.tree = None`，等待训练后写入树结构。
  4. 用 `self._number` 封装数值型判断函数。
- 参数设置：
  - `max_lenght`：树的最大深度，`None` 表示不限制。
  - `min_samples_split`：当前节点样本数小于该值时停止分裂。
  - `min_gain`：最佳特征的信息增益低于该值时停止分裂。
  - `node_chose_limit`：当前节点最多随机挑多少个特征参与最佳特征选择。
  - `random_seed`：随机种子。
- 返回值：无显式返回。

### 2) `InformationEntropy(self, data)`

- 作用：计算数据集标签列的信息熵。
- 逻辑：
  1. 默认最后一列 `data.iloc[:, -1]` 是标签。
  2. 用 `value_counts()` 统计各类别频率。
  3. 计算每个类别概率。
  4. 返回 `-sum(p * log2(p))`。
- 参数设置：
  - `data`：`pandas.DataFrame`，最后一列必须是标签。
- 返回值：`float`，信息熵。
- 实现细节：
  - 该函数写成一层嵌套 `lambda`，功能正确，但可读性较差。

### 3) `is_numeric(self, data)`

- 作用：判断一列数据是否为数值型。
- 逻辑：直接调用 `pd.api.types.is_numeric_dtype`。
- 参数设置：
  - `data`：通常是一列 `Series`。
- 返回值：`bool`。

### 4) `NumericInformationGain(self, data, a)`

- 作用：计算数值特征 `a` 的信息增益，并返回用于切分的阈值。
- 逻辑：
  1. 先算当前数据集总熵。
  2. 将特征 `a` 的均值作为阈值。
  3. 划分为 `<= threshold` 和 `> threshold` 两部分。
  4. 若任一子集为空，则返回 `(0.0, threshold)`。
  5. 否则按加权子熵求出信息增益。
- 参数设置：
  - `data`：训练子集。
  - `a`：数值特征列名。
- 返回值：二元组 `(gain, threshold)`。
- 实现细节：
  - 该实现只使用“均值阈值”，并没有遍历所有候选切分点，因此是一个简化版决策树。

### 5) `InformationGain(self, data, a)`

- 作用：统一计算特征 `a` 的信息增益，自动区分数值特征和离散特征。
- 逻辑：
  1. 如果 `a` 是数值型，则调用 `NumericInformationGain`。
  2. 如果 `a` 是离散型，则：
     - 计算当前整体熵。
     - 遍历该特征的每个取值。
     - 计算每个子集的熵并按比例加权求和。
     - 用整体熵减去加权子熵得到信息增益。
  3. 将结果包装成统一结构返回。
- 参数设置：
  - `data`：当前节点数据。
  - `a`：候选特征名。
- 返回值：`dict`，包含：
  - `gain`：信息增益
  - `feature`：特征名
  - `threshold`：数值型特征的阈值，离散型为 `None`
  - `is_numeric`：是否数值型

### 6) `GetBestFeature(self, data)`

- 作用：在当前节点中选出信息增益最大的分裂特征。
- 逻辑：
  1. 取除最后一列标签外的全部特征。
  2. 如果设置了 `node_chose_limit`，且特征数更多，则随机抽取部分特征。
  3. 遍历候选特征，调用 `InformationGain` 计算每个特征的信息增益。
  4. 保留增益最大的结果。
- 参数设置：
  - `data`：当前节点数据。
- 返回值：最佳特征信息字典。
- 实现细节：
  - 这里的随机抽样就是随机森林中“随机子空间”的核心来源之一。

### 7) `SpliteByFeatureCate(self, data, bestfeature)`

- 作用：按离散特征划分子集，并从子集中删除该特征列。
- 逻辑：
  1. 找出离散特征的所有唯一取值。
  2. 为每个取值提取一个子数据集。
  3. 对每个子数据集删除当前已经使用过的离散特征。
- 参数设置：
  - `data`：当前节点数据。
  - `bestfeature`：被选中的离散特征。
- 返回值：列表 `[(取值, 子集), ...]`。
- 实现细节：
  - 删除该离散特征是为了避免后续递归重复使用同一列。

### 8) `SpliteByFeatureNum(self, data, bestfeature, threshold)`

- 作用：按数值特征和阈值进行二叉分裂。
- 逻辑：
  1. `<= threshold` 进入左子集。
  2. `> threshold` 进入右子集。
  3. 数值特征不删除，允许后续继续使用该特征。
- 参数设置：
  - `data`：当前节点数据。
  - `bestfeature`：数值特征名。
  - `threshold`：分裂阈值。
- 返回值：`{"leq": left_df, "gt": right_df}`。

### 9) `get_most_label(self, data)`

- 作用：返回数据中出现次数最多的标签。
- 逻辑：
  1. 提取最后一列标签。
  2. 用 `value_counts(sort=True)` 统计频数。
  3. 返回频数最高的标签。
- 参数设置：
  - `data`：含标签数据集。
- 返回值：多数类标签。
- 备注：
  - 该函数在当前代码中没有被实际调用。
  - `getTree` 中直接使用了 `x.idxmax()` 实现相同逻辑。

### 10) `getTree(self, data, depth=0)`

- 作用：递归构建整棵决策树，是训练阶段的核心函数。
- 逻辑：
  1. 统计当前节点标签分布。
  2. 检查停止条件：
     - 当前节点标签已纯净。
     - 只剩最后一列标签，没有可用特征。
     - 达到最大深度。
     - 当前节点样本数不足 `min_samples_split`。
  3. 调用 `GetBestFeature` 选择最佳分裂特征。
  4. 若最佳增益低于 `min_gain`，则停止并返回叶节点。
  5. 构造内部节点字典，保存当前节点多数类、分裂特征、阈值、子节点容器等信息。
  6. 如果最佳特征是数值型：
     - 调用 `SpliteByFeatureNum` 划分左右子树。
     - 若任一侧为空，则退化成叶节点。
     - 分别递归构造左右子树。
  7. 如果最佳特征是离散型：
     - 调用 `SpliteByFeatureCate` 生成多个分支。
     - 为每个特征取值递归生成子树。
- 参数设置：
  - `data`：当前递归子问题的数据集。
  - `depth`：当前节点深度，根节点为 `0`。
- 返回值：
  - 叶节点：`{"type": "leaf", "prediction": 类别}`
  - 内部节点：包含 `prediction`、`feature`、`is_numeric`、`threshold`、`children`
- 实现细节：
  - 内部节点保留一个 `prediction` 字段，表示当前节点的多数类，用于预测阶段未知情况时兜底。

### 11) `fit(self, train)`

- 作用：训练决策树模型。
- 逻辑：调用 `getTree(train, depth=0)` 并将返回结果保存到 `self.tree`。
- 参数设置：
  - `train`：训练数据，默认最后一列为标签。
- 返回值：`self`。

### 12) `predit_one(self, a)`

- 作用：对单个样本进行预测。
- 逻辑：
  1. 从根节点开始向下遍历。
  2. 若当前节点是数值型分裂：
     - 与阈值比较后走 `leq` 或 `gt`。
     - 若下一节点为空，则返回当前节点多数类。
  3. 若当前节点是离散型分裂：
     - 如果样本取值不在已有分支中，返回当前节点多数类。
     - 否则进入对应子节点。
  4. 最终到达叶节点后返回预测类别。
- 参数设置：
  - `a`：单个样本，通常是一行 `Series`。
- 返回值：预测标签。
- 实现细节：
  - 函数名拼写为 `predit_one`，源码如此。
  - 对“测试集出现训练集中未见过的离散取值”提供了兜底逻辑。

### 13) `predit(self, dp)`

- 作用：对一个数据集做批量预测。
- 逻辑：
  1. 逐行遍历特征数据。
  2. 对每一行调用 `predit_one`。
  3. 收集所有预测结果。
- 参数设置：
  - `dp`：仅包含特征列的 `DataFrame`。
- 返回值：`list`，每个样本一个预测结果。
- 实现细节：
  - 函数名拼写为 `predit`，源码如此。
  - 使用 `iterrows()`，实现简单但速度一般。

### 14) `score(self, dp)`

- 作用：在带标签的数据集上计算准确率。
- 逻辑：
  1. 默认最后一列是标签列。
  2. 拆分出特征 `X` 和真实标签 `y`。
  3. 调用 `predit(X)` 得到预测结果。
  4. 返回预测正确的比例。
- 参数设置：
  - `dp`：完整数据集，最后一列为标签。
- 返回值：`float`，准确率。

### 文件级总结

- 这是一个手写版分类决策树实现。
- 数值特征采用“均值阈值 + 二叉切分”。
- 离散特征采用“按类别多叉切分 + 子节点删除该特征”。
- 训练停止条件完整，预测阶段也对异常路径提供了多数类回退。

---

## `random_forest.py`

### 文件职责

- 在 `DecisionTree` 的基础上实现随机森林。
- 通过 bootstrap 抽样和节点级随机特征选择提升泛化能力。

### 类：`RandomForest`

### 1) `__init__(self, n_trees=10, max_lenght=None, min_samples_split=2, min_gain=1e-6, node_chose_limit=None, random_seed=None)`

- 作用：初始化随机森林超参数和树容器。
- 逻辑：
  1. 保存树数量和每棵树共享的训练超参数。
  2. 构造一个 `RandomState` 作为森林层级的随机源。
  3. 初始化 `self.trees = []`。
- 参数设置：
  - `n_trees`：树的数量。
  - `max_lenght`：每棵树的最大深度。
  - `min_samples_split`：每棵树的最小分裂样本数。
  - `min_gain`：每棵树最小信息增益阈值。
  - `node_chose_limit`：每个节点候选特征数上限。
  - `random_seed`：森林随机种子。
- 返回值：无显式返回。

### 2) `fit(self, df)`

- 作用：训练整片随机森林。
- 逻辑：
  1. 清空旧树。
  2. 重复训练 `n_trees` 棵树：
     - 从原始训练集进行有放回抽样，构造 bootstrap 数据集。
     - 使用当前森林随机源为每棵树再生成一个独立随机种子。
     - 创建 `DecisionTree` 并训练。
     - 将训练完成的树加入 `self.trees`。
- 参数设置：
  - `df`：训练数据，最后一列为标签。
- 返回值：`self`。
- 实现细节：
  - bootstrap 负责制造“样本扰动”。
  - `DecisionTree` 里的 `node_chose_limit` 负责制造“特征扰动”。

### 3) `predict(self, X)`

- 作用：使用整片森林对样本进行预测，并通过多数投票得到最终结果。
- 逻辑：
  1. 依次调用每棵树的 `predit(X)` 得到所有树的预测。
  2. 将结果转置为按样本组织的二维数组。
  3. 对每个样本统计各类别出现次数。
  4. 返回票数最多的类别。
- 参数设置：
  - `X`：仅包含特征列的数据。
- 返回值：`np.ndarray`，最终预测结果。
- 实现细节：
  - 这里调用的是 `DecisionTree.predit`，因为底层决策树函数名就是这个拼写。

### 4) `score(self, X, y)`

- 作用：计算随机森林的准确率。
- 逻辑：
  1. 调用 `predict(X)` 得到预测结果。
  2. 与 `y` 做逐元素比较。
  3. 返回平均正确率。
- 参数设置：
  - `X`：特征数据。
  - `y`：真实标签。
- 返回值：`float`，准确率。

### 文件级总结

- 该文件实现的是一个标准结构的简化版随机森林。
- 其随机性主要来自两部分：
  - 训练集 bootstrap 抽样
  - 每个树节点只在随机抽取的部分特征中选最佳特征
- 最终分类结果由多数投票得到。

---

## `train.py`

### 文件职责

- 负责读取数据集、做分层 `k` 折划分、训练随机森林、计算指标、保存实验结果和绘图。
- 是整个 HW3 实验流程的主驱动文件。

### 1) `random_seed(seed=42)`

- 作用：设置 NumPy 的全局随机种子。
- 逻辑：调用 `np.random.seed(seed)`。
- 参数设置：
  - `seed`：随机种子，默认 `42`。
- 返回值：无显式返回。

### 2) `read_csv(file_path)`

- 作用：读取 CSV 数据集。
- 逻辑：直接调用 `pd.read_csv(file_path)`。
- 参数设置：
  - `file_path`：CSV 文件路径。
- 返回值：`pandas.DataFrame`。

### 3) `cal_acc(y_true: np.ndarray, y_pred: np.ndarray)`

- 作用：计算分类准确率。
- 逻辑：返回 `np.mean(y_true == y_pred)`。
- 参数设置：
  - `y_true`：真实标签。
  - `y_pred`：预测标签。
- 返回值：`float`。

### 4) `cal_conf_matrix(y_true, y_pred, positive_label=1)`

- 作用：手工统计二分类混淆矩阵四个元素。
- 逻辑：
  1. 初始化 `tp, fp, tn, fn = 0`。
  2. 遍历真实标签与预测标签。
  3. 根据 `positive_label` 判断并累计到四种情况。
- 参数设置：
  - `y_true`：真实标签序列。
  - `y_pred`：预测标签序列。
  - `positive_label`：被视为正类的标签值，默认 `1`。
- 返回值：`(tp, fp, tn, fn)`。

### 5) `cal_precision(y_true, y_pred, positive_label=1)`

- 作用：计算精确率。
- 逻辑：
  1. 先调用 `cal_conf_matrix`。
  2. 根据公式 `tp / (tp + fp)` 计算。
  3. 若分母为 0，则返回 `0.0`。
- 参数设置：
  - `y_true`：真实标签。
  - `y_pred`：预测标签。
  - `positive_label`：正类标签。
- 返回值：`float`。

### 6) `cal_recall(y_true, y_pred, positive_label=1)`

- 作用：计算召回率。
- 逻辑：
  1. 调用 `cal_conf_matrix`。
  2. 根据公式 `tp / (tp + fn)` 计算。
  3. 若分母为 0，则返回 `0.0`。
- 参数设置：
  - `y_true`：真实标签。
  - `y_pred`：预测标签。
  - `positive_label`：正类标签。
- 返回值：`float`。

### 7) `cal_f1_score(y_true, y_pred, positive_label=1)`

- 作用：计算 F1 分数。
- 逻辑：
  1. 调用 `cal_precision` 和 `cal_recall`。
  2. 按 `2 * (prec * rec) / (prec + rec)` 计算。
  3. 若分母为 0，则返回 `0.0`。
- 参数设置：
  - `y_true`：真实标签。
  - `y_pred`：预测标签。
  - `positive_label`：正类标签。
- 返回值：`float`。

### 8) `k_fold(df, label_name, k=5, random_seed=42)`

- 作用：对数据集做分层 `k` 折划分。
- 逻辑：
  1. 用 `RandomState` 构造可复现随机源。
  2. 按标签值把样本索引分组。
  3. 对每个类别内部索引随机打乱。
  4. 采用轮转分配方式把同一类别样本均匀分到 `k` 个折中。
  5. 根据各折索引重新取出对应 `DataFrame`。
- 参数设置：
  - `df`：完整数据集。
  - `label_name`：标签列名。
  - `k`：折数。
  - `random_seed`：随机种子。
- 返回值：长度为 `k` 的 `DataFrame` 列表。
- 实现细节：
  - 这是手写版 stratified k-fold。
  - 目标是让每一折中的类别比例尽量接近原始数据集。

### 9) `train_test(fold, text_fold_idx)`

- 作用：从 `k` 折结果中拆出一折做测试集，其余折拼成训练集。
- 逻辑：
  1. `fold[text_fold_idx]` 作为测试集。
  2. 其余折做 `concat` 合并成训练集。
  3. 对训练集和测试集都重新整理索引。
- 参数设置：
  - `fold`：`k_fold` 返回的折列表。
  - `text_fold_idx`：被选为测试集的折编号。
- 返回值：`(train_fold, test_fold)`。
- 实现细节：
  - 这里变量名 `text_fold_idx` 从语义上应当是 `test_fold_idx`，但不影响功能。

### 10) `cal_ntrees(df, label_name, k=5, random_seed=42, positive_label=1, n_trees=10, max_lenght=None, min_samples_split=2, min_gain=1e-6, node_chose_limit=None)`

- 作用：在固定超参数下，对某个 `n_trees` 设置做 `k` 折交叉验证并统计平均指标。
- 逻辑：
  1. 调用 `k_fold` 得到分层折。
  2. 初始化 `accs`、`prec`、`recall`、`f1` 四个列表。
  3. 依次以每一折作为测试集：
     - 用 `train_test` 构造训练集和测试集。
     - 创建 `RandomForest`。
     - 在训练集上训练森林。
     - 从测试集拆出 `X_test` 和 `y_test`。
     - 预测并计算四类指标。
  4. 对各折结果求平均。
  5. 返回字典形式结果。
- 参数设置：
  - `df`：完整数据集。
  - `label_name`：标签列名。
  - `k`：折数。
  - `random_seed`：随机种子。
  - `positive_label`：正类标签。
  - `n_trees`：随机森林树数。
  - `max_lenght`：树最大深度。
  - `min_samples_split`：最小分裂样本数。
  - `min_gain`：最小信息增益阈值。
  - `node_chose_limit`：每个节点候选特征数。
- 返回值：`dict`，包括：
  - `ntrees`
  - `acc`
  - `prec`
  - `recall`
  - `f1`

### 11) `make_graph_plt(df, x_lab, y_lab, title, dir, filename=None)`

- 作用：根据结果表绘制折线图并保存图片。
- 逻辑：
  1. 新建画布。
  2. 绘制 `x_lab` 对 `y_lab` 的折线图。
  3. 设置横纵轴标签、标题、网格与紧凑布局。
  4. 保存到指定目录。
  5. 关闭图像对象。
- 参数设置：
  - `df`：结果数据表。
  - `x_lab`：横轴列名。
  - `y_lab`：纵轴列名。
  - `title`：图标题。
  - `dir`：输出目录。
  - `filename`：输出图片名。
- 返回值：无显式返回。

### 12) `cal_one_dataset(dataset_name, csv_path, label_col, dir, filename=None, ntree_values=(1, 5, 10, 20, 30, 40, 50), k=5, random_seed=42, max_lenght=None, min_samples_split=2, min_gain=1e-6, node_chose_limit=None, positive_label=1)`

- 作用：对单个数据集执行一整套实验流程。
- 逻辑：
  1. 读取 CSV 数据。
  2. 遍历不同的 `ntree_values`。
  3. 对每个树数调用 `cal_ntrees` 得到平均指标。
  4. 汇总为结果表 `DataFrame`。
  5. 创建输出目录。
  6. 保存结果 CSV。
  7. 分别绘制 Accuracy、Precision、Recall、F1 四张曲线图。
- 参数设置：
  - `dataset_name`：实验名称，用于输出文件命名。
  - `csv_path`：数据集路径。
  - `label_col`：标签列名。
  - `dir`：结果输出目录。
  - `filename`：当前实现中未使用。
  - `ntree_values`：要测试的树数量列表。
  - `k`：折数。
  - `random_seed`：随机种子。
  - `max_lenght`：树最大深度。
  - `min_samples_split`：最小分裂样本数。
  - `min_gain`：最小信息增益。
  - `node_chose_limit`：候选特征数。
  - `positive_label`：正类标签。
- 返回值：`DataFrame`，每一行对应一个 `n_trees` 的平均评估结果。
- 实现细节：
  - `filename` 参数没有被实际使用，可以视为冗余参数。

### 13) `x(k=5, max_length=10, min_samples_split=2)`

- 作用：封装一次完整实验，分别在 `WDBC` 和 `Loan` 数据集上运行。
- 逻辑：
  1. 设置全局随机种子。
  2. 定义两个数据集的绝对路径和标签列名。
  3. 调用 `cal_one_dataset` 跑 `WDBC` 数据集实验。
  4. 调用 `cal_one_dataset` 跑 `Loan` 数据集实验。
  5. 打印两组实验结果。
- 参数设置：
  - `k`：交叉验证折数。
  - `max_length`：树深度，随后传给 `max_lenght`。
  - `min_samples_split`：最小分裂样本数。
- 返回值：无显式返回。
- 实现细节：
  - 这里把 `node_chose_limit` 固定为：
    - `WDBC` 使用 `5`
    - `Loan` 使用 `3`
  - 注释中说明它们对应特征数平方根的近似值。
  - 使用的是绝对路径，因此代码可移植性较差。

### 14) `if __name__ == "__main__":`

- 作用：脚本入口，批量执行多组实验配置。
- 逻辑：
  1. 先测试 `max_depths = [5, 8, 10, 15]`，固定 `k=5` 和 `min_samples_split=2`。
  2. 再测试 `min_splits = [5, 10]`，固定 `max_length=10`。
  3. 每组参数都调用一次 `x(...)`。
- 参数设置：无。
- 返回值：无。
- 实现细节：
  - 运行该脚本会批量生成多个结果目录、CSV 文件和图片。

### 文件级总结

- `train.py` 是 HW3 的实验主控脚本。
- 它没有实现新的模型结构，而是围绕 `RandomForest` 构建了：
  - 数据读取
  - 分层交叉验证
  - 指标计算
  - 多组超参数实验
  - 结果保存与绘图
- 如果要复现实验结果，核心入口就是这个文件。

---

## 代码中值得注意的命名与实现细节

- 多处命名直接沿用了源码原始拼写：
  - `max_lenght`
  - `node_chose_limit`
  - `SpliteByFeatureCate`
  - `SpliteByFeatureNum`
  - `predit`
  - `predit_one`
  - `finial_predictions`
  - `text_fold_idx`
- `decision_tree.py` 中还保留了作者的中英混合注释和口语化注释。
- `train.py` 中模板原先说“空文件”，但实际不是空文件，而是整个实验流程的核心文件。
- 决策树的数值特征阈值选择是“均值切分”，不是标准 ID3/C4.5/CART 那种遍历所有候选阈值。
- 随机森林的随机性来自：
  - bootstrap 抽样
  - 节点特征随机子集选择
- 预测时对未见过的离散特征值有回退机制，会返回当前节点的多数类。

---

## 三个文件之间的调用关系

- `train.py`
  - 调用 `RandomForest.fit(...)`
  - 调用 `RandomForest.predict(...)`
- `random_forest.py`
  - 依赖 `DecisionTree`
  - 在 `fit(...)` 中创建多棵 `DecisionTree`
  - 在 `predict(...)` 中调用每棵树的 `predit(...)`
- `decision_tree.py`
  - 负责底层单棵树的训练与预测

整体流程可以概括为：

`train.py` 组织实验 -> `random_forest.py` 训练森林 -> `decision_tree.py` 训练单棵树并完成节点级预测。
