# Report and  Demonstration on Multimodal Tumor Auxiliary Diagnosis Tasks

---

## 1. Literature Review

**所有代码以及结果在如下链接：**[three_classification_and_dection_tasks/src/task2_metastasis.py at master · inogi-sama/three_classification_and_dection_tasks · GitHub](https://github.com/inogi-sama/three_classification_and_dection_tasks)
**The complete implementation and corresponding results can be found via the link below:**
[three_classification_and_dection_tasks/src/task2_metastasis.py at master · inogi-sama/three_classification_and_dection_tasks · GitHub](https://github.com/inogi-sama/three_classification_and_dection_tasks)
**Data Source:** All data used are derived from the following publication:
Wang, X., et al. "A multimodal dataset for precision oncology in head and neck cancer." (Nature Communications, 2023).

Github Repo Structure：
	data：
			——interim
			——processed
			——raw(from HANCOCK_MultimodalDataset)：
				————dictionaries
				————structured
				————text
				————uni_embeddings
	outputs：
		——task1
		——task2
		——task3
	src：
		——1_build_master_table.py
		——2_attach_labels.py
		——task1_margin.py
        ——task2_metastasis.py
		——task3.1_match_warning.py
        ——task3.2_match_warning.py

In recent years, AI-assisted diagnosis based on digital pathology whole slide images (WSI) has made significant progress in oncology. Studies have shown that deep learning models can effectively identify key pathological features such as tumor margin status and lymph node metastasis (Campanella et al., Nature Medicine 2019; Lu et al., Nature Communications 2021). Furthermore, comparing image analysis results with pathology report texts helps improve diagnostic accuracy and safety by identifying potential missed or misdiagnosed cases.

近年来，基于数字病理切片（WSI, Whole Slide Image）和人工智能的辅助诊断在肿瘤学领域取得了显著进展。相关研究表明，深度学习模型能够有效识别肿瘤切缘状态、淋巴结转移等关键病理特征（Campanella et al., Nature Medicine 2019；Lu et al., Nature Communications 2021）。此外，将影像分析结果与病理报告文本进行比对，有助于提升诊断的准确性和安全性，及时发现潜在的漏诊或误诊。

### 1.1 Task_1_Margin Classification / 切缘判别

Traditional methods rely on pathologists' visual inspection, which is subjective. In recent years, deep learning methods such as convolutional neural networks (CNN) have been widely used for WSI margin classification, improving automation and accuracy.
	Reference: Campanella G, et al. Clinical-grade computational pathology using weakly supervised deep learning on whole slide images. Nat Med. 2019.

传统方法依赖病理医生肉眼判读，主观性强。近年来，卷积神经网络（CNN）等深度学习方法被广泛应用于WSI切缘判别，提升了自动化和准确性。

### 1.2 Task_2_Lymph Node Metastasis Detection / 淋巴结转移检测

Traditional detection relies on manual search, which is time-consuming and labor-intensive. AI models (such as ResNet, EfficientNet) can automatically detect metastatic foci, and some studies have reached clinical usability.
	Reference: Lu MY, et al. AI-based pathology predicts origins for cancers of unknown primary. Nat Commun. 2021.

传统检测依赖人工逐个查找，费时费力。AI模型（如ResNet、EfficientNet等）可自动检测转移灶，部分研究已达临床可用水平。

### 1.3 Task_3_Image-Text Matching and Warning / 影像-文本比对与预警

Multimodal fusion (image + text) is a current research hotspot. Automatic comparison can help discover missed or misdiagnosed cases in reports or models, improving diagnostic safety.

多模态融合（影像+文本）是当前研究热点。自动比对可辅助发现报告遗漏或模型误判，提高诊断安全性。

---

## 2. Data Processing & Preprocessing / 数据处理与预处理

### 2.1 Data Source & Structure / 数据来源与结构

- The raw data includes WSI feature files (h5/npy/pt), pathology report texts (txt), and structured labels (json/csv).
- The script `1_build_master_table.py` automatically indexes and merges all sample text, feature, and label paths.

原始数据包括WSI特征文件（h5/npy/pt等）、病理报告文本（txt）、结构化标签（json/csv）。通过 `1_build_master_table.py`自动索引并合并所有样本的文本、特征、标签路径。

```python
# Traverse raw data files to generate master table
text_df = build_text_index(text_root)
emb_df = build_embedding_index(emb_root)
master = pd.merge(text_df, emb_df, on="patient_id", how="outer")
master.to_csv(PROCESSED_DIR / "master_table_stage1.csv", index=False)
```

### 2.2 Label Mapping / 标签映射

- The script `2_attach_labels.py` automatically extracts margin and metastasis labels from structured json/csv files.
- Supports multiple label formats and names, and normalizes them automatically.

通过 `2_attach_labels.py`自动从结构化json/csv文件中提取切缘、转移等标签。支持多种标签格式和命名，自动归一化。

```python
def map_margin(v):
    if pd.isna(v): return pd.NA
    s = str(v).strip().lower()
    if any(k in s for k in ["positive", "pos", "r1", "r2", "involved", "yes"]): return 1
    if any(k in s for k in ["negative", "neg", "r0", "clear", "no"]): return 0
    try:
        clean_s = s.replace('<', '').replace('>', '').replace('cm', '').strip()
        val = float(clean_s)
        return 1 if val < 0.2 else 0
    except ValueError:
        pass
    return pd.NA
```

---

### 2.3 Feature Aggregation / 特征聚合

For 1024-dimensional UNI features extracted from WSI, this task adopts task-specific pooling strategies based on anatomical and pathological characteristics, rather than a one-size-fits-all approach.

- **Max Pooling (Task 1):** For margin detection, since residual tumor cells are extremely sparse, mean pooling would dilute the pathological signal. We use $max(axis=0)$ to extract the most responsive feature dimension across all patches, effectively preserving signals from tiny residual lesions.
- **Mean Pooling (Task 2):** For lymph node metastasis, where metastasis often involves large-scale tissue remodeling, mean pooling better captures the global pathological context and filters local noise, resulting in more robust feature representation.

针对WSI提取的1024维UNI特征，本项目未采用通用的单一池化方式，而是根据病理学任务的解剖学特质进行了差异化设计：

- **最大池化（Task 1）：** 由于残留癌细胞在切片中占比极低（高稀疏性），传统的平均池化会显著稀释病理性信号。本项目采用 $max(axis=0)$ 操作，旨在提取所有图像块中响应最剧烈的特征维度，有效保留了微小残留病灶的信号特征。
- **平均池化（Task 2）：** 考虑到转移通常伴随大面积的组织结构重构，采用平均池化能更好地捕捉全片的病理背景，过滤局部噪声，从而实现更稳健的特征表示。

### 2.4 Threshold Optimization / 阈值调整

The core challenge in medical diagnosis models is the trade-off between false negatives (missed diagnosis) and false positives (over-diagnosis). This project introduces a recall-constrained threshold search algorithm:

- **Constrained Optimization:** Instead of a static threshold of 0.5, a $grid\_search$ strategy dynamically scans the $[0.05, 0.95]$ interval.
- **Sensitivity-First:** For high-risk margin detection, a constraint of $min\_recall=0.60$ is set. By locking the best threshold at 0.43, a recall of 91.3% was achieved on the independent test set.

  医疗诊断模型的核心矛盾在于漏诊与误诊的权衡。本项目引入了基于召回率约束的阈值搜索算法：

- **约束优化：** 代码实现中放弃了0.5的静态阈值，采用 $grid\_search$ 策略在 $[0.05, 0.95]$ 区间内动态扫描。
- **灵敏度优先：** 针对切缘检测的高风险性，设定了 $min\_recall=0.60$ 的约束条件。通过将最佳阈值锁定在0.43，在独立测试集上实现了91.3%的灵敏度（Recall）。

### 2.5 RF and LR

To verify the classification effectiveness of the feature space, systematic model comparison experiments were conducted:

- **LR:** StandardScaler + Logistic Regression as a linear baseline.
- **RF:** Random Forest ensemble to capture complex interactions among 1024 features.
- **Conclusion:** Results show Task 1 has significant non-linear characteristics (RF outperforms LR), while Task 2 is more linearly separable (LR achieves AUC 0.88). This comparative analysis supports the scientific model selection rather than blind parameter tuning.

为验证特征空间的分类效能，本项目实施了系统性的模型对比实验：

- **LR：** 构建了“标准化+逻辑回归”（StandardScaler + LogReg）作为线性基准。
- **RF：** 采用集成学习（Random Forest）捕捉1024维特征间的复杂交互（Interactions）。
- **总结：** 实验结果显示Task 1具有显著的非线性特征（RF显著优于LR），而Task 2展现了较强的线性可分性（LR亦能达到0.88 AUC）。这种对比分析证明了模型选择的科学性，而非盲目调参。

---

## 3 Task1: Margin Classification/ 切缘判别

### 3.1 Feature Extraction / 特征提取

- Read WSI features and use Max Pooling to capture local residual signals.

读取WSI特征，采用Max Pooling捕捉局部残留信号。

```python
def load_h5_embedding(h5_path: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        feats = f["features"][:]  # (Npatch, 1024)
    return feats.max(axis=0).astype(np.float32)
```

### 3.2 Model Construction

- Use Logistic Regression and Random Forest models.
- Stratified split of train/test sets, cross-validation evaluation.

采用Logistic Regression和Random Forest两种模型。训练集/测试集分层划分，交叉验证评估。

```python
models = {
    "logreg": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))
    ]),
    "rf": RandomForestClassifier(
        n_estimators=500, max_depth=12, min_samples_leaf=5,
        max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=42
    )
}
```

### 3.3 Threshold Optimization & Evaluation / 阈值优化与评估

- Automatically scan different thresholds, prioritizing recall.
- Output AUC, accuracy, recall, F1, etc.

自动扫描不同阈值，优先保证Recall。输出AUC、准确率、召回率、F1等指标。

```python
def find_best_threshold(y_true, y_prob, min_recall=0.60):
    grid = np.arange(0.05, 0.96, 0.01)
    rows = [eval_at_threshold(y_true, y_prob, th) for th in grid]
    dfm = pd.DataFrame(rows)
    cand = dfm[dfm["recall_pos"] >= min_recall].copy()
    if len(cand) > 0:
        best = cand.sort_values(["f1_pos", "precision_pos"], ascending=False).iloc[0]
        mode = f"constrained_recall>={min_recall}"
    else:
        best = dfm.sort_values(["f1_pos", "recall_pos"], ascending=False).iloc[0]
        mode = "max_f1_fallback"
    return best.to_dict(), dfm, mode
```

### 3.4 Results

- Total samples: 581 / 总样本数: 581
- Label distribution: 0 (negative): 295, 1 (positive): 286 / 标签分布: 0（阴性）: 295, 1（阳性）: 286
- Best model: Random Forest / 最佳模型: 随机森林
- Test AUC: 0.66 / 测试AUC: 0.66
- Best threshold: 0.43 / 最优阈值: 0.43
  At best threshold: / 最优阈值下：
  - Accuracy: 0.60 / 准确率: 0.60
  - Precision (positive): 0.56 / 阳性精确率: 0.56
  - Recall (positive): 0.91 / 阳性召回率: 0.91
  - F1 (positive): 0.69 / 阳性F1: 0.69

**ROC Curve**

![Task 1 ROC Curve]([roc_curve_plot.png](https://hackmd-prod-images.s3.ap-northeast-1.amazonaws.com/uploads/upload_e33eaf509b9cc4c0225afc7518017f41.png?AWSAccessKeyId=AKIA3XSAAW6AWSKNINWO&Expires=1774187822&Signature=TU5tr2nAhpQDezITQ1mpZXD7YWE%3D))

---

## 4. Task 2: Lymph Node Metastasis Detection Model / 淋巴结转移检测

### 4.1 Feature Extraction / 特征提取

- Use Mean Pooling to aggregate WSI features into slide-level vectors.

采用Mean Pooling对WSI特征聚合，获得slide-level向量。

```python
def load_h5_embedding(h5_path: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        feats = f["features"][:]   # (Npatch, 1024)
    return feats.mean(axis=0).astype(np.float32)
```

### 4.2 Model Construction

- Use Logistic Regression and Random Forest.
- Stratified split of train/test sets, cross-validation evaluation.

同样采用Logistic Regression和Random Forest。训练集/测试集分层划分，交叉验证评估。

```python
cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
model.fit(X_train, y_train)
prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, prob)
```

### 4.3 Threshold Optimization & Evaluation / 阈值优化与评估

- Automatically scan different thresholds, balancing recall and precision.
- Output AUC, F1, recall, precision, etc.

自动扫描不同阈值，兼顾召回率和准确率。输出AUC、F1、召回率、精确率等。

### 4.4 Results

- Total samples: 743 (train: 594, test: 149)
- Label distribution: 0: 327, 1: 416
- Best model: Random Forest
- Test AUC: 0.92
- Best threshold: 0.55
  At best threshold:
  - Accuracy: 0.93 / 准确率: 0.93
  - Precision (positive): 0.99 / 阳性精确率: 0.99
  - Recall (positive): 0.88 / 阳性召回率: 0.88
  - F1 (positive): 0.93

---

## 5. Task 3: Image-Text Matching Warning / 影像-文本比对预警

### 5.1 Automatic Text Label Extraction / 文本标签自动提取

Use regular expressions and keyword rules to extract metastasis labels from pathology reports.

利用正则表达式和关键词规则从病理报告文本中提取转移标签。

```python
def rule_label(t):
    if not isinstance(t, str) or not t.strip():
        return None, "empty"
    if PAT_PN_POS.search(t):
        return 1, "rule:pN_pos"
    if PAT_PN0.search(t):
        return 0, "rule:pN0"
    # ... and so on
```

### 5.2 Supplementary Text Classification Model / 文本分类模型补充

- For samples not covered by rules, train a TF-IDF + Logistic Regression text classifier.

对规则无法覆盖的样本，训练TF-IDF+Logistic Regression文本分类器。

```python
feats = FeatureUnion([
    ("word", TfidfVectorizer(analyzer="word", ngram_range=(1,2), max_features=30000)),
    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=50000)),
])
clf = Pipeline([
    ("feats", feats),
    ("lr", LogisticRegression(max_iter=3000, class_weight="balanced"))
])
clf.fit(X_train, y_train)
```

### 5.3 Image-Text Comparison & Warning / 影像-文本比对与预警

- Compare text labels with image model predictions; if the report says "no metastasis" but the model detects suspicious tumor, automatically flag a warning.

将文本标签与影像模型预测结果比对，若报告为“无转移”但模型检测为“可疑肿瘤”，则自动发出预警。

```python
if (text_label == 0) and (image_pred == 1):
    warning = True
else:
    warning = False
```

### 5.4 Results / 结果展示

- Test samples: 149
- Text label known: 137, unknown: 12
- Total warnings: 15 (conflict: 11, unknown high-risk: 4)
- Text label extraction coverage: 89% / 文本标签提取覆盖率: 89%
- Label counts: positive: 540, negative: 141, unknown: 82 / 标签计数: 阳性: 540，阴性: 141，未知: 82

**Warning Case Table / 预警案例表**

| patient_id | y_true | y_prob | y_pred@0.5 | text_metastasis_label | warning_flag | warning_reason                                               |
| ---------- | ------ | ------ | ---------- | --------------------- | ------------ | ------------------------------------------------------------ |
| 655        | 1      | 0.958  | 1          | 0.0                   | 1            | Conflict: text says no-metastasis, image predicts metastasis |
| 505        | 1      | 0.958  | 1          | 0.0                   | 1            | Conflict: text says no-metastasis, image predicts metastasis |
| 761        | 1      | 0.92   | 1          | 0.0                   | 1            | Conflict: text says no-metastasis, image predicts metastasis |
| 197        | 1      | 0.93   | 1          | 0.0                   | 1            | Conflict: text says no-metastasis, image predicts metastasis |
| ...        | ...    | ...    | ...        | ...                   | ...          | ...                                                          |
