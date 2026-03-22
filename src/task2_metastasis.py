import os
import json
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, roc_curve
)

DATA_CSV = "data/processed/master_table_stage2_labeled.csv"

OUT_DIR = Path("outputs/task2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_h5_embedding(h5_path: str) -> np.ndarray:
    """
    从单个 h5 文件读取 patch 特征，并做 mean pooling 得到 slide-level 向量
    输入:
        h5_path: h5 文件路径，内部需包含 "features" 数据集，形状通常为 (Npatch, 1024)
    输出:
        形状为 (1024,) 的 float32 向量
    """
    with h5py.File(h5_path, "r") as f:
        feats = f["features"][:]   # (Npatch, 1024)
    return feats.mean(axis=0).astype(np.float32)


def eval_at_threshold(y_true, y_prob, th):
    """
    在给定概率阈值 th 下计算分类指标
    """
    y_pred = (y_prob >= th).astype(int)
    return {
        "threshold": float(th),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_pos": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_pos": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_pos": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }


def find_best_threshold(y_true, y_prob, min_recall=0.60):
    """
    阈值策略：
    - 先在 [0.05, 0.95] 范围内搜索
    - 优先满足 recall >= min_recall 的阈值里，选 F1 最高（若并列再看 precision）
    - 如果一个都不满足，则退化为全局 F1 最优
    返回:
        best_metrics: 最优阈值对应指标(dict)
        th_table: 全阈值扫描表(DataFrame)
        mode: 选择模式说明字符串
    """
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

#ROC image
def plot_results(roc_df, best_auc, output_dir):
    plt.figure(figsize=(8, 6))
    plt.plot(roc_df['fpr'], roc_df['tpr'], color='darkorange', lw=2,
             label=f'ROC curve (area = {best_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Task 2)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(output_dir / "roc_curve_plot.png")
    print(f"ROC 曲线图已保存至: {output_dir}")


def main():
    df = pd.read_csv(DATA_CSV)

    df["metastasis_label"] = pd.to_numeric(df["metastasis_label"], errors="coerce")

    #仅保留标签非空和embedding_path非空
    df = df[df["metastasis_label"].notna() & df["embedding_path"].notna()].copy()

    #路径统一转字符串，过滤掉磁盘上不存在的特征文件
    df["embedding_path"] = df["embedding_path"].astype(str)
    df = df[df["embedding_path"].apply(os.path.exists)].copy()

    X_list, y_list, pid_list = [], [], []
    for _, r in df.iterrows():
        try:
            emb = load_h5_embedding(r["embedding_path"])
            X_list.append(emb)
            y_list.append(int(r["metastasis_label"]))
            pid_list.append(r["patient_id"])
        except Exception:
            #jump
            continue

    X = np.stack(X_list, axis=0)                 # (N, 1024)
    y = np.array(y_list, dtype=np.int64)         # (N,)
    pids = np.array(pid_list)                    # (N,)


    X_train, X_test, y_train, y_test, pid_train, pid_test = train_test_split(
        X, y, pids, test_size=0.2, random_state=42, stratify=y
    )

    # LR: 标准化 + 逻辑回归（类别不平衡加权）
    # RF: 随机森林（类别不平衡加权）
    models = {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))
        ]),
        "rf": RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1
        )
    }

    #训练集内部做5折CV，用于评估稳定性
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    #test AUC 选最佳模型
    all_results = {}
    best_model_name, best_auc = None, -1
    best_probs, best_pred_05 = None, None

    # ---------- 5.5 训练与评估 ----------
    for name, model in models.items():
        # (a) 训练集5折交叉验证AUC
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

        # (b) 在完整训练集拟合，然后在独立测试集评估
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred05 = (prob >= 0.5).astype(int)
        auc = roc_auc_score(y_test, prob)

        # (c) 汇总指标
        all_results[name] = {
            "cv_auc_mean": float(cv_auc.mean()),
            "cv_auc_std": float(cv_auc.std()),
            "test_auc": float(auc),
            "test_acc@0.5": float(accuracy_score(y_test, pred05)),
            "test_f1_pos@0.5": float(f1_score(y_test, pred05, zero_division=0)),
            "test_recall_pos@0.5": float(recall_score(y_test, pred05, zero_division=0)),
            "test_precision_pos@0.5": float(precision_score(y_test, pred05, zero_division=0)),
            "test_cm@0.5": confusion_matrix(y_test, pred05).tolist()
        }

        # (d) 按 test AUC 选择最佳模型
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_probs = prob
            best_pred_05 = pred05

    # ---------- 5.6 对最佳模型做阈值优化 ----------
    best_th_metrics, th_table, th_mode = find_best_threshold(
        y_test, best_probs, min_recall=0.60
    )
    best_th = float(best_th_metrics["threshold"])
    pred_best = (best_probs >= best_th).astype(int)

    # ---------- 5.7 导出ROC曲线点 ----------
    # 方便后续画图（Excel/Matplotlib都能用）
    fpr, tpr, ths = roc_curve(y_test, best_probs)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": ths})

    # ---------- 5.8 保存中间与结果文件 ----------
    # 阈值扫描全表
    th_table.to_csv(OUT_DIR / "threshold_scan.csv", index=False)
    # ROC曲线点
    roc_df.to_csv(OUT_DIR / "roc_curve_best_model.csv", index=False)

    # 测试集逐样本预测
    pred_df = pd.DataFrame({
        "patient_id": pid_test,
        "y_true": y_test,
        "y_prob": best_probs,
        "y_pred@0.5": best_pred_05,
        f"y_pred@{best_th:.2f}": pred_best
    })
    pred_df.to_csv(OUT_DIR / "test_predictions_best_model.csv", index=False)

    # 总结JSON
    summary = {
        "n_total": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "label_distribution_total": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
        "model_compare": all_results,
        "selected_model_by_test_auc": best_model_name,
        "selected_model_test_auc": float(best_auc),
        "threshold_selection_mode": th_mode,
        "best_threshold_metrics_on_test": best_th_metrics,
        "cm_at_best_threshold": confusion_matrix(y_test, pred_best).tolist()
    }

    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---------- 5.9 控制台输出 ----------
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("saved to:", OUT_DIR.resolve())
    plot_results(roc_df, best_auc, OUT_DIR)


if __name__ == "__main__":
    main()