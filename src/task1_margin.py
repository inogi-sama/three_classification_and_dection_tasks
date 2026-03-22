import os
import json
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
#import matplotlib.pyplot as plt

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
OUT_DIR = Path("outputs/task1")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_h5_embedding(h5_path: str) -> np.ndarray:
    """
    针对切缘任务(Task1)，使用 Max Pooling 捕捉局部残留信号
    """
    with h5py.File(h5_path, "r") as f:
        feats = f["features"][:]  # (Npatch, 1024)
    # 使用 max 而不是 mean，因为切缘阳性往往是局部微小病灶
    return feats.max(axis=0).astype(np.float32)


def eval_at_threshold(y_true, y_prob, th):
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
    针对切缘任务，优先保证 Recall
    """
    grid = np.arange(0.05, 0.96, 0.01)
    rows = [eval_at_threshold(y_true, y_prob, th) for th in grid]
    dfm = pd.DataFrame(rows)

    #在满足最低 Recall 的前提下，选F1最高的一个
    cand = dfm[dfm["recall_pos"] >= min_recall].copy()
    if len(cand) > 0:
        best = cand.sort_values(["f1_pos", "precision_pos"], ascending=False).iloc[0]
        mode = f"constrained_recall>={min_recall}"
    else:
        best = dfm.sort_values(["f1_pos", "recall_pos"], ascending=False).iloc[0]
        mode = "max_f1_fallback"

    return best.to_dict(), dfm, mode


def main():
    df = pd.read_csv(DATA_CSV)
    df["margin_label"] = pd.to_numeric(df["margin_label"], errors="coerce")
    df = df[df["margin_label"].notna() & df["embedding_path"].notna()].copy()
    df["embedding_path"] = df["embedding_path"].astype(str)
    df = df[df["embedding_path"].apply(os.path.exists)].copy()

    X_list, y_list, pid_list = [], [], []
    for _, r in df.iterrows():
        try:
            emb = load_h5_embedding(r["embedding_path"])
            X_list.append(emb)
            y_list.append(int(r["margin_label"]))
            pid_list.append(r["patient_id"])
        except Exception:
            continue

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    pids = np.array(pid_list)

    X_train, X_test, y_train, y_test, pid_train, pid_test = train_test_split(
        X, y, pids, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),  #LR对特征缩放敏感
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))
        ]),
        "rf": RandomForestClassifier(
            n_estimators=500,
            max_depth=12,  # 限制深度防止过拟合
            min_samples_leaf=5,  # 每个叶子最少5个样本，过滤噪声
            max_features="sqrt",  # 每次分裂只看部分特征
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_results = {}
    best_model_name, best_auc = None, -1
    best_probs = None

    for name, model in models.items():
        #5折交叉验证
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

        #测试集评估
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, prob)

        all_results[name] = {
            "cv_auc_mean": float(cv_auc.mean()),
            "test_auc": float(auc),
            "test_acc@0.5": float(accuracy_score(y_test, (prob >= 0.5))),
            "test_recall_pos@0.5": float(recall_score(y_test, (prob >= 0.5), zero_division=0))
        }

        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_probs = prob

    #更在乎Recall，所以阈值通常偏低
    best_th_metrics, th_table, th_mode = find_best_threshold(y_test, best_probs, min_recall=0.60)
    best_th = float(best_th_metrics["threshold"])

    #导出ROC点
    fpr, tpr, ths = roc_curve(y_test, best_probs)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": ths})
    roc_df.to_csv(OUT_DIR / "roc_curve_best_model.csv", index=False)
    th_table.to_csv(OUT_DIR / "threshold_scan.csv", index=False)

    pred_df = pd.DataFrame({
        "patient_id": pid_test,
        "y_true": y_test,
        "y_prob": best_probs,
        f"y_pred@{best_th:.2f}": (best_probs >= best_th).astype(int)
    })
    pred_df.to_csv(OUT_DIR / "test_predictions_best_model.csv", index=False)

    summary = {
        "n_total": int(len(y)),
        "label_distribution": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
        "model_compare": all_results,
        "best_model": best_model_name,
        "best_th_selected": best_th,
        "metrics_at_best_th": best_th_metrics
    }
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f">>> Task 1 Analysis Complete. Best Model: {best_model_name}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()