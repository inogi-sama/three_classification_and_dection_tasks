import json
import pandas as pd
from pathlib import Path

IMG_PRED_CSV = "outputs/task2/test_predictions_best_model.csv"
TEXT_CSV = "outputs/task3/text_labels.csv"
OUT_CSV = "outputs/task3/match_results.csv"
OUT_SUMMARY = "outputs/task3/summary.json"

Path("outputs/task3").mkdir(parents=True, exist_ok=True)

# 可调阈值
HIGH_RISK_THRESHOLD = 0.70

def main():
    img = pd.read_csv(IMG_PRED_CSV)   # patient_id, y_true, y_pred, y_prob
    txt = pd.read_csv(TEXT_CSV)       # patient_id, text_metastasis_label, text_rule_reason

    df = img.merge(txt, on="patient_id", how="left")

    # 标准化
    df["text_metastasis_label"] = pd.to_numeric(df["text_metastasis_label"], errors="coerce")
    df["y_pred@0.55"] = pd.to_numeric(df["y_pred@0.55"], errors="coerce")
    df["y_prob"] = pd.to_numeric(df["y_prob"], errors="coerce")

    # 规则1：明确冲突（文本阴性 vs 图像阳性）
    df["warning_conflict"] = (
        (df["text_metastasis_label"] == 0) &
        (df["y_pred@0.55"] == 1)
    ).astype(int)

    # 规则2：文本未知但图像高度可疑
    df["warning_unknown_highrisk"] = (
        (df["text_metastasis_label"].isna()) &
        (df["y_prob"] >= HIGH_RISK_THRESHOLD)
    ).astype(int)

    df["warning_flag"] = ((df["warning_conflict"] == 1) | (df["warning_unknown_highrisk"] == 1)).astype(int)

    # 原因字段
    reasons = []
    for _, r in df.iterrows():
        if r["warning_conflict"] == 1:
            reasons.append("Conflict: text says no-metastasis, image predicts metastasis")
        elif r["warning_unknown_highrisk"] == 1:
            reasons.append(f"Text unknown, image high-risk prob>={HIGH_RISK_THRESHOLD}")
        else:
            reasons.append("")
    df["warning_reason"] = reasons

    df.to_csv(OUT_CSV, index=False)

    summary = {
        "n_samples_test": int(len(df)),
        "n_text_known": int(df["text_metastasis_label"].notna().sum()),
        "n_text_unknown": int(df["text_metastasis_label"].isna().sum()),
        "n_warning_total": int(df["warning_flag"].sum()),
        "n_warning_conflict": int(df["warning_conflict"].sum()),
        "n_warning_unknown_highrisk": int(df["warning_unknown_highrisk"].sum()),
        "high_risk_threshold": HIGH_RISK_THRESHOLD
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("saved:", OUT_CSV)
    print("saved:", OUT_SUMMARY)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()