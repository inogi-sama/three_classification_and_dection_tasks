import re
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

IN_CSV = "data/processed/master_table_stage2_labeled.csv"
OUT_CSV = "outputs/task3/text_labels.csv"
OUT_SUMMARY = "outputs/task3/text_hybrid_summary.json"
TARGET_COVERAGE = 0.80

Path("outputs/task3").mkdir(parents=True, exist_ok=True)

PAT_PN0 = re.compile(r"\bp?n\s*0\b", re.I)
PAT_PN_POS = re.compile(r"\bp?n\s*[1-3]\b", re.I)
PAT_RATIO = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")

NEG_KW = [re.compile(r"\bkein\w*\b.*\bmetastas\w*\b", re.I),
          re.compile(r"\bohne\b.*\bmetastas\w*\b", re.I),
          re.compile(r"\bno\b.*\bmetastas\w*\b", re.I)]
POS_KW = [re.compile(r"\bmetastas\w*\b", re.I),
          re.compile(r"\blk[- ]?metastas\w*\b", re.I)]

def choose_text_col(df):
    # 强制原文优先
    for c in ["report_text", "report_text_de", "report_text_en", "text"]:
        if c in df.columns:
            return c
    raise ValueError("No text column found.")

def rule_label(t):
    if not isinstance(t, str) or not t.strip():
        return None, "empty"
    if PAT_PN_POS.search(t):
        return 1, "rule:pN_pos"
    if PAT_PN0.search(t):
        return 0, "rule:pN0"

    ms = PAT_RATIO.findall(t)
    vals = [(int(a), int(b)) for a, b in ms if int(b) > 0]
    if any(a > 0 for a, _ in vals):
        return 1, "rule:ratio_pos"
    if any(a == 0 for a, _ in vals):
        return 0, "rule:ratio_neg"

    for p in NEG_KW:
        if p.search(t):
            return 0, "rule:neg_kw"
    for p in POS_KW:
        if p.search(t):
            return 1, "rule:pos_kw"
    return None, "unknown"

def apply_threshold(probs, base_labels, pos_th, neg_th):
    out = []
    for i, b in enumerate(base_labels):
        if b is not None and not (isinstance(b, float) and np.isnan(b)):
            out.append(int(b))
        else:
            p = probs[i]
            if p >= pos_th:
                out.append(1)
            elif p <= neg_th:
                out.append(0)
            else:
                out.append(np.nan)
    return np.array(out, dtype=float)

def main():
    df = pd.read_csv(IN_CSV)
    text_col = choose_text_col(df)
    texts = df[text_col].fillna("").astype(str).tolist()

    rule_y, rule_reason = [], []
    for t in texts:
        y, r = rule_label(t)
        rule_y.append(y)
        rule_reason.append(r)

    tmp = pd.DataFrame({
        "patient_id": df["patient_id"],
        "text": texts,
        "rule_label": rule_y,
        "rule_reason": rule_reason
    })

    labeled = tmp[tmp["rule_label"].notna()].copy()
    if len(labeled) < 30 or labeled["rule_label"].nunique() < 2:
        final = tmp["rule_label"].astype(float).values
        reasons = tmp["rule_reason"].tolist()
    else:
        X_train = labeled["text"].tolist()
        y_train = labeled["rule_label"].astype(int).values

        feats = FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=1, max_features=30000)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1, max_features=50000)),
        ])

        clf = Pipeline([
            ("feats", feats),
            ("lr", LogisticRegression(max_iter=3000, class_weight="balanced"))
        ])
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(tmp["text"].tolist())[:, 1]

        #自动找阈值，优先达到覆盖率
        best = None
        for m in np.arange(0.49, -0.001, -0.01):  # margin from center 0.5
            pos_th = 0.5 + m
            neg_th = 0.5 - m
            pred = apply_threshold(probs, tmp["rule_label"].values, pos_th, neg_th)
            cov = np.mean(~np.isnan(pred))
            if cov >= TARGET_COVERAGE:
                best = (pos_th, neg_th, pred, cov)
                break

        if best is None:
            best_cov = -1
            best_pack = None
            for m in np.arange(0.49, 0.01, -0.02):
                pos_th = 0.5 + m
                neg_th = 0.5 - m
                pred = apply_threshold(probs, tmp["rule_label"].values, pos_th, neg_th)
                cov = np.mean(~np.isnan(pred))
                if cov > best_cov:
                    best_cov = cov
                    best_pack = (pos_th, neg_th, pred, cov)
            best = best_pack

        pos_th, neg_th, final, cov = best

        reasons = []
        for i, r in enumerate(tmp["rule_reason"].tolist()):
            if r != "unknown":
                reasons.append(r)
            else:
                p = probs[i]
                if p >= pos_th:
                    reasons.append(f"model_pos_p={p:.3f}")
                elif p <= neg_th:
                    reasons.append(f"model_neg_p={p:.3f}")
                else:
                    reasons.append(f"model_uncertain_p={p:.3f}")

    out = pd.DataFrame({
        "patient_id": tmp["patient_id"],
        "text_metastasis_label": final,
        "text_rule_reason": reasons
    })
    out.to_csv(OUT_CSV, index=False)

    summary = {
        "text_col": text_col,
        "n_total": int(len(out)),
        "n_rule_labeled": int(len(labeled)),
        "coverage": float(np.mean(out["text_metastasis_label"].notna())),
        "label_counts": out["text_metastasis_label"].value_counts(dropna=False).to_dict()
    }
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("saved:", OUT_CSV)
    print("saved:", OUT_SUMMARY)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()