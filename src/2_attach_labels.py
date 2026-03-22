import os
import json
import re
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MASTER_IN = ROOT / "data" / "processed" / "master_table_stage1.csv"
MASTER_OUT = ROOT / "data" / "processed" / "master_table_stage2_labeled.csv"
LOG_OUT = ROOT / "data" / "interim" / "attach_labels_from_structured_log.txt"

STRUCT_DIR = ROOT / "data" / "raw" / "structured" / "StructuredData"
CLINICAL_JSON = STRUCT_DIR / "clinical_data.json"
PATHO_JSON = STRUCT_DIR / "pathological_data.json"


def norm_pid(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x)
    nums = re.findall(r"\d+", s)
    if not nums: return None
    return f"{int(nums[-1]):03d}"


def load_json_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Warning: File not found {path}")
        return pd.DataFrame()
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list): return pd.DataFrame(data)
    if isinstance(data, dict):
        for _, v in data.items():
            if isinstance(v, list): return pd.DataFrame(v)
        return pd.json_normalize(data)
    return pd.DataFrame()


def find_pid_col(cols):
    """补全 main 函数中调用的 ID 列查找逻辑"""
    keys = ["patient_id", "patient", "case_id", "id", "case"]
    low = {c.lower(): c for c in cols}
    for k in keys:
        for lc, c in low.items():
            if k == lc or k in lc: return c
    return None


def find_col(cols, keywords):
    """改进的字段查找：优先全字匹配，再模糊包含"""
    low_cols = [str(c).lower() for c in cols]
    for kw in keywords:
        if kw in low_cols:
            return cols[low_cols.index(kw)]
    for kw in keywords:
        for i, lc in enumerate(low_cols):
            if kw in lc: return cols[i]
    return None


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


def map_metastasis(v):
    if pd.isna(v): return pd.NA
    s = str(v).strip().lower()
    if "pn0" in s: return 0
    if re.search(r'pn[1-3]', s): return 1
    if any(k in s for k in ["positive", "metastasis", "present", "1", "yes"]): return 1
    if any(k in s for k in ["negative", "no metastasis", "absent", "0", "no"]): return 0
    return pd.NA


def main():
    logs = []
    if not MASTER_IN.exists():
        raise FileNotFoundError(f"找不到 Stage 1 文件: {MASTER_IN}")

    master = pd.read_csv(MASTER_IN, dtype={"patient_id": str})
    master["patient_id"] = master["patient_id"].str.zfill(3)

    for col in ["margin_label", "metastasis_label", "site"]:
        if col not in master.columns: master[col] = pd.NA

    clin = load_json_table(CLINICAL_JSON)
    patho = load_json_table(PATHO_JSON)

    #识别 ID 列
    clin_pid = find_pid_col(clin.columns) if not clin.empty else None
    patho_pid = find_pid_col(patho.columns) if not patho.empty else None

    #识别特征列
    patho_margin_col = find_col(patho.columns, ["resection_margin", "margin", "r_status", "residual"])
    patho_meta_col = find_col(patho.columns, ["pn_stage", "pn-stage", "metastasis", "n_stage"])
    patho_site_col = find_col(patho.columns, ["icd_code", "icd10", "site", "localization"])

    clin_margin_col = find_col(clin.columns, ["resection_margin", "margin", "r_status"])
    clin_meta_col = find_col(clin.columns, ["cn_stage", "metastasis", "n_stage"])

    logs.append(f"Detected Patho Cols: Margin={patho_margin_col}, Meta={patho_meta_col}, Site={patho_site_col}")

    merged = master.copy()

    #合并病理数据
    if patho_pid is not None and not patho.empty:
        p = patho.copy()
        p["patient_id"] = p[patho_pid].apply(norm_pid)
        p = p.dropna(subset=["patient_id"]).drop_duplicates("patient_id")

        keep_p = ["patient_id"]
        for c in [patho_margin_col, patho_meta_col, patho_site_col]:
            if c: keep_p.append(c)

        merged = merged.merge(p[keep_p], on="patient_id", how="left")

        if patho_margin_col:
            merged["margin_label"] = merged[patho_margin_col].apply(map_margin)
        if patho_meta_col:
            merged["metastasis_label"] = merged[patho_meta_col].apply(map_metastasis)
        if patho_site_col:
            merged["site"] = merged[patho_site_col].astype(str)

    #合并临床数据补充
    if clin_pid is not None and not clin.empty:
        c = clin.copy()
        c["patient_id"] = c[clin_pid].apply(norm_pid)
        c = c.dropna(subset=["patient_id"]).drop_duplicates("patient_id")

        keep_c = ["patient_id"]
        if clin_margin_col: keep_c.append(clin_margin_col)
        if clin_meta_col: keep_c.append(clin_meta_col)

        merged = merged.merge(c[keep_c], on="patient_id", how="left", suffixes=("", "_clin"))

        if clin_margin_col:
            c_col = clin_margin_col if clin_margin_col not in master.columns else f"{clin_margin_col}_clin"
            if c_col in merged.columns:
                merged["margin_label"] = merged["margin_label"].fillna(merged[c_col].apply(map_margin))

        if clin_meta_col:
            c_col = clin_meta_col if clin_meta_col not in master.columns else f"{clin_meta_col}_clin"
            if c_col in merged.columns:
                merged["metastasis_label"] = merged["metastasis_label"].fillna(merged[c_col].apply(map_metastasis))

    #最终清理与保存
    final_cols = ["patient_id", "report_path", "embedding_path", "report_text_en",
                  "margin_label", "metastasis_label", "site"]
    existing_cols = [c for c in final_cols if c in merged.columns]
    merged = merged[existing_cols]

    MASTER_OUT.parent.mkdir(parents=True, exist_ok=True)
    LOG_OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MASTER_OUT, index=False, encoding="utf-8")
    LOG_OUT.write_text("\n".join(logs), encoding="utf-8")

    print(f"Margin_labels: {merged['margin_label'].notna().sum()}")
    print(f"Meta_labels: {merged['metastasis_label'].notna().sum()}")
    print(f"log_saved_at: {LOG_OUT}")


if __name__ == "__main__":
    main()