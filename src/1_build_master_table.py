from pathlib import Path
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)


def find_files(root: Path, patterns):
    files = []
    if not root.exists():
        return files
    for p in root.rglob("*"):
        if p.is_file():
            for pat in patterns:
                if p.name.lower().endswith(pat):
                    files.append(p)
                    break
    return files


def normalize_patient_id(s: str):#用正则表达式\d+从文件名中抓取数字并统一格式为三位数字字符串
    if s is None:
        return None
    base = Path(s).stem
    nums = re.findall(r"\d+", base)
    if not nums:
        return None
    pid = int(nums[-1])
    if pid <= 0:
        return None
    return f"{pid:03d}"


def build_text_index(text_root: Path):#遍历text
    txt_files = find_files(text_root, [".txt"])
    rows = []
    for f in txt_files:
        rows.append({
            "report_path": str(f),
            "patient_id": normalize_patient_id(f.name),
        })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return pd.DataFrame(columns=["patient_id", "report_path", "report_text_en"])
    df = df.dropna(subset=["patient_id"]).drop_duplicates("patient_id", keep="first")

    # 预读取前2w个字符
    texts = []
    for p in df["report_path"].tolist():
        try:
            t = Path(p).read_text(encoding="utf-8", errors="ignore")
            texts.append(t[:20000])
        except Exception:
            texts.append("")
    df["report_text_en"] = texts
    return df


def build_embedding_index(emb_root: Path):#遍历embedding
    emb_files = find_files(emb_root, [".h5", ".h5p", ".npy", ".npz", ".pt"])
    rows = []
    for f in emb_files:
        rows.append({
            "embedding_path": str(f),
            "patient_id": normalize_patient_id(f.name),
        })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return pd.DataFrame(columns=["patient_id", "embedding_path"])
    df = df.dropna(subset=["patient_id"]).drop_duplicates("patient_id", keep="first")
    return df


def load_possible_label_tables(dict_root: Path):#
    rows = []
    for f in find_files(dict_root, [".csv", ".json"]):
        rows.append({"file": str(f), "name": f.name})
    return pd.DataFrame(rows)


def main():
    text_root = RAW_DIR / "text"
    emb_root = RAW_DIR / "uni_embeddings"
    dict_root = RAW_DIR / "dictionaries"

    text_df = build_text_index(text_root)
    emb_df = build_embedding_index(emb_root)
    label_files_df = load_possible_label_tables(dict_root)

    #合并index
    master = pd.merge(text_df, emb_df, on="patient_id", how="outer")

    # # 先放空标签列，之后再映射填充
    # if "margin_label" not in master.columns:
    #     master["margin_label"] = pd.NA
    # if "metastasis_label" not in master.columns:
    #     master["metastasis_label"] = pd.NA
    # if "site" not in master.columns:
    #     master["site"] = pd.NA

    master = master.sort_values("patient_id").reset_index(drop=True)

    master.to_csv(PROCESSED_DIR / "master_table_stage1.csv", index=False, encoding="utf-8")
    label_files_df.to_csv(INTERIM_DIR / "candidate_label_files.csv", index=False, encoding="utf-8")

    print("Saved:", PROCESSED_DIR / "master_table_stage1.csv")
    print("Saved:", INTERIM_DIR / "candidate_label_files.csv")
    print("master_table_stage1 shape:", master.shape)
    print(master.head(10))


if __name__ == "__main__":
    main()