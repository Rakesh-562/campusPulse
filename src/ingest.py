# src/ingest.py
import pandas as pd
from scipy.io import arff
import os

def load_lifestyle(path="data/raw/student_lifestyle_dataset.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    print(f"Lifestyle loaded: {df.shape}")
    print(df.head())

    return df

def load_uci(mat="data/raw/student-mat.csv",
             por="data/raw/student-por.csv"):
    for path in (mat, por):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    df_mat = pd.read_csv(mat, sep=";")
    df_por = pd.read_csv(por, sep=";")
    df_mat["subject"] = "math"
    df_por["subject"] = "portuguese"
    df = pd.concat([df_mat, df_por], ignore_index=True)
    print(f"UCI loaded: {df.shape}")
    return df

def load_mental_health(path="data/raw/Student Mental health.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    print(f"Mental Health loaded: {df.shape}")
    return df

if __name__ == "__main__":
    df_ls = load_lifestyle()
    df_uc = load_uci()
    df_mh = load_mental_health()
    print(df_ls.head(2))
    print()
    print(df_uc.head(2))
    print()
    print(df_mh.head(2))