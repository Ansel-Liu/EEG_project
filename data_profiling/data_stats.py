import os
import glob
import numpy as np
import pandas as pd


# =========================
# PATHS
# =========================
DATA_DIR = "/export/hhome/ricse/Epilepsy"
OUTPUT_DIR = "/export/hhome/ricse06/deppLearning/UAB-Deep-Learning-Epilepsy/Ali"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# HELPERS
# =========================
def format_size(size_bytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return round(size_bytes, 2), unit
        size_bytes /= 1024
    return round(size_bytes, 2), "PB"


def safe_numeric_stats(arr):
    stats = {
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "nan_count": None,
        "inf_count": None,
        "zero_count": None,
    }

    if not np.issubdtype(arr.dtype, np.number):
        return stats

    if np.issubdtype(arr.dtype, np.floating):
        stats["nan_count"] = int(np.isnan(arr).sum())
        stats["inf_count"] = int(np.isinf(arr).sum())
        finite_mask = np.isfinite(arr)
        if finite_mask.any():
            finite_vals = arr[finite_mask]
            stats["min"] = float(np.min(finite_vals))
            stats["max"] = float(np.max(finite_vals))
            stats["mean"] = float(np.mean(finite_vals))
            stats["std"] = float(np.std(finite_vals))
        else:
            stats["min"] = None
            stats["max"] = None
            stats["mean"] = None
            stats["std"] = None
        stats["zero_count"] = int(np.sum(arr == 0))
    else:
        stats["nan_count"] = 0
        stats["inf_count"] = 0
        stats["min"] = float(np.min(arr))
        stats["max"] = float(np.max(arr))
        stats["mean"] = float(np.mean(arr))
        stats["std"] = float(np.std(arr))
        stats["zero_count"] = int(np.sum(arr == 0))

    return stats


def extract_patient_id(filename):
    base = os.path.basename(filename)
    return base.split("_")[0] if "_" in base else base


# =========================
# SCAN FILES
# =========================
npz_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
parquet_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))

# =========================
# 1) FOLDER SUMMARY
# =========================
folder_summary = []

total_npz_size = sum(os.path.getsize(f) for f in npz_files) if npz_files else 0
total_parquet_size = sum(os.path.getsize(f) for f in parquet_files) if parquet_files else 0

folder_summary.append({
    "data_dir": DATA_DIR,
    "num_npz_files": len(npz_files),
    "num_parquet_files": len(parquet_files),
    "total_npz_size_bytes": total_npz_size,
    "total_parquet_size_bytes": total_parquet_size,
    "total_size_bytes": total_npz_size + total_parquet_size,
})

pd.DataFrame(folder_summary).to_csv(
    os.path.join(OUTPUT_DIR, "folder_summary.csv"), index=False
)

# =========================
# 2) FILE LIST REPORT
# =========================
file_list_rows = []

for f in npz_files + parquet_files:
    size_value, size_unit = format_size(os.path.getsize(f))
    file_list_rows.append({
        "filename": os.path.basename(f),
        "full_path": f,
        "patient_id": extract_patient_id(f),
        "extension": os.path.splitext(f)[1],
        "size_bytes": os.path.getsize(f),
        "size_value": size_value,
        "size_unit": size_unit,
    })

pd.DataFrame(file_list_rows).to_csv(
    os.path.join(OUTPUT_DIR, "file_list_report.csv"), index=False
)

# =========================
# 3) NPZ REPORT
# =========================
npz_rows = []

for npz_path in npz_files:
    try:
        data = np.load(npz_path)
        keys = list(data.keys())

        for key in keys:
            arr = data[key]
            stats = safe_numeric_stats(arr)

            row = {
                "filename": os.path.basename(npz_path),
                "full_path": npz_path,
                "patient_id": extract_patient_id(npz_path),
                "array_key": key,
                "dtype": str(arr.dtype),
                "ndim": arr.ndim,
                "shape": str(arr.shape),
                "num_elements": int(arr.size),
                "samples_dim0": int(arr.shape[0]) if arr.ndim >= 1 else None,
                "channels_dim1": int(arr.shape[1]) if arr.ndim >= 2 else None,
                "length_dim2": int(arr.shape[2]) if arr.ndim >= 3 else None,
                "min": stats["min"],
                "max": stats["max"],
                "mean": stats["mean"],
                "std": stats["std"],
                "nan_count": stats["nan_count"],
                "inf_count": stats["inf_count"],
                "zero_count": stats["zero_count"],
                "read_status": "ok",
                "error_message": None,
            }
            npz_rows.append(row)

    except Exception as e:
        npz_rows.append({
            "filename": os.path.basename(npz_path),
            "full_path": npz_path,
            "patient_id": extract_patient_id(npz_path),
            "array_key": None,
            "dtype": None,
            "ndim": None,
            "shape": None,
            "num_elements": None,
            "samples_dim0": None,
            "channels_dim1": None,
            "length_dim2": None,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "nan_count": None,
            "inf_count": None,
            "zero_count": None,
            "read_status": "error",
            "error_message": str(e),
        })

df_npz = pd.DataFrame(npz_rows)
df_npz.to_csv(os.path.join(OUTPUT_DIR, "npz_report.csv"), index=False)

# =========================
# 4) PARQUET BASIC REPORT
# =========================
parquet_rows = []
missing_rows = []
class_rows = []

for parquet_path in parquet_files:
    try:
        df = pd.read_parquet(parquet_path)

        parquet_rows.append({
            "filename": os.path.basename(parquet_path),
            "full_path": parquet_path,
            "patient_id": extract_patient_id(parquet_path),
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": " | ".join(df.columns.astype(str)),
            "read_status": "ok",
            "error_message": None,
        })

        for col in df.columns:
            missing_rows.append({
                "filename": os.path.basename(parquet_path),
                "patient_id": extract_patient_id(parquet_path),
                "column_name": col,
                "dtype": str(df[col].dtype),
                "missing_count": int(df[col].isna().sum()),
                "missing_percent": float(df[col].isna().mean() * 100.0),
                "non_missing_count": int(df[col].notna().sum()),
            })

        if "class" in df.columns:
            vc = df["class"].value_counts(dropna=False)
            for cls, cnt in vc.items():
                class_rows.append({
                    "filename": os.path.basename(parquet_path),
                    "patient_id": extract_patient_id(parquet_path),
                    "class_value": cls,
                    "count": int(cnt),
                    "percent": float(cnt / len(df) * 100.0) if len(df) > 0 else 0.0,
                })

    except Exception as e:
        parquet_rows.append({
            "filename": os.path.basename(parquet_path),
            "full_path": parquet_path,
            "patient_id": extract_patient_id(parquet_path),
            "rows": None,
            "columns": None,
            "column_names": None,
            "read_status": "error",
            "error_message": str(e),
        })

df_parquet = pd.DataFrame(parquet_rows)
df_parquet.to_csv(os.path.join(OUTPUT_DIR, "parquet_report.csv"), index=False)

df_missing = pd.DataFrame(missing_rows)
df_missing.to_csv(os.path.join(OUTPUT_DIR, "parquet_missing_report.csv"), index=False)

df_class = pd.DataFrame(class_rows)
df_class.to_csv(os.path.join(OUTPUT_DIR, "parquet_class_distribution.csv"), index=False)

# =========================
# 5) PATIENT-LEVEL SUMMARY
# =========================
patient_rows = []

all_patients = sorted(set([extract_patient_id(f) for f in npz_files + parquet_files]))

for patient in all_patients:
    patient_npz = [f for f in npz_files if extract_patient_id(f) == patient]
    patient_parquet = [f for f in parquet_files if extract_patient_id(f) == patient]

    total_windows = None
    total_rows_meta = None
    total_missing_meta = None

    # try reading parquet totals
    if patient_parquet:
        try:
            dfp = pd.read_parquet(patient_parquet[0])
            total_rows_meta = int(len(dfp))
            total_missing_meta = int(dfp.isna().sum().sum())
        except Exception:
            total_rows_meta = None
            total_missing_meta = None

    # try reading npz totals
    if patient_npz:
        try:
            d = np.load(patient_npz[0])
            first_key = list(d.keys())[0]
            arr = d[first_key]
            total_windows = int(arr.shape[0]) if arr.ndim >= 1 else None
        except Exception:
            total_windows = None

    patient_rows.append({
        "patient_id": patient,
        "num_npz_files": len(patient_npz),
        "num_parquet_files": len(patient_parquet),
        "total_windows_from_npz": total_windows,
        "total_rows_from_parquet": total_rows_meta,
        "total_missing_values_in_parquet": total_missing_meta,
    })

pd.DataFrame(patient_rows).to_csv(
    os.path.join(OUTPUT_DIR, "patient_summary.csv"), index=False
)

print("Done. CSV reports saved in:")
print(OUTPUT_DIR)
print("\nGenerated files:")
print("- folder_summary.csv")
print("- file_list_report.csv")
print("- npz_report.csv")
print("- parquet_report.csv")
print("- parquet_missing_report.csv")
print("- parquet_class_distribution.csv")
print("- patient_summary.csv")
