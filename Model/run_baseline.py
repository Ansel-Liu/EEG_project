import os
import glob
import argparse
import numpy as np
import pandas as pd

# =========================
# HELPERS
# =========================
def format_size(size_bytes):
    """Format file sizes into human-readable units."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return round(size_bytes, 2), unit
        size_bytes /= 1024
    return round(size_bytes, 2), "PB"

def safe_numeric_stats(arr):
    """Safely compute statistics for numeric arrays, handling NaNs and Infs."""
    stats = {
        "min": None, "max": None, "mean": None, "std": None,
        "nan_count": None, "inf_count": None, "zero_count": None,
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
    """Extract patient ID from the filename."""
    base = os.path.basename(filename)
    return base.split("_")[0] if "_" in base else base


# =========================
# MAIN EXECUTION
# =========================
def main(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Scanning directory: {data_dir}")
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    
    if not npz_files and not parquet_files:
        print("Warning: No .npz or .parquet files found in the specified directory.")
        return

    # 1) FOLDER SUMMARY
    folder_summary = []
    total_npz_size = sum(os.path.getsize(f) for f in npz_files)
    total_parquet_size = sum(os.path.getsize(f) for f in parquet_files)

    folder_summary.append({
        "data_dir": data_dir,
        "num_npz_files": len(npz_files),
        "num_parquet_files": len(parquet_files),
        "total_npz_size_bytes": total_npz_size,
        "total_parquet_size_bytes": total_parquet_size,
        "total_size_bytes": total_npz_size + total_parquet_size,
    })
    pd.DataFrame(folder_summary).to_csv(os.path.join(output_dir, "folder_summary.csv"), index=False)

    # 2) FILE LIST REPORT
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
    pd.DataFrame(file_list_rows).to_csv(os.path.join(output_dir, "file_list_report.csv"), index=False)

    # 3) NPZ REPORT
    npz_rows = []
    for npz_path in npz_files:
        try:
            data = np.load(npz_path)
            for key in data.keys():
                arr = data[key]
                stats = safe_numeric_stats(arr)
                npz_rows.append({
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
                    **stats,
                    "read_status": "ok",
                    "error_message": None,
                })
        except Exception as e:
            npz_rows.append({
                "filename": os.path.basename(npz_path),
                "full_path": npz_path,
                "patient_id": extract_patient_id(npz_path),
                "read_status": "error",
                "error_message": str(e),
            })
    pd.DataFrame(npz_rows).to_csv(os.path.join(output_dir, "npz_report.csv"), index=False)

    # 4) PARQUET BASIC, MISSING, AND CLASS REPORTS
    parquet_rows = []
    missing_rows = []
    class_rows = []

    for parquet_path in parquet_files:
        try:
            df = pd.read_parquet(parquet_path)
            
            # Basic info
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

            # Missing values analysis
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

            # Class distribution analysis
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
                "read_status": "error",
                "error_message": str(e),
            })

    pd.DataFrame(parquet_rows).to_csv(os.path.join(output_dir, "parquet_report.csv"), index=False)
    pd.DataFrame(missing_rows).to_csv(os.path.join(output_dir, "parquet_missing_report.csv"), index=False)
    pd.DataFrame(class_rows).to_csv(os.path.join(output_dir, "parquet_class_distribution.csv"), index=False)

    # 5) PATIENT-LEVEL SUMMARY
    patient_rows = []
    all_patients = sorted(set([extract_patient_id(f) for f in npz_files + parquet_files]))

    for patient in all_patients:
        patient_npz = [f for f in npz_files if extract_patient_id(f) == patient]
        patient_parquet = [f for f in parquet_files if extract_patient_id(f) == patient]

        total_windows = None
        total_rows_meta = None
        total_missing_meta = None

        if patient_parquet:
            try:
                dfp = pd.read_parquet(patient_parquet[0])
                total_rows_meta = int(len(dfp))
                total_missing_meta = int(dfp.isna().sum().sum())
            except Exception:
                pass

        if patient_npz:
            try:
                d = np.load(patient_npz[0])
                first_key = list(d.keys())[0]
                arr = d[first_key]
                total_windows = int(arr.shape[0]) if arr.ndim >= 1 else None
            except Exception:
                pass

        patient_rows.append({
            "patient_id": patient,
            "num_npz_files": len(patient_npz),
            "num_parquet_files": len(patient_parquet),
            "total_windows_from_npz": total_windows,
            "total_rows_from_parquet": total_rows_meta,
            "total_missing_values_in_parquet": total_missing_meta,
        })

    pd.DataFrame(patient_rows).to_csv(os.path.join(output_dir, "patient_summary.csv"), index=False)

    print("\nGenerated files:")
    print("- folder_summary.csv\n- file_list_report.csv\n- npz_report.csv\n- parquet_report.csv\n- parquet_missing_report.csv\n- parquet_class_distribution.csv\n- patient_summary.csv")
    print(f"\nDone. Profiling reports successfully saved in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate EDA statistics for EEG datasets (.npz & .parquet)")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to the directory containing raw data")
    parser.add_argument("--output_dir", type=str, default="./eda_reports", help="Directory to save CSV reports")
    
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)