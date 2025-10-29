#!/usr/bin/env python3
# QC compare CSV folder trees: expected vs actual
# Edit the three paths below or override them via CLI args.
# Usage (example):
#   py -3 qc_compare_csv_folders.py -e `C:\expected_csvs` -a `C:\actual_csvs` -o `C:\qc_report_root` --recursive --numeric-tolerance 1e-6

# ======= TOP-LEVEL PATH CONFIGURATION (EDIT THESE) ===========================
EXPECTED_DIR = r"C:\Users\rooneyz\Documents\expected_csvs"
ACTUAL_DIR = r"C:\Users\rooneyz\Documents\actual_csvs"
OUTPUT_DIR = r"C:\Users\rooneyz\Documents\qc_report_root"
# ===========================================================================

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import sys
import re

def find_csv_files(root: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.csv" if recursive else "*.csv"
    return sorted([p for p in root.glob(pattern) if p.is_file()])

def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_csv(path, encoding="latin-1", low_memory=False)

def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def safe_value(v) -> Any:
    if pd.isna(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        if np.isfinite(v):
            return float(v)
        return None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v

def compare_dataframes(expected: pd.DataFrame,
                       actual: pd.DataFrame,
                       numeric_tol: float,
                       max_diff_rows: int) -> Dict[str, Any]:
    erows, ecols = expected.shape
    arows, acols = actual.shape

    expected_cols = list(expected.columns)
    actual_cols = list(actual.columns)

    cols_equal = expected_cols == actual_cols
    cols_set_equal = set(expected_cols) == set(actual_cols)
    only_in_expected = [c for c in expected_cols if c not in actual_cols]
    only_in_actual = [c for c in actual_cols if c not in expected_cols]

    if cols_set_equal and not cols_equal:
        actual = actual.reindex(columns=expected_cols)

    min_rows = min(erows, arows)
    extra_rows_expected = max(0, erows - arows)
    extra_rows_actual = max(0, arows - erows)

    comp_cols = [c for c in expected_cols if c in actual.columns]
    mismatch_cells = 0
    mismatch_rows_set = set()
    mismatch_details = []

    # capture dtypes for compared columns
    expected_dtypes = {c: str(expected[c].dtype) for c in comp_cols}
    actual_dtypes = {c: str(actual[c].dtype) for c in comp_cols}

    if comp_cols and min_rows > 0:
        exp_sub = expected.loc[:min_rows-1, comp_cols].reset_index(drop=True)
        act_sub = actual.loc[:min_rows-1, comp_cols].reset_index(drop=True)

        for col in comp_cols:
            ecol = exp_sub[col]
            acol = act_sub[col]

            if is_numeric_series(ecol) and is_numeric_series(acol):
                evals = ecol.to_numpy(dtype='float64', copy=False)
                avals = acol.to_numpy(dtype='float64', copy=False)
                both_nan = np.isnan(evals) & np.isnan(avals)
                finite_mask = ~(np.isnan(evals) | np.isnan(avals))
                diff = np.full(evals.shape, False, dtype=bool)
                if finite_mask.any():
                    diff[finite_mask] = ~(np.isclose(evals[finite_mask], avals[finite_mask], atol=numeric_tol, rtol=0))
                nan_mismatch = (~both_nan) & (np.isnan(evals) ^ np.isnan(avals))
                col_mismatch_mask = diff | nan_mismatch
            else:
                col_mismatch_mask = ~(ecol.fillna("__NA_REP__").astype(str) == acol.fillna("__NA_REP__").astype(str))

            indices = np.nonzero(col_mismatch_mask)[0]
            mismatch_cells += len(indices)
            expected_dtype_str = str(ecol.dtype)
            actual_dtype_str = str(acol.dtype)
            for i in indices:
                pandas_row = int(i)
                mismatch_rows_set.add(pandas_row)
                if len(mismatch_details) < max_diff_rows:
                    # map pandas 0-based data row -> CSV file line (header = line 1)
                    file_line = pandas_row + 2

                    # determine reason
                    if is_numeric_series(ecol) and is_numeric_series(acol):
                        if nan_mismatch[i]:
                            reason = "nan_mismatch"
                        elif finite_mask[i] and diff[i]:
                            reason = "numeric_diff"
                        else:
                            reason = "value_mismatch"
                    else:
                        if expected_dtype_str != actual_dtype_str:
                            reason = "dtype_mismatch"
                        else:
                            reason = "string_mismatch"

                    mismatch_details.append({
                        "row_index": file_line,     # human-friendly CSV file line number
                        "pandas_row": pandas_row,   # pandas 0-based row index
                        "column": col,
                        "expected": safe_value(ecol.iat[pandas_row]),
                        "actual": safe_value(acol.iat[pandas_row]),
                        "reason": reason
                    })

    # column-level dtype mismatches (reported even if values matched)
    dtype_mismatches = []
    for c in comp_cols:
        ed = expected_dtypes.get(c)
        ad = actual_dtypes.get(c)
        if ed != ad:
            dtype_mismatches.append({"column": c, "expected_dtype": ed, "actual_dtype": ad})

    n_mismatch_rows = len(mismatch_rows_set)
    result = {
        "expected_rows": int(erows),
        "actual_rows": int(arows),
        "common_rows_compared": int(min_rows),
        "extra_rows_expected": int(extra_rows_expected),
        "extra_rows_actual": int(extra_rows_actual),
        "expected_columns": expected_cols,
        "actual_columns": actual_cols,
        "columns_equal": bool(cols_equal),
        "columns_set_equal": bool(cols_set_equal),
        "only_in_expected": only_in_expected,
        "only_in_actual": only_in_actual,
        "n_mismatch_rows": int(n_mismatch_rows),
        "n_mismatch_cells": int(mismatch_cells),
        "mismatch_samples": mismatch_details,
        "expected_dtypes": expected_dtypes,
        "actual_dtypes": actual_dtypes,
        "dtype_mismatches": dtype_mismatches
    }
    return result

def write_per_file_diff_csv(out_csv: Path, mismatches: List[Dict[str, Any]]):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["row_index", "pandas_row", "column", "expected", "actual", "reason"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in mismatches:
            out = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(out)

def _sanitize_sheet_name(name: str) -> str:
    # remove invalid excel sheet chars and limit length to 31
    sanitized = re.sub(r'[:\\/*?\[\]]', '_', name)
    return sanitized[:31]

def write_excel_from_reports(reports_dir: Path, excel_path: Path):
    reports = []
    for jf in sorted(reports_dir.glob("*.json")):
        try:
            with jf.open("r", encoding="utf-8") as fh:
                reports.append(json.load(fh))
        except Exception:
            continue

    if not reports:
        return

    summary_rows = []
    for r in reports:
        metrics = r.get("metrics", {})
        summary_rows.append({
            "relative_path": r.get("relative_path"),
            "status": r.get("status"),
            "expected_rows": metrics.get("expected_rows"),
            "actual_rows": metrics.get("actual_rows"),
            "common_rows_compared": metrics.get("common_rows_compared"),
            "extra_rows_expected": metrics.get("extra_rows_expected"),
            "extra_rows_actual": metrics.get("extra_rows_actual"),
            "n_mismatch_rows": metrics.get("n_mismatch_rows"),
            "n_mismatch_cells": metrics.get("n_mismatch_cells"),
            "timestamp": r.get("timestamp")
        })

    excel_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(str(excel_path), engine="openpyxl") as writer:
        # Summary sheet
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_excel(writer, sheet_name="summary", index=False)

        # One sheet per report: metrics then mismatch samples (if any)
        for r in reports:
            rel = r.get("relative_path", "unknown")
            metrics = r.get("metrics", {}) or {}
            mismatches = metrics.get("mismatch_samples") or []

            base = _sanitize_sheet_name(rel.replace("/", "__"))
            sheet_name = base
            idx = 1
            while sheet_name in writer.book.sheetnames:
                sheet_name = (base[:27] + f"_{idx}") if len(base) > 27 else f"{base}_{idx}"
                idx += 1

            metric_rows = []
            for k in sorted(metrics.keys()):
                v = metrics[k]
                val = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                metric_rows.append({"metric": k, "value": val})
            df_metrics = pd.DataFrame(metric_rows)
            df_metrics.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)

            df_mismatches = pd.DataFrame(mismatches)
            if not df_mismatches.empty:
                startrow = len(df_metrics) + 2
                df_mismatches.to_excel(writer, sheet_name=sheet_name, index=False, startrow=startrow)

def main():
    parser = argparse.ArgumentParser(description="QC compare two folders of CSVs.")
    parser.add_argument("-e", "--expected-dir", required=False, help="Path to expected CSV folder (overrides top-of-file `EXPECTED_DIR`)")
    parser.add_argument("-a", "--actual-dir", required=False, help="Path to actual CSV folder (overrides top-of-file `ACTUAL_DIR`)")
    parser.add_argument("-o", "--output-dir", required=False, help="Path to write QC reports (overrides top-of-file `OUTPUT_DIR`)")
    parser.add_argument("--recursive", action="store_true", help="Traverse subfolders")
    parser.add_argument("--numeric-tolerance", type=float, default=0.0, help="Absolute tolerance for numeric comparisons")
    parser.add_argument("--max-diff-rows", type=int, default=200, help="Maximum per-file mismatch rows to record")
    args = parser.parse_args()

    chosen_expected = args.expected_dir or EXPECTED_DIR
    chosen_actual = args.actual_dir or ACTUAL_DIR
    chosen_output = args.output_dir or OUTPUT_DIR

    expected_root = Path(chosen_expected).expanduser().resolve()
    actual_root = Path(chosen_actual).expanduser().resolve()
    out_root = Path(chosen_output).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Place all generated outputs under a single `qc_reports` folder
    qc_base = out_root / "qc_reports"
    reports_dir = qc_base / "reports"
    diffs_dir = qc_base / "diffs"
    qc_base.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    diffs_dir.mkdir(parents=True, exist_ok=True)

    if not expected_root.exists():
        print(f"Expected folder not found: `{expected_root}`", file=sys.stderr)
        return
    if not actual_root.exists():
        print(f"Actual folder not found: `{actual_root}`", file=sys.stderr)
        return

    expected_files = find_csv_files(expected_root, args.recursive)
    actual_files = find_csv_files(actual_root, args.recursive)

    exp_map = {p.relative_to(expected_root).as_posix().casefold(): p for p in expected_files}
    act_map = {p.relative_to(actual_root).as_posix().casefold(): p for p in actual_files}

    all_norm_rel = sorted(set(exp_map.keys()) | set(act_map.keys()))

    summary_rows = []
    combined_reports = []

    for norm_rel in all_norm_rel:
        exp_path = exp_map.get(norm_rel)
        act_path = act_map.get(norm_rel)

        if exp_path is not None:
            rel_print = exp_path.relative_to(expected_root).as_posix()
        else:
            rel_print = act_path.relative_to(actual_root).as_posix()

        report: Dict[str, Any] = {"relative_path": rel_print, "timestamp": datetime.utcnow().isoformat()}

        if exp_path is None:
            report.update({"status": "extra_in_actual", "message": f"File present in actual only: `{rel_print}`"})
            print(f"[EXTRA] {rel_print}")
            summary_rows.append({
                "relative_path": rel_print,
                "status": "extra_in_actual",
                "expected_rows": None,
                "actual_rows": None,
                "n_mismatch_rows": None,
                "n_mismatch_cells": None
            })
            combined_reports.append(report)
            continue

        if act_path is None:
            report.update({"status": "missing_in_actual", "message": f"Missing in actual: `{rel_print}`"})
            print(f"[MISSING] {rel_print}")
            summary_rows.append({
                "relative_path": rel_print,
                "status": "missing_in_actual",
                "expected_rows": None,
                "actual_rows": None,
                "n_mismatch_rows": None,
                "n_mismatch_cells": None
            })
            combined_reports.append(report)
            continue

        try:
            exp_df = read_csv_safe(exp_path)
            act_df = read_csv_safe(act_path)
        except Exception as ex:
            report.update({"status": "read_error", "error": str(ex)})
            print(f"[ERROR_READ] {rel_print}: {ex}")
            summary_rows.append({
                "relative_path": rel_print,
                "status": "read_error",
                "expected_rows": None,
                "actual_rows": None,
                "n_mismatch_rows": None,
                "n_mismatch_cells": None
            })
            combined_reports.append(report)
            continue

        comp = compare_dataframes(exp_df, act_df, args.numeric_tolerance, args.max_diff_rows)
        # fail if any row mismatches, extra rows, column set differences, or dtype mismatches
        status = "pass" if (comp["n_mismatch_rows"] == 0 and comp["extra_rows_expected"] == 0 and comp["extra_rows_actual"] == 0 and comp["columns_set_equal"] and not comp.get("dtype_mismatches")) else "fail"
        report.update({"status": status, "metrics": comp})

        out_json = reports_dir / (rel_print.replace("/", "__") + ".json")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as jf:
            json.dump(report, jf, indent=2, ensure_ascii=False, default=str)

        if comp["mismatch_samples"]:
            out_diff_csv = diffs_dir / (rel_print.replace("/", "__") + "__mismatches.csv")
            write_per_file_diff_csv(out_diff_csv, comp["mismatch_samples"])

        summary_rows.append({
            "relative_path": rel_print,
            "status": status,
            "expected_rows": comp.get("expected_rows"),
            "actual_rows": comp.get("actual_rows"),
            "common_rows_compared": comp.get("common_rows_compared"),
            "extra_rows_expected": comp.get("extra_rows_expected"),
            "extra_rows_actual": comp.get("extra_rows_actual"),
            "n_mismatch_rows": comp.get("n_mismatch_rows"),
            "n_mismatch_cells": comp.get("n_mismatch_cells"),
        })
        combined_reports.append(report)
        print(f"[{status.upper()}] {rel_print}: mismatched_rows={comp['n_mismatch_rows']} mismatched_cells={comp['n_mismatch_cells']} dtype_mismatches={len(comp.get('dtype_mismatches', []))}")

    # ensure summary CSV always has a header and is written (even if no rows)
    summary_columns = [
        "relative_path",
        "status",
        "expected_rows",
        "actual_rows",
        "common_rows_compared",
        "extra_rows_expected",
        "extra_rows_actual",
        "n_mismatch_rows",
        "n_mismatch_cells",
    ]
    df_summary = pd.DataFrame(summary_rows, columns=summary_columns)
    summary_path = qc_base / "qc_summary.csv"
    df_summary.to_csv(str(summary_path), index=False)
    print(f"- Summary CSV written to `{summary_path}` (rows: {len(df_summary)})")

    combined_path = qc_base / "qc_combined_reports.json"
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump({"generated_at": datetime.utcnow().isoformat(), "reports": combined_reports}, f, indent=2, ensure_ascii=False)

    try:
        write_excel_from_reports(reports_dir, qc_base / "qc_reports.xlsx")
        print(f"- Excel workbook: `{qc_base / 'qc_reports.xlsx'}`")
    except Exception as ex:
        print(f"Failed to write Excel workbook: {ex}", file=sys.stderr)

    print("\nQC complete.")
    print(f"- Summary CSV: `{summary_path}`")
    print(f"- Per-file JSON reports: `{reports_dir}`")
    print(f"- Per-file diffs (some files): `{diffs_dir}`")
    print(f"- Combined JSON: `{combined_path}`")

if __name__ == "__main__":
    main()
