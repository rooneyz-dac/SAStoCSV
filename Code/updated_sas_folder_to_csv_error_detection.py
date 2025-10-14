# Updated Script with Positional Input and Dynamic Output Directory
# Precedence: positional input > --input-dir > global INPUT_DIR
# Output defaults to <parent_of_input>/DAC_CSV unless --output-dir is provided


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert a folder of .sas7bdat files to CSV while capturing SAS metadata.

Edit the CONFIGURATION block below (no .env, no CLI args).

Outputs:
  - <OUTPUT_DIR>/<relative_path>/<dataset>.csv
  - <OUTPUT_DIR>/<relative_path>/<dataset>__metadata.json
  - <OUTPUT_DIR>/metadata_summary.csv      (per-variable metadata across datasets)
  - <OUTPUT_DIR>/value_labels.csv          (flattened code->label mappings)
  - <OUTPUT_DIR>/combined_codebook.json    (single aggregated JSON for all datasets)
  - <OUTPUT_DIR>/conversion_errors.csv     (optional; files that failed with errors)

Requirements:
  pip install pyreadstat pandas
"""
""" Recommended Edits:
INPUT_DIR  = r"C:\data\sas" # <--- CHANGE
OUTPUT_DIR = r"C:\data\out" # <--- CHANGE
APPLY_VALUE_LABELS = True
SAS_CATALOG_DIR = r"C:\data\catalogs"  # or point SAS_CATALOG_FILE to a single .sas7bcat
"""



# ==== CONFIGURATION (EDIT THESE) ==============================================
INPUT_DIR = r"C:\Users\rooneyz\Documents\TestData\MirumTestData"   # ← change this
OUTPUT_DIR = r"\output"      # ← and this

# Recursion & performance
RECURSIVE = True             # traverse subfolders
CHUNKSIZE = None             # e.g., 250000 for big files, or None to read all rows

# CSV formatting
NA_REP = ""                  # value to write for missing values in CSV
ENCODING = None              # rarely needed; pyreadstat auto-detects (e.g., "latin-1")

# SAS value labels (formats)
APPLY_VALUE_LABELS = False   # True to replace codes using SAS formats via catalog(s)
SAS_CATALOG_FILE = None      # e.g., r"C:\path\to\formats.sas7bcat" (single catalog)
SAS_CATALOG_DIR  = r"FilePath"      # e.g., r"C:\path\to\catalogs" (auto-match by file stem)
SAS_FORMATS_AS_CATEGORY = True           # labeled columns become pandas categoricals
SAS_FORMATS_AS_ORDERED = False           # ordered categorical for formats with order

# Error report
WRITE_ERROR_CSV = True                      # set to False to skip writing the error CSV
ERROR_CSV_FILENAME = "00_conversion_errors.csv"  # change the file name if desired
# ==============================================================================

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone

import pandas as pd
import argparse

try:
    import pyreadstat
except ImportError as e:
    raise SystemExit("Missing dependency: pyreadstat. Install with: pip install pyreadstat") from e


# ------------------------- Serialization utilities ---------------------------
def safe_serialize(obj):
    """Safely convert non-JSON-serializable objects to built-in types/strings."""
    import numpy as np
    import datetime as dt

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [safe_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (pd.Timestamp, dt.datetime, dt.date, dt.time)):
        return obj.isoformat()
    if isinstance(obj, (pd.Timedelta, dt.timedelta)):
        return str(obj)
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.astype(object).map(safe_serialize).tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="list")
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def extract_meta_dict(meta) -> Dict[str, Any]:
    """
    Build a comprehensive, JSON-serializable metadata dict from a pyreadstat meta object.
    """
    md: Dict[str, Any] = {}
    md["dataset"] = {
        "file_label": getattr(meta, "file_label", None),
        "file_encoding": getattr(meta, "file_encoding", None),
        "table_name": getattr(meta, "table_name", None),
        "number_rows": getattr(meta, "number_rows", None),
        "number_columns": getattr(meta, "number_columns", None),
        "notes": getattr(meta, "notes", None),
        "creation_time": getattr(meta, "creation_time", None),
        "modified_time": getattr(meta, "modified_time", None),
        "compression": getattr(meta, "compression", None),
    }

    # Variable-level structures
    column_names = getattr(meta, "column_names", None) or []
    column_labels = getattr(meta, "column_labels", None) or []
    column_formats = getattr(meta, "column_formats", None) or []
    variable_types = getattr(meta, "variable_types", None)  # dict(var -> "numeric"/"string"), if available
    column_display_width = getattr(meta, "column_display_width", None) or []

    # Value labels mapping per variable (if available)
    value_labels_by_var: Dict[str, Dict[Any, str]] = {}
    vv = getattr(meta, "variable_value_labels", None)
    if isinstance(vv, dict):
        # Some builds provide var->dict; others var->labelset_name
        any_is_dict = any(isinstance(v, dict) for v in vv.values())
        if any_is_dict:
            for var, mapping in vv.items():
                if isinstance(mapping, dict):
                    value_labels_by_var[var] = mapping

    if not value_labels_by_var:
        var_to_labelset = getattr(meta, "variable_to_label", None)
        if not isinstance(var_to_labelset, dict):
            if isinstance(vv, dict) and all(isinstance(v, str) for v in vv.values()):
                var_to_labelset = vv
        labelsets = getattr(meta, "value_labels", None)
        if isinstance(var_to_labelset, dict) and isinstance(labelsets, dict):
            for var, labelset_name in var_to_labelset.items():
                mapping = labelsets.get(labelset_name)
                if isinstance(mapping, dict):
                    value_labels_by_var[var] = mapping

    variables: List[Dict[str, Any]] = []
    for i, name in enumerate(column_names):
        var_info: Dict[str, Any] = {
            "name": name,
            "label": column_labels[i] if i < len(column_labels) else None,
            "sas_format": column_formats[i] if i < len(column_formats) else None,
            "display_width": column_display_width[i] if i < len(column_display_width) else None,
            "sas_type": None,
            "value_labels": value_labels_by_var.get(name, None),
        }
        if isinstance(variable_types, dict):
            var_info["sas_type"] = variable_types.get(name)
        variables.append(var_info)

    md["variables"] = variables

    raw_labelsets = getattr(meta, "value_labels", None)
    if isinstance(raw_labelsets, dict) and raw_labelsets:
        md["labelsets"] = raw_labelsets

    return safe_serialize(md)


def write_json(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


# ------------------------------ Catalog helpers ------------------------------
def find_catalog_for_dataset(dataset_path: Path,
                             catalog_file_env: Optional[str],
                             catalog_dir_env: Optional[str]) -> Optional[Path]:
    """
    Determine which .sas7bcat to use for a given dataset.

    Priority:
      1) SAS_CATALOG_FILE (explicit)
      2) Sibling file with same stem: <dataset_dir>/<stem>.sas7bcat
      3) SAS_CATALOG_DIR: exact stem match, else first catalog found
    """
    # 1) Explicit single file
    if catalog_file_env:
        p = Path(catalog_file_env).expanduser()
        if p.exists():
            return p

    # 2) Sibling next to dataset (same stem)
    sibling = dataset_path.with_suffix(".sas7bcat")
    if sibling.exists():
        return sibling

    # 3) Directory search
    if catalog_dir_env:
        cd = Path(catalog_dir_env).expanduser()
        if cd.exists():
            exact = cd / f"{dataset_path.stem}.sas7bcat"
            if exact.exists():
                return exact
            for p in cd.glob("*.sas7bcat"):
                return p  # fallback: first available

    return None


# ----------------------------- Reporting helpers -----------------------------
def flatten_value_labels(dataset_name: str, rel_path: str, meta_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for var in meta_dict.get("variables", []):
        var_name = var.get("name")
        vmap = var.get("value_labels")
        if isinstance(vmap, dict) and vmap:
            for code, label in vmap.items():
                rows.append({
                    "dataset": dataset_name,
                    "relative_path": rel_path,
                    "variable": var_name,
                    "code": code,
                    "label": label
                })
    return rows


def append_metadata_summary(summary_rows: List[Dict[str, Any]],
                            dataset_name: str,
                            rel_path: str,
                            meta_dict: Dict[str, Any],
                            df_sample: Optional[pd.DataFrame] = None):
    pandas_dtypes = {}
    if df_sample is not None:
        pandas_dtypes = {c: str(t) for c, t in df_sample.dtypes.items()}

    for var in meta_dict.get("variables", []):
        name = var.get("name")
        has_vl = isinstance(var.get("value_labels"), dict) and len(var.get("value_labels")) > 0
        summary_rows.append({
            "dataset": dataset_name,
            "relative_path": rel_path,
            "variable": name,
            "label": var.get("label"),
            "sas_format": var.get("sas_format"),
            "sas_type": var.get("sas_type"),
            "pandas_dtype": pandas_dtypes.get(name),
            "has_value_labels": has_vl,
            "n_value_labels": len(var.get("value_labels") or {}),
        })


# ------------------------------- Core convert --------------------------------
def convert_one_sas(
    in_path: Path,
    out_csv_path: Path,
    out_meta_path: Path,
    apply_value_labels: bool = False,
    na_rep: str = "",
    chunksize: Optional[int] = None,
    encoding: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    """
    Convert one SAS7BDAT to CSV and write per-dataset JSON metadata.
    Returns: (metadata_dict, df_sample) where df_sample is a small head for dtype reporting.
    """
    ensure_parent_dir(out_csv_path)
    ensure_parent_dir(out_meta_path)

    df_sample: Optional[pd.DataFrame] = None
    meta_first = None

    read_kwargs = {
        "dates_as_pandas_datetime": True,
        "user_missing": True,
    }
    if encoding:
        read_kwargs["encoding"] = encoding

    # Attach catalog if requested & available
    if apply_value_labels:
        catalog_path = find_catalog_for_dataset(
            dataset_path=in_path,
            catalog_file_env=SAS_CATALOG_FILE,
            catalog_dir_env=SAS_CATALOG_DIR,
        )
        if catalog_path:
            read_kwargs["catalog_file"] = str(catalog_path)
            read_kwargs["formats_as_category"] = SAS_FORMATS_AS_CATEGORY
            read_kwargs["formats_as_ordered_category"] = SAS_FORMATS_AS_ORDERED
            print(f"  using catalog: {catalog_path}")
        else:
            print("  (info) No .sas7bcat catalog found for labels; proceeding without applying formats.")

    if chunksize and chunksize > 0:
        offset = 0
        first = True
        while True:
            df, meta = pyreadstat.read_sas7bdat(
                str(in_path),
                row_offset=offset,
                row_limit=chunksize,
                **read_kwargs
            )
            if meta_first is None:
                meta_first = meta
                df_sample = df.head(50).copy()

            if df.empty:
                break

            df.to_csv(
                out_csv_path,
                mode="w" if first else "a",
                index=False,
                header=first,
                na_rep=na_rep,
            )
            written_rows = len(df)
            offset += written_rows
            first = False
            # Encourage GC to drop chunk memory before next read
            del df
            import gc
            gc.collect()

            print(f"  wrote {written_rows} rows (offset={offset}) for {in_path.name}")
            if written_rows < chunksize:
                break
    else:
        df, meta = pyreadstat.read_sas7bdat(str(in_path), **read_kwargs)
        meta_first = meta
        df_sample = df.head(50).copy()
        df.to_csv(out_csv_path, index=False, na_rep=na_rep)

    meta_dict = extract_meta_dict(meta_first)
    write_json(out_meta_path, meta_dict)
    return meta_dict, df_sample


def find_sas_files(root: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.sas7bdat" if recursive else "*.sas7bdat"
    return [p for p in root.glob(pattern) if p.is_file()]


# ------------------------------------ main -----------------------------------
def main():
    # Allow runtime overrides; fall back to top-of-file constants
    parser = argparse.ArgumentParser(description="Convert folder of .sas7bdat files to CSV with metadata.")
    parser.add_argument("--input-dir", "-i", dest="input_dir", help="Path to input folder containing .sas7bdat files. Overrides top-level INPUT_DIR.")
    parser.add_argument("--output-dir", "-o", dest="output_dir", help="Path to output folder. Overrides top-level OUTPUT_DIR.")
    parser.add_argument("--dry-run", action="store_true", help="List files but don't convert")
    args = parser.parse_args()

    chosen_input = args.input_dir or INPUT_DIR
    chosen_output = args.output_dir or OUTPUT_DIR

    in_dir = Path(chosen_input).expanduser().resolve()
    out_dir = Path(chosen_output).expanduser().resolve()

    # Basic validation (require either the constant or an override)
    if not chosen_input or str(chosen_input).strip() == "" or "path\\to" in str(chosen_input) or "path/to" in str(chosen_input):
        print("Please provide an input folder via --input-dir or set INPUT_DIR at the top of the script.")
        return
    if not chosen_output or str(chosen_output).strip() == "" or "path\\to" in str(chosen_output) or "path/to" in str(chosen_output):
        print("Please provide an output folder via --output-dir or set OUTPUT_DIR at the top of the script.")
        return
    if not in_dir.exists():
        print(f"Input directory does not exist: {in_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Config echo
    print("Configuration:")
    print(f"  INPUT_DIR               = {in_dir}")
    print(f"  OUTPUT_DIR              = {out_dir}")
    print(f"  RECURSIVE               = {RECURSIVE}")
    print(f"  CHUNKSIZE               = {CHUNKSIZE}")
    print(f"  NA_REP                  = {repr(NA_REP)}")
    print(f"  ENCODING                = {ENCODING}")
    print(f"  APPLY_VALUE_LABELS      = {APPLY_VALUE_LABELS}")
    print(f"  SAS_CATALOG_FILE        = {SAS_CATALOG_FILE}")
    print(f"  SAS_CATALOG_DIR         = {SAS_CATALOG_DIR}")
    print(f"  SAS_FORMATS_AS_CATEGORY = {SAS_FORMATS_AS_CATEGORY}")
    print(f"  SAS_FORMATS_AS_ORDERED  = {SAS_FORMATS_AS_ORDERED}")
    print(f"  WRITE_ERROR_CSV         = {WRITE_ERROR_CSV}")
    print(f"  ERROR_CSV_FILENAME      = {ERROR_CSV_FILENAME}")
    print()

    sas_files = find_sas_files(in_dir, RECURSIVE)
    if not sas_files:
        print("No .sas7bdat files found.")
        return

    print(f"Found {len(sas_files)} SAS dataset(s). Starting conversion...\n")

    summary_rows: List[Dict[str, Any]] = []
    value_label_rows: List[Dict[str, Any]] = []
    combined_entries: List[Dict[str, Any]] = []

    # Error collection (optional)
    error_rows: Optional[List[Dict[str, Any]]] = [] if WRITE_ERROR_CSV else None

    for f in sas_files:
        rel = f.relative_to(in_dir) if RECURSIVE else Path(f.name)
        rel_no_ext = rel.with_suffix("")  # remove .sas7bdat
        dataset_name = rel_no_ext.name
        rel_str = str(rel.parent).replace("\\", "/")

        out_csv = out_dir / rel.parent / f"{rel_no_ext.name}.csv"
        out_meta = out_dir / rel.parent / f"{rel_no_ext.name}__metadata.json"

        print(f"Processing: {rel}")

        try:
            meta_dict, df_sample = convert_one_sas(
                in_path=f,
                out_csv_path=out_csv,
                out_meta_path=out_meta,
                apply_value_labels=APPLY_VALUE_LABELS,
                na_rep=NA_REP,
                chunksize=CHUNKSIZE,
                encoding=ENCODING,
            )
        except Exception as ex:
            # Always print the error
            print(f"  ERROR converting {rel}: {ex}")

            # Optionally collect for the error CSV
            if error_rows is not None:
                try:
                    size_bytes = f.stat().st_size
                except Exception:
                    size_bytes = None

                error_rows.append({
                    "dataset": dataset_name,
                    "relative_path": rel_str,
                    "source_file": str(rel).replace("\\", "/"),
                    "error_type": type(ex).__name__,
                    "error_message": str(ex),
                    "file_size_bytes": size_bytes,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                })
            continue

        # Success: aggregate
        append_metadata_summary(summary_rows, dataset_name, rel_str, meta_dict, df_sample)
        value_label_rows.extend(flatten_value_labels(dataset_name, rel_str, meta_dict))

        combined_entries.append({
            "dataset_name": dataset_name,
            "relative_path": rel_str,
            "source_file": str(rel).replace("\\", "/"),
            "metadata": meta_dict,
        })

    # Aggregated CSVs
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.sort_values(["relative_path", "dataset", "variable"], inplace=True)
        df_summary.to_csv(out_dir / "metadata_summary.csv", index=False)

    if value_label_rows:
        df_vl = pd.DataFrame(value_label_rows)
        df_vl.sort_values(["relative_path", "dataset", "variable", "code"], inplace=True)
        df_vl.to_csv(out_dir / "value_labels.csv", index=False)

    # Combined JSON codebook
    combined_codebook = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(in_dir),
        "output_root": str(out_dir),
        "recursive": bool(RECURSIVE),
        "apply_value_labels": bool(APPLY_VALUE_LABELS),
        "encoding": ENCODING,
        "datasets": combined_entries,
    }
    write_json(out_dir / "combined_codebook.json", combined_codebook)

    # Error report (optional)
    if WRITE_ERROR_CSV and error_rows:
        df_err = pd.DataFrame(error_rows)
        df_err.sort_values(["relative_path", "dataset"], inplace=True)
        err_path = out_dir / ERROR_CSV_FILENAME
        df_err.to_csv(err_path, index=False)
        print(f"\nCompleted with {len(error_rows)} error(s).")
        print(f"- Error report: {err_path}")

    print("\nDone.")
    print(f"- CSVs + per-dataset JSON metadata in: {out_dir}")
    if summary_rows:
        print(f"- Aggregated per-variable metadata: {out_dir/'metadata_summary.csv'}")
    if value_label_rows:
        print(f"- Flattened value labels: {out_dir/'value_labels.csv'}")
    print(f"- Combined codebook: {out_dir/'combined_codebook.json'}")


if __name__ == "__main__":
    main()
