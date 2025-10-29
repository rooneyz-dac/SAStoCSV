# python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert a folder of .sas7bdat files to CSV while capturing SAS metadata.

This variant can use SAS (via saspy) to perform the CSV export instead of
using pyreadstat/pandas to write CSVs. Toggle with USE_SASPY.

Edit the CONFIGURATION block below.
"""

# ==== CONFIGURATION (EDIT THESE) ==============================================
INPUT_DIR = r"C:\Users\rooneyz\Documents\TestData\MirumTestData"
OUTPUT_DIR = r"output"

RECURSIVE = True

NA_REP = ""
ENCODING = None

APPLY_VALUE_LABELS = False
SAS_CATALOG_FILE = None
SAS_CATALOG_DIR  = None
SAS_FORMATS_AS_CATEGORY = False
SAS_FORMATS_AS_ORDERED = False

USE_SASPY = True

WRITE_METADATA_JSON = True

WRITE_ERROR_CSV = True
ERROR_CSV_FILENAME = "00_conversion_errors.csv"

# New: batch configuration
BATCH_SIZE_FILES = 8       # number of files to process per batch (reduces concurrent resource usage)
CHUNK_ROWS = 100000        # if >0, attempt to read large SAS files in row chunks when not using SASPY
# Set CHUNK_ROWS = None or 0 to disable row-chunking
# ==============================================================================

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone
import traceback
import math

import pandas as pd
import argparse

_saspy_available = False
_saspy_import_error: Optional[str] = None
if USE_SASPY:
    try:
        import saspy  # type: ignore
        _saspy_available = True
    except Exception as e:
        _saspy_available = False
        _saspy_import_error = str(e)

try:
    import pyreadstat
except ImportError as e:
    raise SystemExit("Missing dependency: pyreadstat. Install with: pip install pyreadstat") from e


def safe_serialize(obj):
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

    column_names = getattr(meta, "column_names", None) or []
    column_labels = getattr(meta, "column_labels", None) or []
    column_formats = getattr(meta, "column_formats", None) or []
    variable_types = getattr(meta, "variable_types", None)
    column_display_width = getattr(meta, "column_display_width", None) or []

    value_labels_by_var: Dict[str, Dict[Any, str]] = {}
    vv = getattr(meta, "variable_value_labels", None)
    if isinstance(vv, dict):
        # some pyreadstat versions provide per-variable dicts mapping codes to labels
        for var, mapping in vv.items():
            if isinstance(mapping, dict):
                value_labels_by_var[var] = mapping

    if not value_labels_by_var:
        # fallback using variable_to_label + value_labels mapping
        var_to_labelset = getattr(meta, "variable_to_label", None)
        labelsets = getattr(meta, "value_labels", None)
        if isinstance(var_to_labelset, dict) and isinstance(labelsets, dict):
            for var, labelset_name in var_to_labelset.items():
                ls = labelsets.get(labelset_name)
                if isinstance(ls, dict):
                    value_labels_by_var[var] = ls

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


def find_catalog_for_dataset(dataset_path: Path,
                             catalog_file_env: Optional[str],
                             catalog_dir_env: Optional[str]) -> Optional[Path]:
    if catalog_file_env:
        p = Path(catalog_file_env).expanduser()
        if p.exists():
            return p
    sibling = dataset_path.with_suffix(".sas7bcat")
    if sibling.exists():
        return sibling
    if catalog_dir_env:
        cd = Path(catalog_dir_env).expanduser()
        if cd.exists():
            exact = cd / f"{dataset_path.stem}.sas7bcat"
            if exact.exists():
                return exact
            for p in cd.glob("*.sas7bcat"):
                return p
    return None


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
                    "label": label,
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
        has_vl = isinstance(var.get("value_labels"), dict) and len(var.get("value_labels") or {}) > 0
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


def _ensure_sas_session(cfgname: Optional[str] = None) -> Tuple[Optional["saspy.SASsession"], Optional[str]]:
    if not _saspy_available:
        return None, _saspy_import_error
    try:
        if cfgname:
            sas = saspy.SASsession(cfgname=cfgname)
        else:
            sas = saspy.SASsession()
        return sas, None
    except Exception as e:
        err = str(e)
        # If on Windows and STDIO transport was attempted, try IOM fallback if configured
        if os.name == 'nt' and 'STDIO' in err.upper():
            try:
                # fallback config name expected in sascfg_personal.py (adjust if you use a different name)
                sas = saspy.SASsession(cfgname='iomwin')
                return sas, None
            except Exception as e2:
                return None, f"{err}  (IOM fallback failed: {e2})"
        return None, err


def _sas_export_csv(sas: "saspy.SASsession", in_path: Path, out_csv_path: Path, dataset_stem: str) -> None:
    lib_path = str(in_path.parent).replace("\\", "/")
    libref = "inlib"
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_csv_unix = str(out_csv_path).replace("\\", "/")
    sas_code = f"""
libname {libref} "{lib_path}";
proc export data={libref}."{dataset_stem}"n
    outfile="{out_csv_unix}"
    dbms=csv
    replace;
    putnames=yes;
run;
libname {libref} clear;
"""
    sas.submit(sas_code)


def convert_one_sas(
    in_path: Path,
    out_csv_path: Path,
    out_meta_path: Path,
    apply_value_labels: bool = False,
    na_rep: str = "",
    encoding: Optional[str] = None,
    write_metadata: bool = True,
    sas_session_override: Optional["saspy.SASsession"] = None,
    chunk_rows: Optional[int] = None,
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    """
    Convert a single SAS dataset to CSV + metadata.
    If `sas_session_override` is provided it will be used and not closed here.
    If `chunk_rows` is set (>0), attempt to stream the file in chunks via pyreadstat to avoid full memory load.
    """
    ensure_parent_dir(out_csv_path)
    if write_metadata:
        ensure_parent_dir(out_meta_path)

    df_sample: Optional[pd.DataFrame] = None
    dataset_stem = in_path.with_suffix("").name

    used_sas = False
    sas_reason: Optional[str] = None
    sas_session = None

    # Try to use SAS first (reusing provided session when available)
    if not USE_SASPY:
        sas_reason = "USE_SASPY disabled by configuration"
    elif not _saspy_available:
        sas_reason = f"saspy import failed: {_saspy_import_error}"
    else:
        # Use override session if provided (batch-level reuse), else create per-file
        if sas_session_override is not None:
            sas_session = sas_session_override
        else:
            sas_session, sess_err = _ensure_sas_session()
            if sas_session is None:
                sas_reason = f"SAS session creation failed: {sess_err}"

        if sas_session is not None and sas_session_override is not None:
            # reuse provided session
            try:
                print(f"  attempting SASPY export (reused session): {in_path} -> {out_csv_path}")
                _sas_export_csv(sas_session, in_path, out_csv_path, dataset_stem)
                used_sas = True
            except Exception as e:
                sas_reason = f"SAS export error (reused session): {e}"
        elif sas_session is not None and sas_session_override is None:
            try:
                print(f"  attempting SASPY export: {in_path} -> {out_csv_path}")
                _sas_export_csv(sas_session, in_path, out_csv_path, dataset_stem)
                used_sas = True
            except Exception as e:
                sas_reason = f"SAS export error: {e}"
            finally:
                try:
                    sas_session.endsas()
                except Exception:
                    pass

    # Read metadata (always via pyreadstat) and write CSV via chosen path
    if used_sas:
        # SAS wrote the CSV; still read metadata via pyreadstat
        try:
            df, meta = pyreadstat.read_sas7bdat(str(in_path), dates_as_pandas_datetime=True, user_missing=True)
            df_sample = df.head(50).copy()
            meta_dict = extract_meta_dict(meta)
            if write_metadata:
                write_json(out_meta_path, meta_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to read metadata after SAS export: {e}") from e
    else:
        # fallback: read (possibly in chunks) and write CSV incrementally
        print(f"  exporting with pyreadstat (fallback): {in_path} -> {out_csv_path}")
        read_kwargs = {
            "dates_as_pandas_datetime": True,
            "user_missing": True,
        }
        if encoding:
            read_kwargs["encoding"] = encoding
        if apply_value_labels:
            catalog_path = find_catalog_for_dataset(in_path, SAS_CATALOG_FILE, SAS_CATALOG_DIR)
            if catalog_path:
                read_kwargs["formats_catalog_path"] = str(catalog_path)
                read_kwargs["apply_value_formats"] = True
                read_kwargs["formats_as_category"] = SAS_FORMATS_AS_CATEGORY
                read_kwargs["formats_as_ordered"] = SAS_FORMATS_AS_ORDERED

        # Try chunked reading using pyreadstat.read_file_in_chunks if configured and available
        if chunk_rows and chunk_rows > 0 and hasattr(pyreadstat, "read_file_in_chunks"):
            # pyreadstat.read_file_in_chunks returns an iterator of (df, meta) tuples for many formats
            first_chunk = True
            meta_dict = None
            try:
                it = pyreadstat.read_file_in_chunks(str(in_path), chunksize=chunk_rows, file_format="sas7bdat", **read_kwargs)
                for chunk_df, chunk_meta in it:
                    if first_chunk:
                        df_sample = chunk_df.head(50).copy()
                        meta_dict = extract_meta_dict(chunk_meta)
                        if write_metadata and meta_dict is not None:
                            write_json(out_meta_path, meta_dict)
                    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
                    chunk_df.to_csv(out_csv_path, index=False, na_rep=na_rep, mode="a", header=first_chunk)
                    first_chunk = False
                if meta_dict is None:
                    raise RuntimeError("No data read in chunks; unable to obtain metadata.")
            except Exception as e:
                # fallback to full read if chunked reading fails
                print(f"    chunked read failed: {e}; falling back to full read")
                df, meta = pyreadstat.read_sas7bdat(str(in_path), **read_kwargs)
                df_sample = df.head(50).copy()
                out_csv_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(out_csv_path, index=False, na_rep=na_rep)
                meta_dict = extract_meta_dict(meta)
                if write_metadata:
                    write_json(out_meta_path, meta_dict)
        else:
            # full read
            df, meta = pyreadstat.read_sas7bdat(str(in_path), **read_kwargs)
            df_sample = df.head(50).copy()
            out_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv_path, index=False, na_rep=na_rep)
            meta_dict = extract_meta_dict(meta)
            if write_metadata:
                write_json(out_meta_path, meta_dict)

    # Print final SASPY usage status
    if used_sas:
        print(f"  SASPY used: True")
    else:
        print(f"  SASPY used: False; reason: {sas_reason}")

    return meta_dict, df_sample


def find_sas_files(root: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.sas7bdat" if recursive else "*.sas7bdat"
    return [p for p in root.glob(pattern) if p.is_file()]


def _chunks(lst: List[Path], size: int):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]


def main():
    parser = argparse.ArgumentParser(description="Convert folder of .sas7bdat files to CSV with metadata.")
    parser.add_argument("--input-dir", "-i", dest="input_dir", help="Path to input folder containing .sas7bdat files. Overrides top-level INPUT_DIR.")
    parser.add_argument("--output-dir", "-o", dest="output_dir", help="Path to output folder. Overrides top-level OUTPUT_DIR.")
    parser.add_argument("--dry-run", action="store_true", help="List files but don't convert")
    args = parser.parse_args()

    chosen_input = args.input_dir or INPUT_DIR
    chosen_output = args.output_dir or OUTPUT_DIR

    in_dir = Path(chosen_input).expanduser().resolve()
    out_dir = Path(chosen_output).expanduser().resolve()

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

    csv_root = out_dir / "DAC_CSV"
    meta_root = out_dir / "DAC_JSON_Metadata"
    csv_root.mkdir(parents=True, exist_ok=True)
    if WRITE_METADATA_JSON:
        meta_root.mkdir(parents=True, exist_ok=True)

    print("Configuration:")
    print(f"  INPUT_DIR               = {in_dir}")
    print(f"  OUTPUT_DIR              = {out_dir}")
    print(f"  CSV_ROOT                = {csv_root}")
    print(f"  META_ROOT               = {meta_root}  (created: {WRITE_METADATA_JSON})")
    print(f"  RECURSIVE               = {RECURSIVE}")
    print(f"  NA_REP                  = {repr(NA_REP)}")
    print(f"  ENCODING                = {ENCODING}")
    print(f"  APPLY_VALUE_LABELS      = {APPLY_VALUE_LABELS}")
    print(f"  SAS_CATALOG_FILE        = {SAS_CATALOG_FILE}")
    print(f"  SAS_CATALOG_DIR         = {SAS_CATALOG_DIR}")
    print(f"  SAS_FORMATS_AS_CATEGORY = {SAS_FORMATS_AS_CATEGORY}")
    print(f"  SAS_FORMATS_AS_ORDERED  = {SAS_FORMATS_AS_ORDERED}")
    print(f"  USE_SASPY               = {USE_SASPY} (available: {_saspy_available})")
    if _saspy_import_error:
        print(f"  SASPY_IMPORT_ERROR      = {_saspy_import_error}")
    print(f"  WRITE_METADATA_JSON     = {WRITE_METADATA_JSON}")
    print(f"  WRITE_ERROR_CSV         = {WRITE_ERROR_CSV}")
    print(f"  ERROR_CSV_FILENAME      = {ERROR_CSV_FILENAME}")
    print(f"  BATCH_SIZE_FILES        = {BATCH_SIZE_FILES}")
    print(f"  CHUNK_ROWS              = {CHUNK_ROWS}")
    print()

    sas_files = find_sas_files(in_dir, RECURSIVE)
    if not sas_files:
        print("No .sas7bdat files found.")
        return

    print(f"Found {len(sas_files)} SAS dataset(s). Starting conversion...\n")

    summary_rows: List[Dict[str, Any]] = []
    value_label_rows: List[Dict[str, Any]] = []
    combined_entries: List[Dict[str, Any]] = []
    error_rows: Optional[List[Dict[str, Any]]] = [] if WRITE_ERROR_CSV else None

    # Process files in batches to limit resource usage and optionally reuse SAS sessions
    for batch_idx, batch in enumerate(_chunks(sas_files, BATCH_SIZE_FILES), start=1):
        print(f"Processing batch {batch_idx}: {len(batch)} file(s)")
        # If SASpy is desired and available, create a session to reuse for the whole batch (reduces overhead)
        batch_sas_session = None
        if USE_SASPY and _saspy_available:
            try:
                batch_sas_session, sess_err = _ensure_sas_session()
                if batch_sas_session is None:
                    print(f"  Warning: couldn't create SAS session for batch: {sess_err}")
                else:
                    print("  SAS session established for batch reuse.")
            except Exception as e:
                print(f"  Warning: SAS session creation failed for batch: {e}")
                batch_sas_session = None

        for f in batch:
            rel = f.relative_to(in_dir) if RECURSIVE else Path(f.name)
            rel_no_ext = rel.with_suffix("")
            dataset_name = rel_no_ext.name
            rel_str = str(rel.parent).replace("\\", "/")

            out_csv = csv_root / rel.parent / f"{rel_no_ext.name}.csv"
            out_meta = meta_root / rel.parent / f"{rel_no_ext.name}__metadata.json"

            print(f"Processing: {rel}")

            try:
                # If dry-run, just print and skip
                if args.dry_run:
                    print(f"  dry-run: would convert {f}")
                    continue

                meta_dict, df_sample = convert_one_sas(
                    in_path=f,
                    out_csv_path=out_csv,
                    out_meta_path=out_meta,
                    apply_value_labels=APPLY_VALUE_LABELS,
                    na_rep=NA_REP,
                    encoding=ENCODING,
                    write_metadata=WRITE_METADATA_JSON,
                    sas_session_override=batch_sas_session,
                    chunk_rows=CHUNK_ROWS,
                )
            except Exception as ex:
                print(f"  ERROR converting {rel}: {ex}")
                if error_rows is not None:
                    try:
                        tb_list = traceback.extract_tb(ex.__traceback__) if ex.__traceback__ is not None else []
                        err_file = tb_list[-1].filename if tb_list else None
                        err_line = tb_list[-1].lineno if tb_list else None
                        err_func = tb_list[-1].name if tb_list else None
                    except Exception:
                        err_file = err_line = err_func = None
                    size_bytes = None
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
                        "error_file": err_file,
                        "error_line": err_line,
                        "error_func": err_func,
                        "file_size_bytes": size_bytes,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    })
                continue

            append_metadata_summary(summary_rows, dataset_name, rel_str, meta_dict, df_sample)
            value_label_rows.extend(flatten_value_labels(dataset_name, rel_str, meta_dict))

            combined_entries.append({
                "dataset_name": dataset_name,
                "relative_path": rel_str,
                "source_file": str(rel).replace("\\", "/"),
                "metadata": meta_dict,
            })

        # end batch: close batch SAS session if we opened one
        if batch_sas_session is not None:
            try:
                batch_sas_session.endsas()
                print("  Closed batch SAS session.")
            except Exception:
                pass

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.sort_values(["relative_path", "dataset", "variable"], inplace=True)
        df_summary.to_csv(out_dir / "metadata_summary.csv", index=False)

    if value_label_rows:
        df_vl = pd.DataFrame(value_label_rows)
        df_vl.sort_values(["relative_path", "dataset", "variable", "code"], inplace=True)
        df_vl.to_csv(out_dir / "value_labels.csv", index=False)

    combined_codebook = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(in_dir),
        "output_root": str(out_dir),
        "csv_root": str(csv_root),
        "meta_root": str(meta_root),
        "recursive": bool(RECURSIVE),
        "apply_value_labels": bool(APPLY_VALUE_LABELS),
        "encoding": ENCODING,
        "datasets": combined_entries,
    }
    write_json(out_dir / "combined_codebook.json", combined_codebook)

    if WRITE_ERROR_CSV:
        error_columns = [
            "dataset",
            "relative_path",
            "source_file",
            "error_type",
            "error_message",
            "error_file",
            "error_line",
            "error_func",
            "file_size_bytes",
            "timestamp_utc",
        ]
        df_err = pd.DataFrame(error_rows or [], columns=error_columns)
        if not df_err.empty:
            df_err.sort_values(["relative_path", "dataset"], inplace=True)
        err_path = out_dir / ERROR_CSV_FILENAME
        df_err.to_csv(err_path, index=False)
        n_err = len(df_err)
        print(f"\nCompleted with {n_err} error(s).")
        print(f"- Error report: {err_path}")

    print("\nDone.")
    print(f"- CSVs in: {csv_root}")
    if WRITE_METADATA_JSON:
        print(f"- Per-dataset JSON metadata in: {meta_root}")
    if summary_rows:
        print(f"- Aggregated per-variable metadata: {out_dir/'metadata_summary.csv'}")
    if value_label_rows:
        print(f"- Flattened value labels: {out_dir/'value_labels.csv'}")
    print(f"- Combined codebook: {out_dir/'combined_codebook.json'}")


if __name__ == "__main__":
    main()
