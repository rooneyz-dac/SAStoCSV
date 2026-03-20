# language: python
#!/usr/bin/env python3
import argparse
from functools import reduce
from typing import List, Optional, Any, Tuple, Dict
import pandas as pd

# Default CSV paths (set your file paths here)
FILE_A = r"C:\path\to\file_a.csv"
FILE_B = r"C:\path\to\file_b.csv"
FILE_C = r"C:\path\to\file_c.csv"

def read_csv_with_index(path: str, key_cols: Optional[List[str]], label: str, sep: str):
    # let pandas infer dtypes so we can compare types exactly
    df = pd.read_csv(path, sep=sep, keep_default_na=True)
    if key_cols is None:
        df = df.reset_index(drop=False).rename(columns={"index": "__row_index__"})
        key_cols = ["__row_index__"]
    # add presence marker
    df[f"__present__{label}"] = True
    # rename non-key columns to keep file-sourced names
    rename_map = {}
    for c in df.columns:
        if c not in key_cols and not c.startswith("__present__"):
            rename_map[c] = f"{c}__{label}"
    df = df.rename(columns=rename_map)
    return df, key_cols

def is_missing(x: Any) -> bool:
    return pd.isna(x)

def value_type_name(x: Any) -> str:
    if is_missing(x):
        return "missing"
    return type(x).__name__

def pairwise_info(a: Any, b: Any) -> Tuple[bool, bool]:
    # returns (value_equal, type_equal)
    if is_missing(a) and is_missing(b):
        return True, True
    if is_missing(a) or is_missing(b):
        return False, False
    return (a == b), (type(a) == type(b))

def compare_values(a: Any, b: Any, c: Any) -> Tuple[bool, Dict]:
    # Exact equality required: values equal AND types equal for comparisons.
    vals = {"A": a, "B": b, "C": c}
    types = {k: value_type_name(v) for k, v in vals.items()}

    # If all three missing, consider equal
    if is_missing(a) and is_missing(b) and is_missing(c):
        return False, {"reason": "all_missing", "values": vals, "types": types}

    # Compare pairwise for value and type
    pairs = [
        ("A", "B",) ,
        ("A", "C",) ,
        ("B", "C",)
    ]
    pair_results = []
    any_unequal = False
    for x, y in pairs:
        veq, teq = pairwise_info(vals[x], vals[y])
        pair_results.append({"pair": f"{x}-{y}", "value_equal": bool(veq), "type_equal": bool(teq),
                             "x_value": vals[x], "y_value": vals[y], "x_type": types[x], "y_type": types[y]})
        if not (veq and teq):
            any_unequal = True

    return any_unequal, {"type": "exact", "pairwise": pair_results, "values": vals, "types": types}

def compare_three_csv(file_a: str, file_b: str, file_c: str,
                      key: Optional[List[str]] = None,
                      sep: str = ",",
                      output_report: Optional[str] = None):
    df_a, key_cols = read_csv_with_index(file_a, key, "A", sep)
    df_b, _ = read_csv_with_index(file_b, key, "B", sep)
    df_c, _ = read_csv_with_index(file_c, key, "C", sep)

    merged = reduce(lambda left, right: pd.merge(left, right, on=key_cols, how="outer"), [df_a, df_b, df_c])

    # presence flags robustly
    for label in ("A", "B", "C"):
        col = f"__present__{label}"
        if col in merged.columns:
            merged[f"__in_{label}__"] = merged[col].fillna(False).astype(bool)
        else:
            merged[f"__in_{label}__"] = False

    # determine base columns (original non-key names)
    base_cols = set()
    for col in merged.columns:
        if col in key_cols or col.startswith("__present__") or col.startswith("__in_"):
            continue
        # expected renamed form: original__A / original__B / original__C
        if col.endswith("__A") or col.endswith("__B") or col.endswith("__C"):
            base_cols.add(col.rsplit("__", 1)[0])

    diffs = []
    for _, row in merged.iterrows():
        present_flags = (bool(row["__in_A__"]), bool(row["__in_B__"]), bool(row["__in_C__"]))
        # row only in one file
        if sum(present_flags) <= 1:
            diffs.append({
                "key": {k: row[k] for k in key_cols},
                "issue": "row_only_in_one_file",
                "in_A": present_flags[0],
                "in_B": present_flags[1],
                "in_C": present_flags[2],
                "column": None,
                "values": None
            })
            continue

        for col in sorted(base_cols):
            a = row.get(f"{col}__A", None)
            b = row.get(f"{col}__B", None)
            c = row.get(f"{col}__C", None)
            unequal, info = compare_values(a, b, c)
            if unequal:
                diffs.append({
                    "key": {k: row[k] for k in key_cols},
                    "issue": "column_mismatch",
                    "column": col,
                    "values": {"A": a, "B": b, "C": c},
                    "detail": info
                })

    # Summary
    total_rows = len(merged)
    rows_only_a = merged[merged["__in_A__"] & ~merged["__in_B__"] & ~merged["__in_C__"]].shape[0]
    rows_only_b = merged[~merged["__in_A__"] & merged["__in_B__"] & ~merged["__in_C__"]].shape[0]
    rows_only_c = merged[~merged["__in_A__"] & ~merged["__in_B__"] & merged["__in_C__"]].shape[0]
    rows_in_all = merged[merged["__in_A__"] & merged["__in_B__"] & merged["__in_C__"]].shape[0]
    total_diff_cells = sum(1 for d in diffs if d["issue"] == "column_mismatch")
    total_row_only = sum(1 for d in diffs if d["issue"] == "row_only_in_one_file")

    print("=== Comparison summary ===")
    print(f"Files: {file_a} (A), {file_b} (B), {file_c} (C)")
    print(f"Key columns: {key_cols}")
    print(f"Total merged rows: {total_rows}")
    print(f"Rows present in all three: {rows_in_all}")
    print(f"Rows only in A: {rows_only_a}, only in B: {rows_only_b}, only in C: {rows_only_c}")
    print(f"Columns with mismatched values (cell-level diffs): {total_diff_cells}")
    print(f"Rows present only in a single file: {total_row_only}")
    print("")

    if not diffs:
        print("No differences found.")
        return

    print("=== Detailed differences ===")
    for entry in diffs:
        key_repr = ", ".join(f"{k}={v}" for k, v in entry["key"].items())
        if entry["issue"] == "row_only_in_one_file":
            print(f"[ROW ONLY] {key_repr} -- in A={entry['in_A']}, B={entry['in_B']}, C={entry['in_C']}")
        else:
            col = entry["column"]
            vals = entry["values"]
            detail = entry.get("detail", {})
            print(f"[COL MISMATCH] {key_repr} :: {col} -> A={vals['A']} ({detail['types']['A']}), "
                  f"B={vals['B']} ({detail['types']['B']}), C={vals['C']} ({detail['types']['C']})")
            for p in detail.get("pairwise", []):
                print(f"  pair {p['pair']}: value_equal={p['value_equal']}, type_equal={p['type_equal']}")

    if output_report:
        out_rows = []
        for e in diffs:
            row = {}
            for k, v in e["key"].items():
                row[f"key_{k}"] = v
            row["issue"] = e["issue"]
            row["column"] = e.get("column")
            vals = e.get("values")
            if vals:
                row["A_value"] = vals.get("A")
                row["B_value"] = vals.get("B")
                row["C_value"] = vals.get("C")
            row["detail"] = str(e.get("detail"))
            out_rows.append(row)
        pd.DataFrame(out_rows).to_csv(output_report, index=False)
        print(f"\nDetailed report written to {output_report}")

def parse_key(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts if parts else None

def main():
    p = argparse.ArgumentParser(description="Compare three CSV files with exact value+type equality.")
    p.add_argument("file_a", nargs='?', help="Path to CSV A (will be labeled A). If omitted uses FILE_A.", default=FILE_A)
    p.add_argument("file_b", nargs='?', help="Path to CSV B (will be labeled B). If omitted uses FILE_B.", default=FILE_B)
    p.add_argument("file_c", nargs='?', help="Path to CSV C (will be labeled C). If omitted uses FILE_C.", default=FILE_C)
    p.add_argument("--key", help="Comma-separated key column(s). If omitted, rows are compared by row index.", default=None)
    p.add_argument("--sep", help="CSV delimiter (default ',').", default=",")
    p.add_argument("--output", help="Write detailed difference CSV to this path.", default=None)
    args = p.parse_args()

    key_cols = parse_key(args.key)
    compare_three_csv(args.file_a, args.file_b, args.file_c, key=key_cols, sep=args.sep, output_report=args.output)

if __name__ == "__main__":
    main()
