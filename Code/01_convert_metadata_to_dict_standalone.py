#!/usr/bin/env python3
"""Standalone metadata-to-dictionary converter.

Default behavior (no arguments):
    - Reads ./meta_data_summary.xlsx
    - Writes ./dict/dictionary.csv and ./dict/dictionary.xlsx

You can optionally pass a different input file as the first argument.
"""

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_FILE = "meta_data_summary.xlsx"
DEFAULT_OUTPUT_DIR = "dict"
DEFAULT_OUTPUT_BASENAME = "dictionary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert metadata Excel sheets into combined dictionary CSV/XLSX outputs."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=DEFAULT_INPUT_FILE,
        help=f"Input metadata Excel file (default: {DEFAULT_INPUT_FILE})",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for generated files (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def build_dictionary(input_file: Path) -> pd.DataFrame:
    excel_data = pd.ExcelFile(input_file)
    combined_df = pd.DataFrame()

    for sheet in excel_data.sheet_names:
        df = pd.read_excel(input_file, sheet_name=sheet, header=2)
        df.columns = df.columns.astype(str).str.upper()
        df["SOURCE_TAB"] = sheet
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    if combined_df.empty:
        raise ValueError("No data was extracted from the Excel workbook.")

    return combined_df


def main() -> int:
    args = parse_args()

    input_path = Path(args.input_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.exists() or not input_path.is_file():
        print(f"Error: Input file not found: {input_path}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        combined_df = build_dictionary(input_path)
    except Exception as exc:
        print(f"Error while processing metadata file: {exc}")
        return 1

    csv_output = output_dir / f"{DEFAULT_OUTPUT_BASENAME}.csv"
    xlsx_output = output_dir / f"{DEFAULT_OUTPUT_BASENAME}.xlsx"

    try:
        combined_df.to_csv(csv_output, index=False)
        combined_df.to_excel(xlsx_output, index=False)
    except Exception as exc:
        print(f"Error while writing output files: {exc}")
        return 1

    print(f"CSV written: {csv_output}")
    print(f"XLSX written: {xlsx_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
