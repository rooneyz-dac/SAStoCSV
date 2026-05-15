#!/usr/bin/env python3
"""Standalone metadata-to-dictionary converter.

Usage:
    python 01_convert_metadata_to_dict_standalone.py [-i INPUT_FILE] [-o OUTPUT_DIR]

If -i is not supplied the script prompts for the input file path interactively.
Outputs are always written as both CSV and XLSX to the output directory.

Examples:
    # interactive prompt for input
    python 01_convert_metadata_to_dict_standalone.py

    # explicit input file via -i flag
    python 01_convert_metadata_to_dict_standalone.py -i /path/to/meta_data_summary.xlsx

    # custom output directory
    python 01_convert_metadata_to_dict_standalone.py -i /path/to/meta_data_summary.xlsx -o /path/to/out
"""

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT_DIR = "dict"
DEFAULT_OUTPUT_BASENAME = "dictionary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a multi-sheet metadata Excel workbook into combined\n"
            "dictionary outputs (CSV and XLSX).\n\n"
            "Every sheet in the workbook is read (header on row 3), column\n"
            "names are normalised to uppercase, a SOURCE_TAB column is added,\n"
            "and all sheets are concatenated into a single output file pair:\n"
            "  <output-dir>/dictionary.csv\n"
            "  <output-dir>/dictionary.xlsx"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # no arguments — script prompts you for the input file path\n"
            "  python 01_convert_metadata_to_dict_standalone.py\n\n"
            "  # provide the input file via -i\n"
            "  python 01_convert_metadata_to_dict_standalone.py -i meta_data_summary.xlsx\n\n"
            "  # custom output directory\n"
            "  python 01_convert_metadata_to_dict_standalone.py "
            "-i meta_data_summary.xlsx -o /path/to/output\n\n"
            "  # show this help message\n"
            "  python 01_convert_metadata_to_dict_standalone.py -h"
        ),
    )
    parser.add_argument(
        "-i",
        "--input-file",
        metavar="INPUT_FILE",
        default=None,
        help=(
            "Path to the input metadata Excel workbook (.xlsx). "
            "If omitted, the script will prompt you to enter the path interactively."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        metavar="OUTPUT_DIR",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            f"Directory where dictionary.csv and dictionary.xlsx are written. "
            f"Created automatically if it does not exist. (default: {DEFAULT_OUTPUT_DIR})"
        ),
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

    # Resolve input file: use -i value if supplied, otherwise prompt the user
    if args.input_file:
        input_path = Path(args.input_file).expanduser().resolve()
    else:
        raw = input("Please enter the path to the metadata Excel file: ").strip()
        input_path = Path(raw).expanduser().resolve()

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
