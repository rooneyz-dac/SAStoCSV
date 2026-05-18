#!/usr/bin/env python3
"""
metadata_to_dict_cli20260320.py

Purpose:
    Build a combined trial dictionary from a metadata summary Excel workbook.

Usage:
    python metadata_to_dict_cli20260320.py <input_excel> <output_dir> <trial_name> [input_dir]

Positional arguments:
    input_excel     Path to the metadata Excel workbook (multiple sheets expected).
    output_dir      Base directory where `DAC_Documents` will be created/used.
    trial_name      Trial short name (will be uppercased) used for output filenames.
    input_dir       (Optional) Path to the input data directory; used to derive the
                    GGG/GG/G filename components matching the SAS naming convention.
                    If omitted, output_dir is used as a fallback.

Behavior:
    - Reads every sheet in the workbook using header row 3 (zero-based index 2).
    - Normalizes column names to uppercase, replaces any whitespace runs (including newlines) with a single space, and strips.
    - Concatenates all sheets and writes two output files to DAC_Documents:
        `DAC_Documents/dictionary_{GGG_PARENT}_{GG_PARENT}_{G_PARENT}_{YYYYMMDD}.csv`
        `DAC_Documents/dictionary_{GGG_PARENT}_{GG_PARENT}_{G_PARENT}_{YYYYMMDD}.xlsx`
    where GGG_PARENT, GG_PARENT, and G_PARENT are the third-to-last, second-to-last,
    and last segments of the input directory path (alphanumeric characters only,
    matching the SAS compress(...,,ka) convention — e.g. "C:" becomes "C").
    This matches the naming convention used by variable_info_cli20260320.sas.

Exit codes:
    0   Success
    1   Incorrect usage / missing args / I/O error

Requirements:
    - Python 3.8+
    - pandas
    - openpyxl (for Excel I/O)

Notes:
    - Intended for CLI use on Windows; paths may be absolute or relative.
    - Header row is configurable in code if format changes.

ChangeLog:
    2026-01-05  Header updated to document behavior and column normalization.
    2026-03-27  Sanitize path-part components with re.sub([^a-zA-Z0-9]) so Windows drive letters
                (e.g. "C:") do not embed invalid characters in output filenames; outputs both
                dictionary_*.csv and dictionary_*.xlsx to DAC_Documents.
    2026-03-27  Accept optional input_dir argument; derive GGG/GG/G filename components from
                input_dir (matching SAS variable_info naming convention) instead of output_dir.
    2026-03-30  Switched argument parsing to argparse.
"""

import argparse
import pandas as pd
import os
import sys
import re
from datetime import date


def main():
    # Step 1: Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Build a combined trial dictionary from a metadata summary Excel workbook."
    )
    parser.add_argument("input_excel", help="Path to the metadata Excel workbook (multiple sheets expected).")
    parser.add_argument("output_dir", help="Base directory where DAC_Documents will be created/used.")
    parser.add_argument("trial_name", help="Trial short name (will be uppercased) used for output filenames.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=None,
        help="(Optional) Path to the input data directory; used to derive GGG/GG/G filename components.",
    )
    args = parser.parse_args()

    file_path = args.input_excel
    output_base_dir = args.output_dir
    trial_name = args.trial_name.upper()
    input_dir = args.input_dir

    print(f"DEBUG: Input file: {file_path}")
    print(f"DEBUG: Output directory: {output_base_dir}")
    print(f"DEBUG: Trial name: {trial_name}")
    print(f"DEBUG: Input directory: {input_dir}")
    print(f"DEBUG: Current working directory: {os.getcwd()}")

    # Step 2: Validate input file exists
    if not os.path.exists(file_path):
        print(f"ERROR: The input file does not exist: {file_path}")
        sys.exit(1)

    # Step 3: Validate output directory exists
    if not os.path.exists(output_base_dir):
        print(f"ERROR: The output directory does not exist: {output_base_dir}")
        sys.exit(1)

    # Step 4: Create DAC_Documents folder if it doesn't exist
    dac_documents_dir = os.path.join(output_base_dir, 'DAC_Documents')
    if not os.path.exists(dac_documents_dir):
        try:
            os.makedirs(dac_documents_dir)
            print(f"DEBUG: Created directory: {dac_documents_dir}")
        except Exception as e:
            print(f"ERROR: Failed to create directory {dac_documents_dir}: {e}")
            sys.exit(1)
    else:
        print(f"DEBUG: Directory already exists: {dac_documents_dir}")

    # Step 5: Load the Excel file
    try:
        excel_data = pd.ExcelFile(file_path)
        print(f"DEBUG: Successfully loaded Excel file with {len(excel_data.sheet_names)} sheets")
    except Exception as e:
        print(f"ERROR: Failed to load Excel file: {e}")
        sys.exit(1)

    # Step 6: Initialize an empty DataFrame for the combined result
    combined_df = pd.DataFrame()

    # Step 7: Loop through all sheets
    for sheet in excel_data.sheet_names:
        print(f"DEBUG: Processing sheet: {sheet}")
        try:
            # Load the sheet data, specifying that the header is in row 3 (index 2)
            df = pd.read_excel(file_path, sheet_name=sheet, header=2)

            # Normalize column names:
            # - Uppercase
            # - Replace any whitespace runs (including newlines/tabs/multiple spaces) with a single space
            # - Strip leading/trailing spaces
            df.columns = (
                df.columns
                .astype(str)
                .str.upper()
                .str.replace(r'\s+', ' ', regex=True)
                .str.strip()
            )

            # Append to the combined DataFrame
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            print(f"DEBUG: Added {len(df)} rows from sheet: {sheet}")
        except Exception as e:
            print(f"WARNING: Failed to process sheet {sheet}: {e}")
            continue

    # Step 8: Validate combined data
    if combined_df.empty:
        print("ERROR: No data was extracted from the Excel file")
        sys.exit(1)

    print(f"DEBUG: Total rows in combined data: {len(combined_df)}")

    # Step 9: Save the combined DataFrame to CSV and Excel in DAC_Documents folder
    date_stamp = date.today().strftime('%Y%m%d')
    # Derive parent directory components from the input directory (if provided) or fall
    # back to the output base directory.  Using the input directory matches the SAS
    # variable_info naming convention: variable_info_GGG_GG_G_DATE.xlsx.
    # Keep only alphanumeric characters in each part to ensure valid filenames on all
    # platforms (e.g. the Windows drive letter "C:" becomes "C", matching SAS compress(...,,ka)).
    naming_dir = input_dir if input_dir is not None else output_base_dir
    path_parts = [p for p in naming_dir.replace('\\', '/').split('/') if p]
    g_parent = re.sub(r'[^a-zA-Z0-9]', '', path_parts[-1]) if len(path_parts) >= 1 else ''
    gg_parent = re.sub(r'[^a-zA-Z0-9]', '', path_parts[-2]) if len(path_parts) >= 2 else ''
    ggg_parent = re.sub(r'[^a-zA-Z0-9]', '', path_parts[-3]) if len(path_parts) >= 3 else ''
    csv_output_path = os.path.join(dac_documents_dir, f"dictionary_{ggg_parent}_{gg_parent}_{g_parent}_{date_stamp}.csv")
    excel_output_path = os.path.join(dac_documents_dir, f"dictionary_{ggg_parent}_{gg_parent}_{g_parent}_{date_stamp}.xlsx")

    try:
        combined_df.to_csv(csv_output_path, index=False)
        print(f"SUCCESS: Combined CSV data saved to: {csv_output_path}")
    except Exception as e:
        print(f"ERROR: Failed to save CSV file: {e}")
        sys.exit(1)

    try:
        combined_df.to_excel(excel_output_path, index=False)
        print(f"SUCCESS: Combined Excel data saved to: {excel_output_path}")
    except Exception as e:
        print(f"ERROR: Failed to save Excel file: {e}")
        sys.exit(1)

    print("Process completed successfully")


if __name__ == "__main__":
    main()
