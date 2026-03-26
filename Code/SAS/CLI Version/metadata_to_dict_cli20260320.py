#!/usr/bin/env python3
"""
metadata_to_dict_cli20260320.py

Purpose:
    Build a combined trial dictionary from a metadata summary Excel workbook.

Usage:
    python metadata_to_dict_cli20260320.py <input_excel> <output_dir> <trial_name>

Positional arguments:
    input_excel     Path to the metadata Excel workbook (multiple sheets expected).
    output_dir      Base directory where `DAC_Documents` will be created/used.
    trial_name      Trial short name (will be uppercased) used for output filenames.

Behavior:
    - Reads every sheet in the workbook using header row 3 (zero-based index 2).
    - Normalizes column names to uppercase, replaces any whitespace runs (including newlines) with a single space, and strips.
    - Renames many variant column names (case-insensitive/multi-line aware) to the standard names:
        NUM      <- 'VARIABLE NUMBER', 'VAR NUM', 'VAR NO', 'NUMBER', 'NO'
        VARIABLE <- 'VARIABLE NAME', 'VAR NAME', 'NAME'
        TYPE     <- 'TYPE', 'VAR TYPE', 'DATA TYPE', 'VARIABLE TYPE'
        LEN      <- 'LENGTH', 'LEN', 'SIZE'
        POS      <- 'POSITION', 'POS', 'START POSITION', 'COLUMN POSITION', 'START'
        LABEL    <- 'VARIABLE LABEL', 'VAR LABEL', 'LABEL', 'DESCRIPTION', 'DESC'
    - Normalizes output to exactly these columns in order: NUM, VARIABLE, TYPE, LEN, POS, LABEL.
      Any column not in the desired set is dropped; any desired column absent from the source
      is added with empty (NaN) values.
    - Concatenates all sheets and writes outputs to:
        `DAC_Documents/dictionary{GGG_PARENT}{GG_PARENT}{G_PARENT}{YYYYMMDD}.csv`
        `DAC_Documents/dictionary{GGG_PARENT}{GG_PARENT}{G_PARENT}{YYYYMMDD}.xlsx`
    where GGG_PARENT, GG_PARENT, and G_PARENT are the third-to-last, second-to-last,
    and last segments of the output directory path, respectively.

Exit codes:
    0   Success
    1   Incorrect usage / missing args / I/O error

Requirements:
    - Python 3.8+
    - pandas
    - openpyxl (for Excel I/O)

Notes:
    - Intended for CLI use on Windows; paths may be absolute or relative.
    - Header row and mapping are configurable in code if format changes.
    - Column names in the mapping keys may contain spaces, carriage returns, or newlines; they are normalized before comparison.

ChangeLog:
    2026-01-05  Header updated to document behavior and column normalization.
    2026-01-06  Added normalize_col function to handle carriage returns and extra whitespace in column name mappings.
    2026-03-20  Expanded column rename map with variant names; added final normalization to enforce
                output columns NUM, VARIABLE, TYPE, LEN, POS, LABEL in that order.
"""

import pandas as pd
import os
import sys
import re
from datetime import date


def normalize_col(s: object) -> str:
    """Normalize a column name: uppercase, collapse any whitespace to single space, strip."""
    return re.sub(r'\s+', ' ', str(s)).upper().strip()


def main():
    # Step 1: Parse command-line arguments
    if len(sys.argv) != 4:
        print("Error: Invalid number of arguments")
        print("Usage: python metadata_to_dict_cli20260320.py <input_file> <output_dir> <trial_name>")
        print(
            "Example: python metadata_to_dict_cli20260320.py `C:\\data\\meta_data_summary.xlsx` `C:\\data\\output` FLINT2")
        sys.exit(1)

    file_path = sys.argv[1]
    output_base_dir = sys.argv[2]
    trial_name = sys.argv[3].upper()

    print(f"DEBUG: Input file: {file_path}")
    print(f"DEBUG: Output directory: {output_base_dir}")
    print(f"DEBUG: Trial name: {trial_name}")
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

    # Column rename mapping - maps many variant names to the standard output column names.
    # Keys may include spaces/carriage returns; they are normalized before comparison.
    col_rename_map = {
        # NUM variants
        'VARIABLE NUMBER': 'NUM',
        'VAR NUM': 'NUM',
        'VAR NO': 'NUM',
        'NUMBER': 'NUM',
        'NO': 'NUM',
        # VARIABLE variants
        'VARIABLE NAME': 'VARIABLE',
        'VAR NAME': 'VARIABLE',
        'NAME': 'VARIABLE',
        # TYPE variants
        'VAR TYPE': 'TYPE',
        'DATA TYPE': 'TYPE',
        'VARIABLE TYPE': 'TYPE',
        # LEN variants
        'LENGTH': 'LEN',
        'SIZE': 'LEN',
        # POS variants
        'POSITION': 'POS',
        'START POSITION': 'POS',
        'COLUMN POSITION': 'POS',
        'START': 'POS',
        # LABEL variants
        'VARIABLE LABEL': 'LABEL',
        'VAR LABEL': 'LABEL',
        'DESCRIPTION': 'LABEL',
        'DESC': 'LABEL',
    }

    # Normalize the keys of the rename map so entries with extra spaces/carriage returns match
    normalized_col_rename_map = {normalize_col(k): v for k, v in col_rename_map.items()}

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

            # Rename specific columns if they exist (use normalized rename map)
            existing_renames = {k: v for k, v in normalized_col_rename_map.items() if k in df.columns}
            if existing_renames:
                df.rename(columns=existing_renames, inplace=True)
                print(f"DEBUG: Renamed columns {existing_renames} in sheet: {sheet}")

            # Remove FORMAT and INFORMAT columns if they exist
            cols_to_drop = [c for c in ('FORMAT', 'INFORMAT') if c in df.columns]
            if cols_to_drop:
                df.drop(columns=cols_to_drop, inplace=True)
                print(f"DEBUG: Dropped columns {cols_to_drop} from sheet: {sheet}")

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

    # Step 9: Normalize output to the desired column set in the required order.
    # Any desired column not present in the data is added with empty (NaN) values;
    # any column outside the desired set is dropped.
    desired_columns = ['NUM', 'VARIABLE', 'TYPE', 'LEN', 'POS', 'LABEL']
    for col in desired_columns:
        if col not in combined_df.columns:
            combined_df[col] = pd.NA
            print(f"DEBUG: Added missing column '{col}' with empty values")
    combined_df = combined_df[desired_columns]
    print(f"DEBUG: Output normalized to columns: {desired_columns}")

    # Step 10: Save the combined DataFrame to CSV and Excel in DAC_Documents folder
    date_stamp = date.today().strftime('%Y%m%d')
    # Derive parent directory components from the output base directory
    path_parts = [p for p in output_base_dir.replace('\\', '/').split('/') if p]
    g_parent = path_parts[-1] if len(path_parts) >= 1 else ''
    gg_parent = path_parts[-2] if len(path_parts) >= 2 else ''
    ggg_parent = path_parts[-3] if len(path_parts) >= 3 else ''
    csv_output_path = os.path.join(dac_documents_dir, f"dictionary{ggg_parent}{gg_parent}{g_parent}{date_stamp}.csv")
    excel_output_path = os.path.join(dac_documents_dir, f"dictionary{ggg_parent}{gg_parent}{g_parent}{date_stamp}.xlsx")

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
