#!/usr/bin/env python3
"""
Script: metadata_to_dict_cli20251122.py
Purpose: Generate trial dictionary from metadata summary Excel file
         Accepts CLI parameters for input/output paths
Usage: python metadata_to_dict_cli20251122.py <input_file> <output_dir> <trial_name>
Example: python metadata_to_dict_cli20251122.py "C:\data\meta_data_summary.xlsx" "C:\data\output" "FLINT2"
"""

import pandas as pd
import os
import sys


def main():
    # Step 1: Parse command-line arguments
    if len(sys.argv) != 4:
        print("Error: Invalid number of arguments")
        print("Usage: python metadata_to_dict_cli20251122.py <input_file> <output_dir> <trial_name>")
        print(
            "Example: python metadata_to_dict_cli20251122.py 'C:\\data\\meta_data_summary.xlsx' 'C:\\data\\output' 'FLINT2'")
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

    # Step 7: Loop through all sheets
    for sheet in excel_data.sheet_names[0:]:
        print(f"DEBUG: Processing sheet: {sheet}")
        try:
            # Load the sheet data, specifying that the header is in row 3 (index 2)
            df = pd.read_excel(file_path, sheet_name=sheet, header=2)

            # Normalize column names to uppercase for consistent comparison
            df.columns = df.columns.str.upper()

            # Add the sheet name as a new column for reference
            df['SOURCE_TAB'] = sheet

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
    csv_output_path = os.path.join(dac_documents_dir, f"{trial_name}_dictionary.csv")
    excel_output_path = os.path.join(dac_documents_dir, f"{trial_name}_dictionary.xlsx")

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
