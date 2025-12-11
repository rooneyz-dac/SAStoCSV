#!/bin/bash

################################################################################
# Script Name: dataPreprocessing.sh
# Description: Automated SAS data preprocessing pipeline that converts SAS
#              datasets between formats and generates documentation
#
# Purpose:
#   - Converts SAS7BDAT files to XPT format (and vice versa if XPT files exist)
#   - Exports SAS datasets to CSV format
#   - Generates variable information documentation
#   - Creates data specifications document
#   - Produces library information summary
#   - Builds trial dictionary from variable metadata
#
# Usage:
#   ./dataPreprocessing.sh <input_directory> <output_directory> [trial_name]
#
# Arguments:
#   input_directory  - Path to directory containing SAS datasets (.sas7bdat or .xpt)
#   output_directory - Path where outputs and documentation will be saved
#   trial_name       - (Optional) Name for trial dictionary; defaults to YYYYMMDD
#
# Examples:
#   ./dataPreprocessing.sh "C:/data/input" "C:/data/output" FLINT2
#   ./dataPreprocessing.sh "/path/to/sas/data" "/path/to/output"
#
# Output Structure:
#   output_directory/
#   ├── DAC_XPT/              - XPT format datasets
#   ├── DAC_SDTM/             - Converted SAS datasets (if XPT input detected)
#   ├── DAC_CSV/              - CSV exports
#   ├── DAC_Documents/        - All documentation files
#   │   ├── variable_info_*.xlsx
#   │   ├── data_specs_*.xlsx
#   │   ├── library_info_*.xlsx
#   │   └── *_dictionary.xlsx
#   ├── *.log                 - SAS execution logs
#   └── pipeline_vars.env     - Environment variables for chaining scripts
#
# Requirements:
#   - SAS Foundation 9.4 installed at default location
#   - Python 3.x with pandas
#   - Bash shell (Git Bash on Windows)
#   - Required SAS scripts in same directory:
#     * SAStoXPTcli20251121v2WORKING.sas
#     * SAStoCSVcli2InputsWORKING20251121.sas
#     * variable_info_cli20251122.sas
#     * data_specs_cli20251122.sas
#     * library_info_cli20251122.sas
#   - Required Python script:
#     * metadata_to_dict_cli20251122.py
#
# Author: [Your Name]
# Created: 2025-11-22
# Version: 1.0
#
# Notes:
#   - Script exits on first error (set -e)
#   - Automatically adjusts input path if XPT conversion creates DAC_SDTM
#   - Exports VARIABLE_INFO_FILE and TRIAL_NAME for downstream processing
#
################################################################################

set -e  # Exit on error

#!/bin/bash

################################################################################
# Data Preprocessing Pipeline Script
# Purpose: Automates SAS data conversion and documentation generation
# Usage: ./dataPreprocessing.sh <input_dir> <output_dir> <trial_name>
################################################################################

# Check if correct number of arguments provided
if [ $# -ne 3 ]; then
    echo "Error: Invalid number of arguments"
    echo "Usage: ./dataPreprocessing.sh <input_dir> <output_dir> <trial_name>"
    echo "Example: ./dataPreprocessing.sh 'C:/data/input' 'C:/data/output' 'FLINT2'"
    exit 1
fi

# Parse command line arguments
INPUT_DIR="$1"
OUTPUT_DIR="$2"
TRIAL_NAME="$3"

# Validate input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Created output directory: $OUTPUT_DIR"
fi

# Get script directory for finding SAS scripts and Python script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define SAS executable path (modify as needed for your system)
SAS_EXE="C:/Program Files/SASHome/SASFoundation/9.4/sas.exe"

# Verify SAS executable exists
if [ ! -f "$SAS_EXE" ]; then
    echo "Error: SAS executable not found at: $SAS_EXE"
    echo "Please update the SAS_EXE variable in the script"
    exit 1
fi

# Create SYSPARM for SAS scripts
SYSPARM="${INPUT_DIR}|${OUTPUT_DIR}"

echo "=========================================="
echo "Data Preprocessing Pipeline"
echo "=========================================="
echo "Input Directory:  $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Trial Name:       $TRIAL_NAME"
echo "Script Directory: $SCRIPT_DIR"
echo "=========================================="

# 1. Run SAS to XPT conversion
echo "[1/6] Converting SAS datasets to XPT format..."
"$SAS_EXE" -sysparm "$SYSPARM" -sysin "$SCRIPT_DIR/SAStoXPTcli20251121v2WORKING.sas" -log "$OUTPUT_DIR/sas_to_xpt.log"
echo "      Complete. Log: $OUTPUT_DIR/sas_to_xpt.log"

# Check if XPT files were converted to SAS7BDAT (DAC_SDTM folder created)
DAC_SDTM_DIR="${OUTPUT_DIR}/DAC_SDTM"
if [ -d "$DAC_SDTM_DIR" ] && [ "$(ls -A "$DAC_SDTM_DIR" 2>/dev/null)" ]; then
    echo "      Note: XPT files detected and converted to SAS7BDAT format"
    echo "      SDTM datasets available in: $DAC_SDTM_DIR"
fi

# 2. Run SAS to CSV conversion
echo "[2/6] Converting SAS datasets to CSV..."
"$SAS_EXE" -sysparm "$SYSPARM" -sysin "$SCRIPT_DIR/SAStoCSVcli2InputsWORKING20251121.sas" -log "$OUTPUT_DIR/sas_to_csv.log"
echo "      Complete. Log: $OUTPUT_DIR/sas_to_csv.log"

# 3. Generate variable information and capture output file location
echo "[3/6] Generating variable information document..."
"$SAS_EXE" -sysparm "$SYSPARM" -sysin "$SCRIPT_DIR/variable_info_cli20251122.sas" -log "$OUTPUT_DIR/variable_info.log"
echo "      Complete. Log: $OUTPUT_DIR/variable_info.log"

# 4. Generate data specifications
echo "[4/6] Generating data specifications document..."
"$SAS_EXE" -sysparm "$SYSPARM" -sysin "$SCRIPT_DIR/data_specs_cli20251122.sas" -log "$OUTPUT_DIR/data_specs.log"
echo "      Complete. Log: $OUTPUT_DIR/data_specs.log"

# 5. Generate library information
echo "[5/6] Generating library information document..."
"$SAS_EXE" -sysparm "$SYSPARM" -sysin "$SCRIPT_DIR/library_info_cli20251122.sas" -log "$OUTPUT_DIR/library_info.log"
echo "      Complete. Log: $OUTPUT_DIR/library_info.log"

# Extract variable info file path from log
LIBNAME=$(basename "$INPUT_DIR" | tr -cd '[:alnum:]')
DATE_STAMP=$(date +%Y%m%d)
export VARIABLE_INFO_FILE="${OUTPUT_DIR}/DAC_Documents/variable_info_${LIBNAME}_${DATE_STAMP}.xlsx"

# Verify file was created
if [ -f "$VARIABLE_INFO_FILE" ]; then
    echo "Variable info file created: $VARIABLE_INFO_FILE"
    echo "=========================================="
else
    echo "Warning: Variable info file not found at expected location:"
    echo "$VARIABLE_INFO_FILE"
    # Try to find it in DAC_Documents
    ACTUAL_FILE=$(find "$OUTPUT_DIR/DAC_Documents" -name "variable_info_*.xlsx" 2>/dev/null | head -1)
    if [ -n "$ACTUAL_FILE" ]; then
        export VARIABLE_INFO_FILE="$ACTUAL_FILE"
        echo "Found variable info file at: $VARIABLE_INFO_FILE"
    fi
fi

# 6. Generate trial dictionary from variable info
echo "[6/6] Generating trial dictionary from variable information..."

# Detect available Python command
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v py &> /dev/null; then
    PYTHON_CMD="py"
fi

# Execute Python script if command found
if [ -n "$PYTHON_CMD" ]; then
    echo "      Using Python command: $PYTHON_CMD"
    "$PYTHON_CMD" "$SCRIPT_DIR/metadata_to_dict_cli20251122.py" "$VARIABLE_INFO_FILE" "$OUTPUT_DIR" "$TRIAL_NAME"
    if [ $? -eq 0 ]; then
        echo "      Complete."
    else
        echo "      Error: Python script failed. Check the error messages above."
        exit 1
    fi
else
    echo "      Error: Python not found in PATH"
    echo "      Please install Python or add it to your PATH"
    echo "      Required commands: python3, python, or py"
    exit 1
fi

echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Output files located in:"
echo "  - Documents: $OUTPUT_DIR/DAC_Documents"
echo "  - CSV:       $OUTPUT_DIR/DAC_CSV"
echo "  - XPT:       $OUTPUT_DIR/DAC_XPT"
if [ -d "$DAC_SDTM_DIR" ]; then
    echo "  - SDTM:      $DAC_SDTM_DIR"
fi
echo "=========================================="

exit 0

