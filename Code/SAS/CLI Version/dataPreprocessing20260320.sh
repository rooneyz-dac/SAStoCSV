#!/bin/bash

################################################################################
# Script Name: dataPreprocessing20260320.sh
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
#   ./dataPreprocessing20260320.sh <input_directory> [output_directory] [OPTIONS]
#
# Arguments:
#   input_directory  - Path to directory containing SAS datasets (.sas7bdat or .xpt) [required]
#   output_directory - Path where outputs and documentation will be saved [optional, default: E:\output]
#
# Options:
#   --trial-name=NAME
#       Name used for the trial dictionary output file.
#       Default: current date (YYYYMMDD).
#
#   --format=long|condensed|wide
#       Controls how variables are listed in each dataset summary tab.
#         long      - One row per variable value (default).
#         condensed - One row per variable with values collapsed into a
#                     single cell.
#         wide      - One row per dataset with variables as columns.
#
#   --order=varnum|name
#       Determines the order in which variables appear in dataset summary tabs.
#         varnum - Order by position in the dataset (default).
#         name   - Order alphabetically by variable name.
#
#   --index=VAR[,VAR...]
#       One or more index variables (e.g. USUBJID) used to count distinct
#       patients or other units of interest within each dataset.
#       Default: none.
#
#   --cat-threshold=N
#       Maximum number of distinct levels a variable may have before
#       individual frequencies and percentages are replaced with
#       distribution statistics (mean, std, min, max). Must be >= 0.
#       Default: 10.
#
#   --where=CLAUSE
#       A WHERE clause applied to the SAS dictionary metadata to subset
#       which datasets are included in the specifications document.
#       Example: --where=%str(memname in ('AE','CM','DM'))
#       Default: none (all datasets included).
#
#   --debug=0|1
#       0 (default) - NOTES suppressed; temporary work datasets cleaned up.
#       1           - NOTES shown in the SAS log; temporary datasets
#                     retained in the WORK library for inspection.
#
#   --log=0|1
#       1 (default) - Save SAS log files to the output directory.
#       0           - Suppress SAS log (.log) files; output is routed to the
#                     null device so no log file is written.
#
#   --lst=0|1
#       0 (default) - Suppress SAS listing (.lst) files; output is routed
#                     to the null device so no listing file is written.
#       1           - Save SAS listing files to the output directory.
#
# Examples:
#   ./dataPreprocessing20260320.sh "C:/data/input"
#   ./dataPreprocessing20260320.sh "C:/data/input" "C:/data/output"
#   ./dataPreprocessing20260320.sh "C:/data/input" "C:/data/output" --trial-name=FLINT2 --format=wide
#   ./dataPreprocessing20260320.sh "/path/to/sas/data" "/path/to/output" --format=condensed --debug=1
#   ./dataPreprocessing20260320.sh "C:/data/input" "C:/data/output" --lst=1
#   ./dataPreprocessing20260320.sh "C:/data/input" "C:/data/output" --log=1
#
# Output Structure:
#   output_directory/
#   ├── DAC_XPT/              - XPT format datasets
#   ├── DAC_SDTM/             - Converted SAS datasets (if XPT input detected)
#   ├── DAC_CSV/              - CSV exports
#   ├── DAC_Documents/        - All documentation files
#   │   ├── variable_info_{GGG_PARENT}_{GG_PARENT}_{G_PARENT}_{YYYYMMDD}.xlsx
#   │   ├── data_specs_{GGG_PARENT}_{GG_PARENT}_{G_PARENT}_{YYYYMMDD}.xlsx
#   │   ├── library_info_{GGG_PARENT}_{GG_PARENT}_{G_PARENT}_{YYYYMMDD}.xlsx
#   │   └── dictionary_{GGG_PARENT}_{GG_PARENT}_{G_PARENT}_{YYYYMMDD}.{csv,xlsx}
#   ├── *.log                 - SAS execution logs (only when --log=1)
#   ├── *.lst                 - SAS listing files (only when --lst=1)
#   └── pipeline_vars.env     - Environment variables for chaining scripts
#
# Requirements:
#   - SAS Foundation 9.4 installed at default location
#   - Python 3.x with pandas
#   - Bash shell (Git Bash on Windows)
#   - Required SAS scripts in same directory:
#     * SAStoXPTcli20260320.sas
#     * SAStoCSVcli20260320.sas
#     * variable_info_cli20260320.sas
#     * data_specs_cli20260320.sas
#     * library_info_cli20260320.sas
#   - Required Python script:
#     * metadata_to_dict_cli20260320.py
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

# Display usage information
usage() {
    echo "Usage: ./dataPreprocessing20260320.sh <input_dir> [output_dir] [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  input_dir              Path to directory containing SAS datasets (required)"
    echo "  output_dir             Path where outputs will be saved (optional, default: E:\\output)"
    echo ""
    echo "Options:"
    echo "  --trial-name=NAME"
    echo "      Name used for the trial dictionary output file."
    echo "      Default: current date (YYYYMMDD)."
    echo ""
    echo "  --format=long|condensed|wide"
    echo "      Controls how variables are listed in each dataset summary tab."
    echo "        long      - One row per variable value (default)."
    echo "        condensed - One row per variable with values collapsed into a single cell."
    echo "        wide      - One row per dataset with variables as columns."
    echo ""
    echo "  --order=varnum|name"
    echo "      Determines the order in which variables appear in dataset summary tabs."
    echo "        varnum - Order by position in the dataset (default)."
    echo "        name   - Order alphabetically by variable name."
    echo ""
    echo "  --index=VAR[,VAR...]"
    echo "      One or more index variables (e.g. USUBJID) used to count distinct"
    echo "      patients or other units of interest within each dataset."
    echo "      Default: none."
    echo ""
    echo "  --cat-threshold=N"
    echo "      Maximum number of distinct levels a variable may have before"
    echo "      individual frequencies and percentages are replaced with distribution"
    echo "      statistics (mean, std, min, max). Must be >= 0. Default: 10."
    echo ""
    echo "  --where=CLAUSE"
    echo "      A WHERE clause applied to the SAS dictionary metadata to subset"
    echo "      which datasets are included in the specifications document."
    echo "      Default: none (all datasets included)."
    echo ""
    echo "  --debug=0|1"
    echo "      0 (default) - NOTES suppressed; temporary work datasets cleaned up."
    echo "      1           - NOTES shown in the SAS log; temporary datasets retained."
    echo ""
    echo "  --log=0|1"
    echo "      1 (default) - Save SAS log files to the output directory."
    echo "      0           - Suppress SAS log (.log) files (routed to null device)."
    echo ""
    echo "  --lst=0|1"
    echo "      0 (default) - Suppress SAS listing (.lst) files."
    echo "      1           - Save SAS listing files to the output directory."
    echo ""
    echo "Examples:"
    echo "  ./dataPreprocessing20260320.sh 'C:/data/input'"
    echo "  ./dataPreprocessing20260320.sh 'C:/data/input' 'C:/data/output'"
    echo "  ./dataPreprocessing20260320.sh 'C:/data/input' 'C:/data/output' --trial-name=FLINT2 --format=wide"
    echo "  ./dataPreprocessing20260320.sh 'C:/data/input' 'C:/data/output' --lst=1"
    echo "  ./dataPreprocessing20260320.sh 'C:/data/input' 'C:/data/output' --log=1"
    exit 1
}

# Require at least one argument (input directory)
if [ $# -lt 1 ]; then
    echo "Error: Input directory is required"
    usage
fi

# Parse first positional argument: input directory
INPUT_DIR="$1"
shift

# Parse optional second positional argument: output directory (if not a flag)
OUTPUT_DIR="E:\\output"
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
    OUTPUT_DIR="$1"
    shift
fi

# Default values for optional data_specs parameters
TRIAL_NAME=""
DS_FORMAT="long"
DS_ORDER="varnum"
DS_INDEX=""
DS_CAT_THRESHOLD="10"
DS_WHERE=""
DS_DEBUG="0"
DS_LOG="1"
DS_LST="0"

# Parse remaining flag arguments
for arg in "$@"; do
    case "$arg" in
        --trial-name=*)
            TRIAL_NAME="${arg#*=}"
            ;;
        --format=*)
            DS_FORMAT="${arg#*=}"
            ;;
        --order=*)
            DS_ORDER="${arg#*=}"
            ;;
        --index=*)
            DS_INDEX="${arg#*=}"
            ;;
        --cat-threshold=*)
            DS_CAT_THRESHOLD="${arg#*=}"
            ;;
        --where=*)
            DS_WHERE="${arg#*=}"
            ;;
        --debug=*)
            DS_DEBUG="${arg#*=}"
            ;;
        --log=*)
            DS_LOG="${arg#*=}"
            ;;
        --lst=*)
            DS_LST="${arg#*=}"
            ;;
        *)
            echo "Error: Unknown option: $arg"
            usage
            ;;
    esac
done

# Default TRIAL_NAME to current date if not provided
if [ -z "$TRIAL_NAME" ]; then
    TRIAL_NAME=$(date +%Y%m%d)
fi

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

# Build base SYSPARM for most SAS scripts
SYSPARM="${INPUT_DIR}|${OUTPUT_DIR}"

# Build the SAS -log and -print arguments based on the --log and --lst toggles.
# When DS_LOG=1 (default), save .log files to the output directory.
# When DS_LOG=0, suppress .log output using the null device.
# When DS_LST=1, save .lst files to the output directory.
# When DS_LST=0 (default), omit the -print flag (SAS uses its default destination).

# Detect null device (used when suppressing .log or .lst output)
if [ -e /dev/null ]; then
    NULL_DEVICE="/dev/null"
else
    NULL_DEVICE="NUL"
fi

if [ "$DS_LST" = "1" ]; then
    LST_ENABLED=1
else
    LST_ENABLED=0
fi

if [ "$DS_LOG" = "0" ]; then
    LOG_ENABLED=0
else
    LOG_ENABLED=1
fi

echo "=========================================="
echo "Data Preprocessing Pipeline"
echo "=========================================="
echo "Input Directory:  $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Trial Name:       $TRIAL_NAME"
echo "Script Directory: $SCRIPT_DIR"
echo "Data Specs Options:"
echo "  Format:         $DS_FORMAT"
echo "  Order:          $DS_ORDER"
echo "  Index:          ${DS_INDEX:-<none>}"
echo "  Cat Threshold:  $DS_CAT_THRESHOLD"
echo "  Where:          ${DS_WHERE:-<none>}"
echo "  Debug:          $DS_DEBUG"
echo "  Log Files:      $DS_LOG"
echo "  Listing Files:  $DS_LST"
echo "=========================================="

# 1. Run SAS to XPT conversion
echo "[1/6] Converting SAS datasets to XPT format..."
LOG_ARG_1=$([ "$LOG_ENABLED" = "1" ] && echo "$OUTPUT_DIR/sas_to_xpt.log" || echo "$NULL_DEVICE")
SAS_PRINT_1=()
if [ "$LST_ENABLED" = "1" ]; then
    SAS_PRINT_1=(-print "$OUTPUT_DIR/sas_to_xpt.lst")
fi
"$SAS_EXE" -sysparm "$SYSPARM" -sysin "$SCRIPT_DIR/SAStoXPTcli20260320.sas" -log "$LOG_ARG_1" "${SAS_PRINT_1[@]}"
echo "      Complete.$([ "$LOG_ENABLED" = "1" ] && echo " Log: $OUTPUT_DIR/sas_to_xpt.log")"

# Check if XPT files were converted to SAS7BDAT (DAC_SDTM folder created)
DAC_SDTM_DIR="${OUTPUT_DIR}/DAC_SDTM"
if [ -d "$DAC_SDTM_DIR" ] && [ "$(ls -A "$DAC_SDTM_DIR" 2>/dev/null)" ]; then
    echo "      Note: XPT files detected and converted to SAS7BDAT format"
    echo "      SDTM datasets available in: $DAC_SDTM_DIR"
    echo "      Updating input path to converted SAS files: $DAC_SDTM_DIR"
    INPUT_DIR="$DAC_SDTM_DIR"
    SYSPARM="${INPUT_DIR}|${OUTPUT_DIR}"
fi

# 2. Run SAS to CSV conversion
echo "[2/6] Converting SAS datasets to CSV..."
LOG_ARG_2=$([ "$LOG_ENABLED" = "1" ] && echo "$OUTPUT_DIR/sas_to_csv.log" || echo "$NULL_DEVICE")
SAS_PRINT_2=()
if [ "$LST_ENABLED" = "1" ]; then
    SAS_PRINT_2=(-print "$OUTPUT_DIR/sas_to_csv.lst")
fi
"$SAS_EXE" -sysparm "$SYSPARM" -sysin "$SCRIPT_DIR/SAStoCSVcli20260320.sas" -log "$LOG_ARG_2" "${SAS_PRINT_2[@]}"
echo "      Complete.$([ "$LOG_ENABLED" = "1" ] && echo " Log: $OUTPUT_DIR/sas_to_csv.log")"

# 3. Generate variable information and capture output file location
echo "[3/6] Generating variable information document..."
LOG_ARG_3=$([ "$LOG_ENABLED" = "1" ] && echo "$OUTPUT_DIR/variable_info.log" || echo "$NULL_DEVICE")
SAS_PRINT_3=()
if [ "$LST_ENABLED" = "1" ]; then
    SAS_PRINT_3=(-print "$OUTPUT_DIR/variable_info.lst")
fi
"$SAS_EXE" -sysparm "$SYSPARM" -sysin "$SCRIPT_DIR/variable_info_cli20260320.sas" -log "$LOG_ARG_3" "${SAS_PRINT_3[@]}"
echo "      Complete.$([ "$LOG_ENABLED" = "1" ] && echo " Log: $OUTPUT_DIR/variable_info.log")"

# 4. Generate data specifications
echo "[4/6] Generating data specifications document..."
DATA_SPECS_SYSPARM="${INPUT_DIR}|${OUTPUT_DIR}|index=${DS_INDEX}|cat_threshold=${DS_CAT_THRESHOLD}|format=${DS_FORMAT}|order=${DS_ORDER}|where=${DS_WHERE}|debug=${DS_DEBUG}"
LOG_ARG_4=$([ "$LOG_ENABLED" = "1" ] && echo "$OUTPUT_DIR/data_specs.log" || echo "$NULL_DEVICE")
SAS_PRINT_4=()
if [ "$LST_ENABLED" = "1" ]; then
    SAS_PRINT_4=(-print "$OUTPUT_DIR/data_specs.lst")
fi
"$SAS_EXE" -sysparm "$DATA_SPECS_SYSPARM" -sysin "$SCRIPT_DIR/data_specs_cli20260320.sas" -log "$LOG_ARG_4" "${SAS_PRINT_4[@]}"
echo "      Complete.$([ "$LOG_ENABLED" = "1" ] && echo " Log: $OUTPUT_DIR/data_specs.log")"

# 5. Generate library information
echo "[5/6] Generating library information document..."
LOG_ARG_5=$([ "$LOG_ENABLED" = "1" ] && echo "$OUTPUT_DIR/library_info.log" || echo "$NULL_DEVICE")
SAS_PRINT_5=()
if [ "$LST_ENABLED" = "1" ]; then
    SAS_PRINT_5=(-print "$OUTPUT_DIR/library_info.lst")
fi
"$SAS_EXE" -sysparm "$SYSPARM" -sysin "$SCRIPT_DIR/library_info_cli20260320.sas" -log "$LOG_ARG_5" "${SAS_PRINT_5[@]}"
echo "      Complete.$([ "$LOG_ENABLED" = "1" ] && echo " Log: $OUTPUT_DIR/library_info.log")"

# Extract variable info file path from log
GGG_PARENT=$(basename "$(dirname "$(dirname "$INPUT_DIR")")" | tr -cd '[:alnum:]')
GG_PARENT=$(basename "$(dirname "$INPUT_DIR")" | tr -cd '[:alnum:]')
G_PARENT=$(basename "$INPUT_DIR" | tr -cd '[:alnum:]')
DATE_STAMP=$(date +%Y%m%d)
export VARIABLE_INFO_FILE="${OUTPUT_DIR}/DAC_Documents/variable_info_${GGG_PARENT}_${GG_PARENT}_${G_PARENT}_${DATE_STAMP}.xlsx"

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
    "$PYTHON_CMD" "$SCRIPT_DIR/metadata_to_dict_cli20260320.py" "$VARIABLE_INFO_FILE" "$OUTPUT_DIR"
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

