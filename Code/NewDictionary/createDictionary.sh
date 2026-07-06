#!/bin/bash

################################################################################
# Script Name: createDictionary.sh
# Description: Generates data dictionaries (CSV and Excel) from a folder of
#              SAS datasets (.sas7bdat).
#
# Purpose:
#   - Runs variable_info_cli20260320.sas to produce a variable-level Excel
#     workbook (one sheet per dataset).
#   - Runs metadata_to_dict_cli20260320.py to combine all sheets into a
#     single trial dictionary exported as both CSV and Excel.
#
# Usage:
#   ./createDictionary.sh -i <input_directory> [-o <output_directory>] [OPTIONS]
#
# Flags:
#   -i input_directory  - Path to directory containing SAS datasets
#                         (.sas7bdat) [required]
#   -o output_directory - Path where DAC_Documents will be created
#                         [optional, default: E:\output\dict]
#   -h, --help          - Display this usage summary
#
# Options:
#   --trial-name=NAME
#       Name used for the trial dictionary output file.
#       Default: grandparent folder name of the input directory.
#
#   --log=0|1
#       1 (default) - Save the SAS log file to DAC_Logs/.
#       0           - Suppress SAS log (.log) output.
#
#   --lst=0|1
#       0 (default) - Suppress SAS listing (.lst) output.
#       1           - Save the SAS listing file to DAC_Logs/.
#
# Examples:
#   ./createDictionary.sh -i "C:/studies/SampleStudy/rawdata"
#   ./createDictionary.sh -i "C:/studies/SampleStudy/rawdata" -o "E:/output/dict"
#   ./createDictionary.sh -i "C:/studies/SampleStudy/rawdata" -o "E:/output/dict" --trial-name=SampleStudy
#   ./createDictionary.sh -i "C:\studies\SampleStudy\rawdata" -o "E:\output\dict"
#   ./createDictionary.sh -i "C:/studies/SampleStudy/rawdata" --log=0
#
# Output Structure:
#   output_directory/
#   ├── DAC_Documents/
#   │   ├── variable_info_<GGG>_<GG>_<G>_<YYYYMMDD>.xlsx
#   │   ├── dictionary_<GGG>_<GG>_<G>_<YYYYMMDD>.csv
#   │   └── dictionary_<GGG>_<GG>_<G>_<YYYYMMDD>.xlsx
#   └── DAC_Logs/
#       ├── variable_info.log  (when --log=1)
#       └── variable_info.lst  (when --lst=1)
#
# Filename components:
#   <GGG>, <GG>, <G> are the third-to-last, second-to-last, and last segments
#   of the input directory path (alphanumeric characters only).
#   Example: C:/studies/SampleStudy/rawdata → C_studies_SampleStudy_rawdata
#
# Requirements:
#   - SAS Foundation 9.4 installed at its default location
#   - Python 3.8+ with pandas and openpyxl
#   - Bash shell — Git Bash on Windows or any POSIX-compatible shell
#   - variable_info_cli20260320.sas and metadata_to_dict_cli20260320.py must
#     be in the same directory as this script
#
# Author: DAC Development Team
#
################################################################################

set -e  # Exit on error

# Display usage information
usage() {
    echo "Usage: ./createDictionary.sh -i <input_dir> [-o <output_dir>] [OPTIONS]"
    echo ""
    echo "Flags:"
    echo "  -i input_dir               Path to directory containing SAS datasets (required)"
    echo "  -o output_dir              Path where DAC_Documents will be created"
    echo "                             (optional, default: E:/output/dict)"
    echo "  -h, --help                 Show this usage summary"
    echo ""
    echo "Options:"
    echo "  --trial-name=NAME"
    echo "      Name used for the trial dictionary output file."
    echo "      Default: grandparent folder name of the input directory."
    echo ""
    echo "  --log=0|1"
    echo "      1 (default) - Save SAS log file to DAC_Logs/."
    echo "      0           - Suppress SAS log (.log) output."
    echo ""
    echo "  --lst=0|1"
    echo "      0 (default) - Suppress SAS listing (.lst) output."
    echo "      1           - Save SAS listing file to DAC_Logs/."
    echo ""
    echo "Examples:"
    echo "  ./createDictionary.sh -i 'C:/studies/SampleStudy/rawdata'"
    echo "  ./createDictionary.sh -i 'C:/studies/SampleStudy/rawdata' -o 'E:/output/dict'"
    echo "  ./createDictionary.sh -i 'C:/studies/SampleStudy/rawdata' -o 'E:/output/dict' --trial-name=SampleStudy"
    exit "${1:-1}"
}

# Normalize a path: replace all backslashes with forward slashes
normalize_path() {
    echo "${1//\\//}"
}

# Default values
INPUT_DIR=""
OUTPUT_DIR=""
TRIAL_NAME=""
DS_LOG="1"
DS_LST="0"

# Parse flag arguments
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage 0
            ;;
        -i)
            if [ $# -lt 2 ]; then
                echo "Error: -i requires an argument"
                usage
            fi
            INPUT_DIR="$2"
            shift 2
            ;;
        -o)
            if [ $# -lt 2 ]; then
                echo "Error: -o requires an argument"
                usage
            fi
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --trial-name=*)
            TRIAL_NAME="${1#*=}"
            shift
            ;;
        --log=*)
            DS_LOG="${1#*=}"
            shift
            ;;
        --lst=*)
            DS_LST="${1#*=}"
            shift
            ;;
        *)
            echo "Error: Unknown option: $1"
            usage
            ;;
    esac
done

# Require -i flag (input directory)
if [ -z "$INPUT_DIR" ]; then
    echo "Error: Input directory is required (-i)"
    usage
fi

# Normalize path separators (support both '\' and '/')
INPUT_DIR=$(normalize_path "$INPUT_DIR")

# Validate input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Default output directory
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="E:/output/dict"
    echo "Output directory not specified; defaulting to: $OUTPUT_DIR"
fi

# Normalize output path separators
OUTPUT_DIR=$(normalize_path "$OUTPUT_DIR")

# Default TRIAL_NAME to the grandparent folder name of INPUT_DIR if not provided
if [ -z "$TRIAL_NAME" ]; then
    _leaf="${INPUT_DIR##*/}"
    _without_leaf="${INPUT_DIR%/*}"
    _parent_name="${_without_leaf##*/}"
    if [ -n "$_parent_name" ] && [ "$_parent_name" != "$INPUT_DIR" ]; then
        TRIAL_NAME="$_parent_name"
    else
        TRIAL_NAME=$(date +%Y%m%d)
    fi
fi

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Created output directory: $OUTPUT_DIR"
fi

# Warn if dictionary output files already exist in DAC_Documents
DAC_DOCS_DIR="${OUTPUT_DIR}/DAC_Documents"
if [ -d "$DAC_DOCS_DIR" ]; then
    _existing_dicts=()
    while IFS= read -r -d '' _dict_file; do
        _existing_dicts+=("$_dict_file")
    done < <(find "$DAC_DOCS_DIR" -maxdepth 1 \( -name 'dictionary_*.csv' -o -name 'dictionary_*.xlsx' \) -print0 2>/dev/null)
    if [ ${#_existing_dicts[@]} -gt 0 ]; then
        echo ""
        echo "Warning: The following dictionary file(s) already exist in: $DAC_DOCS_DIR"
        for _dict_file in "${_existing_dicts[@]}"; do
            echo "  $(basename "$_dict_file")"
        done
        echo ""
        read -rp "These files will be overwritten. Continue? [y/N] " _confirm
        case "$_confirm" in
            [yY]|[yY][eE][sS]) echo "" ;;
            *) echo "Aborted."; exit 1 ;;
        esac
    fi
fi

# Create DAC_Logs directory for SAS log and listing files
DAC_LOGS_DIR="${OUTPUT_DIR}/DAC_Logs"
if [ ! -d "$DAC_LOGS_DIR" ]; then
    mkdir -p "$DAC_LOGS_DIR"
    echo "Created logs directory: $DAC_LOGS_DIR"
fi

# Get script directory for finding SAS and Python scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define SAS executable path
SAS_EXE="C:/Program Files/SASHome/SASFoundation/9.4/sas.exe"

# Verify SAS executable exists
if [ ! -f "$SAS_EXE" ]; then
    echo "Error: SAS executable not found at: $SAS_EXE"
    echo "Please update the SAS_EXE variable in the script"
    exit 1
fi

# Detect null device (used when suppressing .log or .lst output)
if [ -e /dev/null ]; then
    NULL_DEVICE="/dev/null"
else
    NULL_DEVICE="NUL"
fi

LOG_ENABLED=1
[ "$DS_LOG" = "0" ] && LOG_ENABLED=0

LST_ENABLED=0
[ "$DS_LST" = "1" ] && LST_ENABLED=1

echo "=========================================="
echo "Create Dictionary"
echo "=========================================="
echo "Input Directory:  $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Trial Name:       $TRIAL_NAME"
echo "Script Directory: $SCRIPT_DIR"
echo "Log Files:        $DS_LOG"
echo "Listing Files:    $DS_LST"
echo "=========================================="

# Build SYSPARM for SAS scripts
SYSPARM="${INPUT_DIR}|${OUTPUT_DIR}"

# Step 1: Generate variable information workbook
echo "[1/2] Generating variable information document..."
LOG_ARG=$([ "$LOG_ENABLED" = "1" ] && echo "$DAC_LOGS_DIR/variable_info.log" || echo "$NULL_DEVICE")
SAS_PRINT_ARGS=()
if [ "$LST_ENABLED" = "1" ]; then
    SAS_PRINT_ARGS=(-print "$DAC_LOGS_DIR/variable_info.lst")
else
    SAS_PRINT_ARGS=(-print "$NULL_DEVICE")
fi
"$SAS_EXE" -sysparm "$SYSPARM" -sysin "$SCRIPT_DIR/variable_info_cli20260320.sas" -log "$LOG_ARG" "${SAS_PRINT_ARGS[@]}"
echo "      Complete.$([ "$LOG_ENABLED" = "1" ] && echo " Log: $DAC_LOGS_DIR/variable_info.log")"

# Locate the variable info file using the same three-part naming convention
# as variable_info_cli20260320.sas: variable_info_<GGG>_<GG>_<G>_<YYYYMMDD>.xlsx
normalized_path="${INPUT_DIR//\\//}"
IFS='/' read -ra path_segments <<< "$normalized_path"
clean_segments=()
for segment in "${path_segments[@]}"; do [ -n "$segment" ] && clean_segments+=("$segment"); done
segment_count="${#clean_segments[@]}"
[ "$segment_count" -ge 1 ] && leaf_segment_raw="${clean_segments[$(( segment_count-1 ))]}" || leaf_segment_raw=""
[ "$segment_count" -ge 2 ] && parent_segment_raw="${clean_segments[$(( segment_count-2 ))]}" || parent_segment_raw=""
[ "$segment_count" -ge 3 ] && grandparent_segment_raw="${clean_segments[$(( segment_count-3 ))]}" || grandparent_segment_raw=""
G_PARENT=$(echo "$leaf_segment_raw"        | tr -cd '[:alnum:]_')
GG_PARENT=$(echo "$parent_segment_raw"     | tr -cd '[:alnum:]_')
GGG_PARENT=$(echo "$grandparent_segment_raw" | tr -cd '[:alnum:]_')
DATE_STAMP=$(date +%Y%m%d)
VARIABLE_INFO_FILE="${OUTPUT_DIR}/DAC_Documents/variable_info_${GGG_PARENT}_${GG_PARENT}_${G_PARENT}_${DATE_STAMP}.xlsx"

if [ -f "$VARIABLE_INFO_FILE" ]; then
    echo "Variable info file created: $VARIABLE_INFO_FILE"
else
    echo "Warning: Variable info file not found at expected location:"
    echo "$VARIABLE_INFO_FILE"
    ACTUAL_FILE=$(find "$OUTPUT_DIR/DAC_Documents" -name "variable_info_*.xlsx" 2>/dev/null | head -1)
    if [ -n "$ACTUAL_FILE" ]; then
        VARIABLE_INFO_FILE="$ACTUAL_FILE"
        echo "Found variable info file at: $VARIABLE_INFO_FILE"
    else
        echo "Error: Could not locate variable info file. Aborting."
        exit 1
    fi
fi
echo "=========================================="

# Step 2: Build trial dictionary from variable info workbook
echo "[2/2] Generating trial dictionary from variable information..."

# Detect available Python command
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v py &> /dev/null; then
    PYTHON_CMD="py"
fi

if [ -n "$PYTHON_CMD" ]; then
    echo "      Using Python command: $PYTHON_CMD"
    "$PYTHON_CMD" "$SCRIPT_DIR/metadata_to_dict_cli20260320.py" "$VARIABLE_INFO_FILE" "$OUTPUT_DIR" "$TRIAL_NAME" "$INPUT_DIR"
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
echo "Dictionary Creation Complete!"
echo "=========================================="
echo "Output files located in: $OUTPUT_DIR/DAC_Documents"
if [ "$LOG_ENABLED" = "1" ]; then
    echo "Log files located in:    $DAC_LOGS_DIR"
fi
echo "=========================================="

exit 0
