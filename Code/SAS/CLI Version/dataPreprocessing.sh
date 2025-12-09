#!/bin/bash

# dataPreprocessing.sh
# Automated SAS data preprocessing pipeline
# Runs all SAS scripts and captures variable info file location

set -e  # Exit on error

# Check if correct number of arguments provided
if [ "$#" -ne 2 ]; then
    echo "Error: Incorrect number of arguments"
    echo "Usage: $0 <input_directory> <output_directory>"
    echo "Example: $0 /path/to/input /path/to/output"
    exit 1
fi

# Assign command-line arguments
INPUT_DIR="$1"
OUTPUT_DIR="$2"

# Validate input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Define SAS executable path
SAS_EXE="/c/Program Files/SASHome/SASFoundation/9.4/sas.exe"

# Define script directory (assumes scripts are in same directory as this bash script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define SYSPARM parameter
SYSPARM="${INPUT_DIR}|${OUTPUT_DIR}"

echo "=========================================="
echo "SAS Data Preprocessing Pipeline"
echo "=========================================="
echo "Input Directory: $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Script Directory: $SCRIPT_DIR"
echo "=========================================="

# 1. Run SAS to XPT conversion (and XPT to SAS7BDAT if XPT files exist)
echo "[1/5] Converting SAS datasets to XPT format (and XPT to SAS7BDAT if present)..."
"$SAS_EXE" -sysparm "$SYSPARM" -sysin "$SCRIPT_DIR/SAStoXPTcli20251121v2WORKING.sas" -log "$OUTPUT_DIR/sas_to_xpt.log"
echo "      Complete. Log: $OUTPUT_DIR/sas_to_xpt.log"

# Check if XPT files were converted to SAS7BDAT (DAC_SDTM folder created)
DAC_SDTM_DIR="${OUTPUT_DIR}/DAC_SDTM"
if [ -d "$DAC_SDTM_DIR" ] && [ "$(ls -A "$DAC_SDTM_DIR" 2>/dev/null)" ]; then
    echo "      XPT files detected and converted to SAS7BDAT"
    echo "      Updating input directory to: $DAC_SDTM_DIR"
    INPUT_DIR="$DAC_SDTM_DIR"
    SYSPARM="${INPUT_DIR}|${OUTPUT_DIR}"
fi

# 2. Run SAS to CSV conversion
echo "[2/5] Converting SAS datasets to CSV..."
"$SAS_EXE" -sysparm "$SYSPARM" -sysin "$SCRIPT_DIR/SAStoCSVcli2InputsWORKING20251121.sas" -log "$OUTPUT_DIR/sas_to_csv.log"
echo "      Complete. Log: $OUTPUT_DIR/sas_to_csv.log"

# 3. Generate variable information and capture output file location
echo "[3/5] Generating variable information document..."
"$SAS_EXE" -sysparm "$SYSPARM" -sysin "$SCRIPT_DIR/variable_info_cli20251122.sas" -log "$OUTPUT_DIR/variable_info.log"
echo "      Complete. Log: $OUTPUT_DIR/variable_info.log"

# 4. Generate data specifications
echo "[4/5] Generating data specifications document..."
"$SAS_EXE" -sysparm "$SYSPARM" -sysin "$SCRIPT_DIR/data_specs_cli20251122.sas" -log "$OUTPUT_DIR/data_specs.log"
echo "      Complete. Log: $OUTPUT_DIR/data_specs.log"

# 5. Generate library information
echo "[5/5] Generating library information document..."
"$SAS_EXE" -sysparm "$SYSPARM" -sysin "$SCRIPT_DIR/library_info_cli20251122.sas" -log "$OUTPUT_DIR/library_info.log"
echo "      Complete. Log: $OUTPUT_DIR/library_info.log"

# Extract variable info file path from log
LIBNAME=$(basename "$INPUT_DIR" | tr -cd '[:alnum:]')
DATE_STAMP=$(date +%Y%m%d)
export VARIABLE_INFO_FILE="${OUTPUT_DIR}/DAC_Documents/variable_info_${LIBNAME}_${DATE_STAMP}.xlsx"

# Verify file was created
if [ -f "$VARIABLE_INFO_FILE" ]; then
    echo ""
    echo "=========================================="
    echo "Pipeline completed successfully!"
    echo "=========================================="
    echo "Variable Info File: $VARIABLE_INFO_FILE"
    echo "Output Directory: $OUTPUT_DIR"
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

# Export variable for use in subsequent scripts
echo "VARIABLE_INFO_FILE=$VARIABLE_INFO_FILE" > "$OUTPUT_DIR/pipeline_vars.env"

exit 0