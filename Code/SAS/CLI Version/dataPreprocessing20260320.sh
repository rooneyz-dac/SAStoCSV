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
#   ./dataPreprocessing20260320.sh -i <input_directory> [-o <output_directory>] [OPTIONS]
#
# Flags:
#   -i input_directory  - Path to directory containing SAS datasets (.sas7bdat or .xpt) [required]
#   -o output_directory - Path where outputs and documentation will be saved [optional, default: parent of input directory]
#   -H, --detailed-help - Display detailed help showing which scripts use each flag,
#                         which output files are affected, and the effect of each option
#
# Options:
#   --trial-name=NAME
#       Name used for the trial dictionary output file.
#       Default: grandparent folder name of the input directory.
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
#   ./dataPreprocessing20260320.sh -i "C:/data/study/rawdata"
#   ./dataPreprocessing20260320.sh -i "C:/data/study/rawdata" -o "C:/data/output"
#   ./dataPreprocessing20260320.sh -i "C:/data/study/rawdata" -o "C:/data/output" --trial-name=SampleStudy --format=wide
#   ./dataPreprocessing20260320.sh -i "/path/to/sas/data" -o "/path/to/output" --format=condensed --debug=1
#   ./dataPreprocessing20260320.sh -i "C:/data/study/rawdata" -o "C:/data/output" --lst=1
#   ./dataPreprocessing20260320.sh -i "C:/data/study/rawdata" -o "C:/data/output" --log=1
#
# Output Structure:
#   output_directory/
#   ├── DAC_<ParentFolderName>/  - Standardized SAS datasets (if any SAS file was renamed)
#   ├── DAC_XPT/              - XPT format datasets (and/or standardized XPT, if renamed)
#   ├── DAC_SDTM/             - Converted SAS datasets (if XPT input detected)
#   ├── DAC_CSV/              - CSV exports
#   ├── DAC_Documents/        - All documentation files
#   │   ├── variable_info_*.xlsx
#   │   ├── data_specs_*.xlsx
#   │   ├── library_info_*.xlsx
#   │   ├── dictionary_*.csv
#   │   └── dictionary_*.xlsx
#   ├── DAC_Logs/             - SAS log and listing files
#   │   ├── *.log             - SAS execution logs (only when --log=1)
#   │   ├── *.lst             - SAS listing files (only when --lst=1)
#   │   ├── error_log.txt     - Error summary from SAStoCSV step
#   │   ├── xpt_error_log.txt - Conversion activity and error log from XPT step
#   │   └── pipeline_change_log.txt - Detailed log of all files created/changed,
#   │                                 sorted by the script that caused each change
#   └── pipeline_vars.env     - Environment variables for chaining scripts
#
# Requirements:
#   - SAS Foundation 9.4 installed at default location
#   - Python 3.x with pandas
#   - Bash shell (Git Bash on Windows)
#   - Required SAS scripts in same directory:
#     * rename_study_domains_cli20260320.sas
#     * SAStoXPTcli20260320.sas
#     * SAStoCSVcli20260320.sas
#     * variable_info_cli20260320.sas
#     * data_specs_cli20260320.sas
#     * library_info_cli20260320.sas
#   - Required Python script:
#     * metadata_to_dict_cli20260320.py
#
# Author: DAC Development Team
# Created: 2025-11-22
# Version: 1.5
#
# Version History:
#   1.0 (2025-11-22): Initial release
#   1.1 (2026-05-08): Added dataset name standardization step
#     - New step [1/7]: runs rename_study_domains_cli20260320.sas before
#       any conversion so all outputs use consistent domain-only names
#       (e.g., AE_PLACEBO -> AE).
#     - Renamed steps [1/6]-[6/6] to [2/7]-[7/7].
#     - Added DAC_SAS to output structure; INPUT_DIR is updated to DAC_SAS
#       or DAC_XPT when the rename step produces standardized files.
#   1.2 (2026-05-12): Preserve original path segments in documentation filenames
#     - Set NAME_DIR when DAC_SAS or DAC_XPT (rename) is created, matching the
#       existing DAC_SDTM behaviour, so ggg_parent and gg_parent in output
#       filenames always reflect the original input path.
#     - Fixed VARIABLE_INFO_FILE prediction to use all three path segments
#       (ggg_parent_gg_parent_g_parent) instead of only the leaf segment.
#   1.3 (2026-05-12): Added detailed pipeline change log
#     - Generates pipeline_change_log.txt in the DAC_Logs subdirectory on every run.
#     - Records which files each script created, with timestamps.
#     - Entries are sorted by the script/step that caused each change.
#     - Rename step annotates which datasets were renamed and to what names.
#     - XPT step annotates the conversion direction (SAS->XPT or XPT->SDTM).
#   1.4 (2026-05-19): Dynamic SAS output folder naming
#     - SAS standardization folder is now named DAC_<ParentFolderName> where
#       ParentFolderName is the leaf segment of the input directory
#       (e.g. if input is C:/data/rawdata, folder becomes DAC_rawdata).
#     - Both the shell script and rename_study_domains_cli20260320.sas (v1.3)
#       derive the same folder name from the input path.
#   1.5 (2026-05-19): Default output to parent of input; overwrite guard
#     - When -o is omitted, OUTPUT_DIR defaults to the parent folder of
#       INPUT_DIR (e.g. input C:/study/trial/rawdata -> output C:/study/trial).
#     - Before the pipeline begins, if any DAC_* subfolder already exists in
#       OUTPUT_DIR the user is shown the list and prompted to confirm before
#       continuing; answering anything other than y/yes aborts the run.
#
# Notes:
#   - Script exits on first error (set -e)
#   - Step 1 standardizes dataset names; subsequent steps use standardized names
#   - Automatically adjusts input path if DAC_<ParentFolderName>, DAC_XPT, or DAC_SDTM is created
#   - NAME_DIR is set whenever INPUT_DIR is redirected to preserve original path
#     segments (ggg_parent, gg_parent) in all documentation output filenames
#   - Exports VARIABLE_INFO_FILE and TRIAL_NAME for downstream processing
#
################################################################################

set -e  # Exit on error

# Display detailed help information (-H flag)
detailed_help() {
    echo "=========================================="
    echo "Detailed Help: dataPreprocessing20260320.sh"
    echo "=========================================="
    echo ""
    echo "REQUIRED FLAGS"
    echo "──────────────────────────────────────────"
    echo ""
    echo "  -i <input_directory>"
    echo "    Used by (all 7 pipeline steps):"
    echo "      [1/7] rename_study_domains_cli20260320.sas"
    echo "      [2/7] SAStoXPTcli20260320.sas"
    echo "      [3/7] SAStoCSVcli20260320.sas"
    echo "      [4/7] variable_info_cli20260320.sas"
    echo "      [5/7] data_specs_cli20260320.sas"
    echo "      [6/7] library_info_cli20260320.sas"
    echo "      [7/7] metadata_to_dict_cli20260320.py"
    echo "    Output affected: All output subdirectories and documentation files."
    echo "    Effect: Specifies the source directory containing .sas7bdat or .xpt"
    echo "      files. Every pipeline step reads datasets from this path. If step"
    echo "      [1/7] standardizes dataset names, INPUT_DIR is automatically"
    echo "      redirected to DAC_<ParentFolderName> (for .sas7bdat) or DAC_XPT (for .xpt) so"
    echo "      all subsequent steps operate on the standardized copies."
    echo ""
    echo "  -o <output_directory>"
    echo "    Used by (all 7 pipeline steps):"
    echo "      [1/7] rename_study_domains_cli20260320.sas"
    echo "      [2/7] SAStoXPTcli20260320.sas"
    echo "      [3/7] SAStoCSVcli20260320.sas"
    echo "      [4/7] variable_info_cli20260320.sas"
    echo "      [5/7] data_specs_cli20260320.sas"
    echo "      [6/7] library_info_cli20260320.sas"
    echo "      [7/7] metadata_to_dict_cli20260320.py"
    echo "    Output affected:"
    echo "      DAC_<ParentFolderName>/ - Standardized SAS datasets (step 1, if renamed)"
    echo "      DAC_XPT/          - XPT exports (step 2) or standardized XPT (step 1)"
    echo "      DAC_SDTM/         - Converted SAS datasets from XPT input (step 2)"
    echo "      DAC_CSV/          - CSV exports (step 3)"
    echo "      DAC_Documents/    - All documentation files (steps 4-7)"
    echo "      DAC_Logs/*.log    - SAS execution logs (when --log=1; steps 1-6)"
    echo "      DAC_Logs/*.lst    - SAS listing files (when --lst=1; steps 1-6)"
    echo "    Effect: Sets the root output directory. All pipeline subdirectories,"
    echo "      documentation files, and optional log/listing files are created"
    echo "      under this path. Defaults to E:/output if not provided."
    echo ""
    echo "OPTIONAL FLAGS"
    echo "──────────────────────────────────────────"
    echo ""
    echo "  --trial-name=NAME"
    echo "    Used by:"
    echo "      [7/7] metadata_to_dict_cli20260320.py"
    echo "    Output affected:"
    echo "      DAC_Documents/dictionary_*.csv"
    echo "      DAC_Documents/dictionary_*.xlsx"
    echo "    Effect: Sets the trial identifier passed to the dictionary builder."
    echo "      The value is uppercased and logged during step [7/7]. Defaults to"
    echo "      the grandparent folder name of the input directory if not provided."
    echo ""
    echo "  --format=long|condensed|wide"
    echo "    Used by:"
    echo "      [5/7] data_specs_cli20260320.sas"
    echo "    Output affected:"
    echo "      DAC_Documents/data_specs_*.xlsx"
    echo "    Effect: Controls how variables are presented in each dataset summary"
    echo "      tab of the data specifications workbook."
    echo "        long      - One row per variable value (default)."
    echo "        condensed - One row per variable; values collapsed into one cell."
    echo "        wide      - One row per dataset with variables as columns."
    echo ""
    echo "  --order=varnum|name"
    echo "    Used by:"
    echo "      [5/7] data_specs_cli20260320.sas"
    echo "    Output affected:"
    echo "      DAC_Documents/data_specs_*.xlsx"
    echo "    Effect: Determines the order of variables in dataset summary tabs."
    echo "        varnum - Order by variable position in the dataset (default)."
    echo "        name   - Order alphabetically by variable name."
    echo ""
    echo "  --index=VAR[,VAR...]"
    echo "    Used by:"
    echo "      [5/7] data_specs_cli20260320.sas"
    echo "    Output affected:"
    echo "      DAC_Documents/data_specs_*.xlsx"
    echo "    Effect: Specifies one or more index variables (e.g. USUBJID) used to"
    echo "      count distinct patients or other units of interest within each"
    echo "      dataset. Adds a 'unique index count' column to the library summary"
    echo "      tab. Default: none."
    echo ""
    echo "  --cat-threshold=N"
    echo "    Used by:"
    echo "      [5/7] data_specs_cli20260320.sas"
    echo "    Output affected:"
    echo "      DAC_Documents/data_specs_*.xlsx"
    echo "    Effect: Sets the maximum number of distinct levels a variable may"
    echo "      have before individual frequency/percentage rows are replaced with"
    echo "      distribution statistics (mean, std, min, max). Must be >= 0."
    echo "      Default: 10."
    echo ""
    echo "  --where=CLAUSE"
    echo "    Used by:"
    echo "      [5/7] data_specs_cli20260320.sas"
    echo "    Output affected:"
    echo "      DAC_Documents/data_specs_*.xlsx"
    echo "    Effect: Applies a SAS WHERE clause to the dictionary metadata to"
    echo "      subset which datasets are included in the specifications document."
    echo "      Example: --where=%str(memname in ('AE','CM','DM'))"
    echo "      Default: none (all datasets included)."
    echo ""
    echo "  --debug=0|1"
    echo "    Used by:"
    echo "      [5/7] data_specs_cli20260320.sas"
    echo "    Output affected: None (affects log verbosity and WORK library only)."
    echo "    Effect: Controls SAS NOTES and temporary dataset retention for"
    echo "      data_specs_cli20260320.sas."
    echo "        0 (default) - NOTES suppressed; temporary WORK datasets deleted."
    echo "        1           - NOTES shown in the SAS log; temporary datasets"
    echo "                      retained in WORK for inspection."
    echo ""
    echo "  --log=0|1"
    echo "    Used by (all SAS pipeline steps):"
    echo "      [1/7] rename_study_domains_cli20260320.sas  -> rename_domains.log"
    echo "      [2/7] SAStoXPTcli20260320.sas               -> sas_to_xpt.log"
    echo "      [3/7] SAStoCSVcli20260320.sas               -> sas_to_csv.log"
    echo "      [4/7] variable_info_cli20260320.sas         -> variable_info.log"
    echo "      [5/7] data_specs_cli20260320.sas            -> data_specs.log"
    echo "      [6/7] library_info_cli20260320.sas          -> library_info.log"
    echo "    Output affected:"
    echo "      <output_dir>/DAC_Logs/rename_domains.log"
    echo "      <output_dir>/DAC_Logs/sas_to_xpt.log"
    echo "      <output_dir>/DAC_Logs/sas_to_csv.log"
    echo "      <output_dir>/DAC_Logs/variable_info.log"
    echo "      <output_dir>/DAC_Logs/data_specs.log"
    echo "      <output_dir>/DAC_Logs/library_info.log"
    echo "    Effect: Controls whether SAS execution logs are written to disk."
    echo "        1 (default) - Save .log files to the output directory."
    echo "        0           - Route log output to null device; no .log files written."
    echo ""
    echo "  --lst=0|1"
    echo "    Used by (all SAS pipeline steps):"
    echo "      [1/7] rename_study_domains_cli20260320.sas  -> rename_domains.lst"
    echo "      [2/7] SAStoXPTcli20260320.sas               -> sas_to_xpt.lst"
    echo "      [3/7] SAStoCSVcli20260320.sas               -> sas_to_csv.lst"
    echo "      [4/7] variable_info_cli20260320.sas         -> variable_info.lst"
    echo "      [5/7] data_specs_cli20260320.sas            -> data_specs.lst"
    echo "      [6/7] library_info_cli20260320.sas          -> library_info.lst"
    echo "    Output affected:"
    echo "      <output_dir>/DAC_Logs/rename_domains.lst"
    echo "      <output_dir>/DAC_Logs/sas_to_xpt.lst"
    echo "      <output_dir>/DAC_Logs/sas_to_csv.lst"
    echo "      <output_dir>/DAC_Logs/variable_info.lst"
    echo "      <output_dir>/DAC_Logs/data_specs.lst"
    echo "      <output_dir>/DAC_Logs/library_info.lst"
    echo "    Effect: Controls whether SAS listing (.lst) files are written to disk."
    echo "        0 (default) - Suppress listing output; no .lst files written."
    echo "        1           - Save .lst files to the output directory."
    echo ""
    echo "Use -h or --help for a concise summary of all flags and options."
    exit "${1:-0}"
}

# Display usage information
usage() {
    echo "Usage: ./dataPreprocessing20260320.sh -i <input_dir> [-o <output_dir>] [OPTIONS]"
    echo ""
    echo "Flags:"
    echo "  -i input_dir               Path to directory containing SAS datasets (required)"
    echo "  -o output_dir              Path where outputs will be saved (optional, default: parent of input directory)"
    echo "  -H, --detailed-help        Show detailed help: which scripts use each flag,"
    echo "                             which output files are affected, and the effect"
    echo ""
    echo "Options:"
    echo "  --trial-name=NAME"
    echo "      Name used for the trial dictionary output file."
    echo "      Default: grandparent folder name of the input directory."
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
    echo "  ./dataPreprocessing20260320.sh -i 'C:/data/input'"
    echo "  ./dataPreprocessing20260320.sh -i 'C:/data/input' -o 'C:/data/output'"
    echo "  ./dataPreprocessing20260320.sh -i 'C:/data/input' -o 'C:/data/output' --trial-name=SampleStudy --format=wide"
    echo "  ./dataPreprocessing20260320.sh -i 'C:/data/input' -o 'C:/data/output' --lst=1"
    echo "  ./dataPreprocessing20260320.sh -i 'C:/data/input' -o 'C:/data/output' --log=1"
    exit "${1:-1}"
}

# Build NAME_DIR by replacing the leaf segment of ORIG_INPUT_DIR with FOLDER_NAME.
# This preserves the original ggg_parent and gg_parent path segments in all
# documentation output filenames even after INPUT_DIR is redirected to a generated
# folder.  Uses the same path separator style as ORIG_INPUT_DIR (backslash on
# pure-Windows paths, forward slash otherwise) so that downstream SAS %scan calls
# parse segments correctly.  Sets NAME_DIR in the calling scope.
build_name_dir() {
    local folder_name="$1"
    local orig_parent="${ORIG_INPUT_DIR%/*}"
    if [ "$orig_parent" = "$ORIG_INPUT_DIR" ]; then
        # No forward slash found; fall back to Windows backslash separator
        orig_parent="${ORIG_INPUT_DIR%\\*}"
    fi
    if [[ "$ORIG_INPUT_DIR" == *\\* ]] && [[ "$ORIG_INPUT_DIR" != */* ]]; then
        NAME_DIR="${orig_parent}\\${folder_name}"
    else
        NAME_DIR="${orig_parent}/${folder_name}"
    fi
    echo "      File naming path set to: $NAME_DIR"
}

# ── Change log helpers ───────────────────────────────────────────────────────

# Capture a sorted snapshot of all regular files under OUTPUT_DIR.
# The change log file itself is excluded so it never appears as a "new file"
# in any step's diff.
snapshot_output() {
    find "$OUTPUT_DIR" -type f ! -name "pipeline_change_log.txt" 2>/dev/null | sort
}

# Extract the standard domain name from a dataset filename stem.
# Standard name = last underscore-delimited token (uppercased), or the stem
# itself when it contains no underscore.
#   $1 = filename stem (no extension), any case
# Prints the standard name in uppercase.
_std_name_from_stem() {
    local stem_upper
    stem_upper=$(echo "$1" | tr '[:lower:]' '[:upper:]')
    echo "${stem_upper##*_}"
}

# Build rename-detection notes for a set of input files.
# Prints one "Renamed: <orig> -> <std>" line per file that needed renaming.
#   $1 = source directory to scan
#   $2 = file extension without leading dot (sas7bdat or xpt)
_detect_renames() {
    local src_dir="$1"
    local ext="$2"
    while IFS= read -r orig_file; do
        local orig_stem std_upper orig_upper std_lower
        orig_stem=$(basename "$orig_file" ".$ext")
        orig_upper=$(echo "$orig_stem" | tr '[:lower:]' '[:upper:]')
        std_upper=$(_std_name_from_stem "$orig_stem")
        if [ "$orig_upper" != "$std_upper" ]; then
            std_lower=$(echo "$std_upper" | tr '[:upper:]' '[:lower:]')
            printf "Renamed: %s.%s  ->  %s.%s\n" "$orig_stem" "$ext" "$std_lower" "$ext"
        fi
    done < <(find "$src_dir" -maxdepth 1 -name "*.$ext" 2>/dev/null | sort)
}

# Append one pipeline step's changes to the change log.
#   $1 = step label  (e.g. "1/7")
#   $2 = script name (e.g. "rename_study_domains_cli20260320.sas")
#   $3 = before-snapshot: sorted, newline-delimited list of file paths
#   $4 = extra detail lines (optional; newline-delimited)
log_step_changes() {
    local step_label="$1"
    local script_name="$2"
    local before_snapshot="$3"
    local extra_notes="$4"
    local timestamp
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")

    # Compute new files (present in after-snapshot but not in before-snapshot).
    # Both inputs come from snapshot_output() which already sorts; no second
    # sort is needed.  grep . filters out any stray empty lines (e.g. a blank
    # string produced when a snapshot variable is empty) so that comm does not
    # misinterpret an empty before-snapshot as a file with a blank name.
    local after_snapshot
    after_snapshot=$(snapshot_output)
    local new_files
    new_files=$(comm -23 \
        <(echo "$after_snapshot"  | grep .) \
        <(echo "$before_snapshot" | grep .) 2>/dev/null) || true

    {
        printf "\n"
        printf -- "-------------------------------------------------------\n"
        printf "Step [%s]  Script: %s\n" "$step_label" "$script_name"
        printf "Timestamp: %s\n" "$timestamp"
        printf "\n"
        printf "  Files Created:\n"
        if [ -n "$new_files" ]; then
            while IFS= read -r filepath; do
                [ -z "$filepath" ] && continue
                printf "    + %s\n" "${filepath#${OUTPUT_DIR}/}"
            done <<< "$new_files"
        else
            printf "    (none)\n"
        fi
        if [ -n "$extra_notes" ]; then
            printf "\n  Details:\n"
            while IFS= read -r note; do
                [ -z "$note" ] && continue
                printf "    %s\n" "$note"
            done <<< "$extra_notes"
        fi
    } >> "$CHANGE_LOG_FILE"
}

# Default values
INPUT_DIR=""
OUTPUT_DIR=""

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

# Parse flag arguments
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage 0
            ;;
        -H|--detailed-help)
            detailed_help 0
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
        --format=*)
            DS_FORMAT="${1#*=}"
            shift
            ;;
        --order=*)
            DS_ORDER="${1#*=}"
            shift
            ;;
        --index=*)
            DS_INDEX="${1#*=}"
            shift
            ;;
        --cat-threshold=*)
            DS_CAT_THRESHOLD="${1#*=}"
            shift
            ;;
        --where=*)
            DS_WHERE="${1#*=}"
            shift
            ;;
        --debug=*)
            DS_DEBUG="${1#*=}"
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

# Default TRIAL_NAME to grandparent folder name of INPUT_DIR if not provided;
# fall back to current date when the path is too shallow to have a grandparent.
if [ -z "$TRIAL_NAME" ]; then
    _tn_parent="${INPUT_DIR%/*}"
    if [ "$_tn_parent" = "$INPUT_DIR" ]; then
        _tn_parent="${INPUT_DIR%\\*}"
    fi
    _tn_grand="${_tn_parent%/*}"
    if [ "$_tn_grand" = "$_tn_parent" ]; then
        _tn_grand="${_tn_parent%\\*}"
    fi
    if [[ "$_tn_grand" == */* ]]; then
        TRIAL_NAME="${_tn_grand##*/}"
    else
        TRIAL_NAME="${_tn_grand##*\\}"
    fi
    if [ -z "$TRIAL_NAME" ]; then
        TRIAL_NAME=$(date +%Y%m%d)
    fi
fi

# Validate input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Default OUTPUT_DIR to the parent of INPUT_DIR when -o is not supplied.
# Handles both Unix forward slashes and Windows backslashes.
if [ -z "$OUTPUT_DIR" ]; then
    _op="${INPUT_DIR%/*}"
    [ "$_op" = "$INPUT_DIR" ] && _op="${INPUT_DIR%\\*}"
    if [ -n "$_op" ] && [ "$_op" != "$INPUT_DIR" ]; then
        OUTPUT_DIR="$_op"
    else
        OUTPUT_DIR="$INPUT_DIR"
    fi
    echo "Output directory not specified; defaulting to parent of input: $OUTPUT_DIR"
fi

# Warn and confirm before overwriting any existing DAC_* output folders.
if [ -d "$OUTPUT_DIR" ]; then
    _existing_dac=()
    while IFS= read -r -d '' _dac_dir; do
        _existing_dac+=("$_dac_dir")
    done < <(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'DAC_*' -print0 2>/dev/null)
    if [ ${#_existing_dac[@]} -gt 0 ]; then
        echo ""
        echo "Warning: The following DAC output folder(s) already exist in: $OUTPUT_DIR"
        for _dac_dir in "${_existing_dac[@]}"; do
            echo "  $(basename "$_dac_dir")"
        done
        echo ""
        read -rp "Files in these folders may be overwritten. Continue? [y/N] " _confirm
        case "$_confirm" in
            [yY]|[yY][eE][sS]) echo "" ;;
            *) echo "Aborted."; exit 1 ;;
        esac
    fi
fi

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Created output directory: $OUTPUT_DIR"
fi

# Create DAC_Logs directory for SAS log and listing files
DAC_LOGS_DIR="${OUTPUT_DIR}/DAC_Logs"
if [ ! -d "$DAC_LOGS_DIR" ]; then
    mkdir -p "$DAC_LOGS_DIR"
    echo "Created logs directory: $DAC_LOGS_DIR"
fi

# Initialize the pipeline change log
CHANGE_LOG_FILE="${DAC_LOGS_DIR}/pipeline_change_log.txt"
{
    echo "======================================================="
    echo " Pipeline Change Log"
    echo " Generated:  $(date '+%Y-%m-%d %H:%M:%S')"
    echo " Input:      ${INPUT_DIR}"
    echo " Output:     ${OUTPUT_DIR}"
    echo " Trial:      ${TRIAL_NAME:-<not set yet>}"
    echo "======================================================="
    echo " Changes are recorded below, sorted by the script that"
    echo " caused each change.  Each section shows the files"
    echo " created during that pipeline step."
    echo "======================================================="
} > "$CHANGE_LOG_FILE"

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

# Save the original input directory so file names can reflect the original path
# even after INPUT_DIR is updated to a generated folder (DAC_<ParentFolderName>, DAC_XPT, or
# DAC_SDTM).  NAME_DIR is set whenever INPUT_DIR is redirected and is passed to
# the documentation scripts so that ggg_parent and gg_parent in output filenames
# always come from the original input path.
ORIG_INPUT_DIR="$INPUT_DIR"
NAME_DIR=""  # Will be set if files are renamed/converted to a generated folder

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

# 1. Standardize dataset names (rename study-specific suffixes)
echo "[1/7] Standardizing dataset names..."
SNAP_0=$(snapshot_output)
LOG_ARG_0=$([ "$LOG_ENABLED" = "1" ] && echo "$DAC_LOGS_DIR/rename_domains.log" || echo "$NULL_DEVICE")
SAS_PRINT_0=()
if [ "$LST_ENABLED" = "1" ]; then
    SAS_PRINT_0=(-print "$DAC_LOGS_DIR/rename_domains.lst")
else
    SAS_PRINT_0=(-print "$NULL_DEVICE")
fi
"$SAS_EXE" -sysparm "$SYSPARM" -sysin "$SCRIPT_DIR/rename_study_domains_cli20260320.sas" -log "$LOG_ARG_0" "${SAS_PRINT_0[@]}"
echo "      Complete.$([ "$LOG_ENABLED" = "1" ] && echo " Log: $DAC_LOGS_DIR/rename_domains.log")"

# Check if the rename script created DAC_<ParentFolderName> (SAS files renamed) or DAC_XPT (XPT files renamed)
# and update INPUT_DIR accordingly so all subsequent steps use standardized names.
# Note: this check runs before SAStoXPTcli, so any DAC_XPT found here was created by the
# rename script (not by the XPT conversion step that follows).
INPUT_BASENAME=$(basename "$INPUT_DIR")
DAC_SAS_DIR="${OUTPUT_DIR}/DAC_${INPUT_BASENAME}"
DAC_XPT_RENAMED_DIR="${OUTPUT_DIR}/DAC_XPT"

_rename_notes=""
if [ -d "$DAC_SAS_DIR" ] && [ "$(ls -A "$DAC_SAS_DIR" 2>/dev/null)" ]; then
    echo "      Note: SAS files with non-standard names were standardized"
    echo "      Standardized SAS datasets available in: $DAC_SAS_DIR"
    echo "      Updating input path to standardized SAS files: $DAC_SAS_DIR"
    INPUT_DIR="$DAC_SAS_DIR"
    SYSPARM="${INPUT_DIR}|${OUTPUT_DIR}"
    # Build a naming path that preserves the original study context: keep the
    # original input path's parent directory but replace the leaf folder with
    # the DAC_<ParentFolderName> folder name.  This ensures documentation filenames
    # retain the original ggg_parent and gg_parent path segments even though the
    # actual input is now the generated DAC_<ParentFolderName> folder.
    build_name_dir "$(basename "$DAC_SAS_DIR")"
    # Detect which SAS datasets were renamed for the change log
    _rename_notes=$(_detect_renames "$ORIG_INPUT_DIR" "sas7bdat")
elif [ -d "$DAC_XPT_RENAMED_DIR" ] && [ "$(ls -A "$DAC_XPT_RENAMED_DIR" 2>/dev/null)" ]; then
    echo "      Note: XPT files with non-standard names were standardized"
    echo "      Standardized XPT datasets available in: $DAC_XPT_RENAMED_DIR"
    echo "      Updating input path to standardized XPT files: $DAC_XPT_RENAMED_DIR"
    INPUT_DIR="$DAC_XPT_RENAMED_DIR"
    SYSPARM="${INPUT_DIR}|${OUTPUT_DIR}"
    # Build a naming path that preserves the original study context, same
    # approach as the DAC_<ParentFolderName> case above.
    build_name_dir "$(basename "$DAC_XPT_RENAMED_DIR")"
    # Detect which XPT datasets were renamed for the change log
    _rename_notes=$(_detect_renames "$ORIG_INPUT_DIR" "xpt")
fi
log_step_changes "1/7" "rename_study_domains_cli20260320.sas" "$SNAP_0" "$_rename_notes"
SNAP_1=$(snapshot_output)

# 2. Run SAS to XPT conversion
echo "[2/7] Converting SAS datasets to XPT format..."
LOG_ARG_1=$([ "$LOG_ENABLED" = "1" ] && echo "$DAC_LOGS_DIR/sas_to_xpt.log" || echo "$NULL_DEVICE")
SAS_PRINT_1=()
if [ "$LST_ENABLED" = "1" ]; then
    SAS_PRINT_1=(-print "$DAC_LOGS_DIR/sas_to_xpt.lst")
else
    SAS_PRINT_1=(-print "$NULL_DEVICE")
fi
"$SAS_EXE" -sysparm "$SYSPARM" -sysin "$SCRIPT_DIR/SAStoXPTcli20260320.sas" -log "$LOG_ARG_1" "${SAS_PRINT_1[@]}"
echo "      Complete.$([ "$LOG_ENABLED" = "1" ] && echo " Log: $DAC_LOGS_DIR/sas_to_xpt.log")"

# Check if XPT files were converted to SAS7BDAT (DAC_SDTM folder created)
DAC_SDTM_DIR="${OUTPUT_DIR}/DAC_SDTM"
_xpt_notes=""
if [ -d "$DAC_SDTM_DIR" ] && [ "$(ls -A "$DAC_SDTM_DIR" 2>/dev/null)" ]; then
    echo "      Note: XPT files detected and converted to SAS7BDAT format"
    echo "      SDTM datasets available in: $DAC_SDTM_DIR"
    echo "      Updating input path to converted SAS files: $DAC_SDTM_DIR"
    INPUT_DIR="$DAC_SDTM_DIR"
    SYSPARM="${INPUT_DIR}|${OUTPUT_DIR}"
    # Build a naming path that preserves the original study context: keep the
    # original input path's parent directory but replace the leaf folder with
    # the DAC_SDTM folder name.  This means generated file names will still
    # reflect the original data location while correctly showing the conversion
    # step in the innermost folder segment.
    build_name_dir "$(basename "$DAC_SDTM_DIR")"
    _xpt_notes="Conversion direction: XPT -> SAS7BDAT (input contained XPT files; SDTM datasets written to DAC_SDTM)"
else
    _xpt_notes="Conversion direction: SAS7BDAT -> XPT (XPT exports written to DAC_XPT)"
fi
log_step_changes "2/7" "SAStoXPTcli20260320.sas" "$SNAP_1" "$_xpt_notes"
SNAP_2=$(snapshot_output)

# 3. Run SAS to CSV conversion
echo "[3/7] Converting SAS datasets to CSV..."
LOG_ARG_2=$([ "$LOG_ENABLED" = "1" ] && echo "$DAC_LOGS_DIR/sas_to_csv.log" || echo "$NULL_DEVICE")
SAS_PRINT_2=()
if [ "$LST_ENABLED" = "1" ]; then
    SAS_PRINT_2=(-print "$DAC_LOGS_DIR/sas_to_csv.lst")
else
    SAS_PRINT_2=(-print "$NULL_DEVICE")
fi
"$SAS_EXE" -sysparm "$SYSPARM" -sysin "$SCRIPT_DIR/SAStoCSVcli20260320.sas" -log "$LOG_ARG_2" "${SAS_PRINT_2[@]}"
echo "      Complete.$([ "$LOG_ENABLED" = "1" ] && echo " Log: $DAC_LOGS_DIR/sas_to_csv.log")"
log_step_changes "3/7" "SAStoCSVcli20260320.sas" "$SNAP_2"
SNAP_3=$(snapshot_output)

# 4. Generate variable information and capture output file location
echo "[4/7] Generating variable information document..."
LOG_ARG_3=$([ "$LOG_ENABLED" = "1" ] && echo "$DAC_LOGS_DIR/variable_info.log" || echo "$NULL_DEVICE")
SAS_PRINT_3=()
if [ "$LST_ENABLED" = "1" ]; then
    SAS_PRINT_3=(-print "$DAC_LOGS_DIR/variable_info.lst")
else
    SAS_PRINT_3=(-print "$NULL_DEVICE")
fi
VAR_INFO_SYSPARM="$SYSPARM"
if [ -n "$NAME_DIR" ]; then
    VAR_INFO_SYSPARM="${INPUT_DIR}|${OUTPUT_DIR}|name_dir=${NAME_DIR}"
fi
"$SAS_EXE" -sysparm "$VAR_INFO_SYSPARM" -sysin "$SCRIPT_DIR/variable_info_cli20260320.sas" -log "$LOG_ARG_3" "${SAS_PRINT_3[@]}"
echo "      Complete.$([ "$LOG_ENABLED" = "1" ] && echo " Log: $DAC_LOGS_DIR/variable_info.log")"
log_step_changes "4/7" "variable_info_cli20260320.sas" "$SNAP_3"
SNAP_4=$(snapshot_output)

# 5. Generate data specifications
echo "[5/7] Generating data specifications document..."
DATA_SPECS_SYSPARM="${INPUT_DIR}|${OUTPUT_DIR}|index=${DS_INDEX}|cat_threshold=${DS_CAT_THRESHOLD}|format=${DS_FORMAT}|order=${DS_ORDER}|where=${DS_WHERE}|debug=${DS_DEBUG}"
if [ -n "$NAME_DIR" ]; then
    DATA_SPECS_SYSPARM="${DATA_SPECS_SYSPARM}|name_dir=${NAME_DIR}"
fi
LOG_ARG_4=$([ "$LOG_ENABLED" = "1" ] && echo "$DAC_LOGS_DIR/data_specs.log" || echo "$NULL_DEVICE")
SAS_PRINT_4=()
if [ "$LST_ENABLED" = "1" ]; then
    SAS_PRINT_4=(-print "$DAC_LOGS_DIR/data_specs.lst")
else
    SAS_PRINT_4=(-print "$NULL_DEVICE")
fi
"$SAS_EXE" -sysparm "$DATA_SPECS_SYSPARM" -sysin "$SCRIPT_DIR/data_specs_cli20260320.sas" -log "$LOG_ARG_4" "${SAS_PRINT_4[@]}"
echo "      Complete.$([ "$LOG_ENABLED" = "1" ] && echo " Log: $DAC_LOGS_DIR/data_specs.log")"
log_step_changes "5/7" "data_specs_cli20260320.sas" "$SNAP_4"
SNAP_5=$(snapshot_output)

# 6. Generate library information
echo "[6/7] Generating library information document..."
LOG_ARG_5=$([ "$LOG_ENABLED" = "1" ] && echo "$DAC_LOGS_DIR/library_info.log" || echo "$NULL_DEVICE")
SAS_PRINT_5=()
if [ "$LST_ENABLED" = "1" ]; then
    SAS_PRINT_5=(-print "$DAC_LOGS_DIR/library_info.lst")
else
    SAS_PRINT_5=(-print "$NULL_DEVICE")
fi
LIB_INFO_SYSPARM="$SYSPARM"
if [ -n "$NAME_DIR" ]; then
    LIB_INFO_SYSPARM="${INPUT_DIR}|${OUTPUT_DIR}|name_dir=${NAME_DIR}"
fi
"$SAS_EXE" -sysparm "$LIB_INFO_SYSPARM" -sysin "$SCRIPT_DIR/library_info_cli20260320.sas" -log "$LOG_ARG_5" "${SAS_PRINT_5[@]}"
echo "      Complete.$([ "$LOG_ENABLED" = "1" ] && echo " Log: $DAC_LOGS_DIR/library_info.log")"
log_step_changes "6/7" "library_info_cli20260320.sas" "$SNAP_5"
SNAP_6=$(snapshot_output)

# Construct the expected variable info file path using all three naming path
# segments (ggg_parent, gg_parent, g_parent) and today's date stamp, matching
# the three-part naming convention used in variable_info_cli20260320.sas.
# When any generated folder is active, NAME_DIR reflects the original study
# path with the generated folder as the leaf, so all three segments are correct.
# Falls back to a glob search in DAC_Documents if the constructed path does
# not exist (e.g. timezone or clock-skew edge case).
NAMING_DIR="${NAME_DIR:-$INPUT_DIR}"
# Normalize path separators to forward slash and split into non-empty segments
normalized_path="${NAMING_DIR//\\//}"
IFS='/' read -ra path_segments <<< "$normalized_path"
clean_segments=()
for segment in "${path_segments[@]}"; do [ -n "$segment" ] && clean_segments+=("$segment"); done
segment_count="${#clean_segments[@]}"
[ "$segment_count" -ge 1 ] && leaf_segment_raw="${clean_segments[$(( segment_count-1 ))]}" || leaf_segment_raw=""
[ "$segment_count" -ge 2 ] && parent_segment_raw="${clean_segments[$(( segment_count-2 ))]}" || parent_segment_raw=""
[ "$segment_count" -ge 3 ] && grandparent_segment_raw="${clean_segments[$(( segment_count-3 ))]}" || grandparent_segment_raw=""
# G_PARENT = last/leaf segment; GG_PARENT = one level up; GGG_PARENT = two levels up.
# This matches the ggg_parent/gg_parent/g_parent convention used in the SAS scripts.
G_PARENT=$(echo "$leaf_segment_raw" | tr -cd '[:alnum:]')
GG_PARENT=$(echo "$parent_segment_raw" | tr -cd '[:alnum:]')
GGG_PARENT=$(echo "$grandparent_segment_raw" | tr -cd '[:alnum:]')
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

# 7. Generate trial dictionary from variable info
echo "[7/7] Generating trial dictionary from variable information..."

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
    "$PYTHON_CMD" "$SCRIPT_DIR/metadata_to_dict_cli20260320.py" "$VARIABLE_INFO_FILE" "$OUTPUT_DIR" "$TRIAL_NAME" "${NAME_DIR:-$INPUT_DIR}"
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
log_step_changes "7/7" "metadata_to_dict_cli20260320.py" "$SNAP_6"

echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Output files located in:"
echo "  - Documents: $OUTPUT_DIR/DAC_Documents"
echo "  - CSV:       $OUTPUT_DIR/DAC_CSV"
echo "  - XPT:       $OUTPUT_DIR/DAC_XPT"
if [ -d "$DAC_SAS_DIR" ]; then
    echo "  - SAS:       $DAC_SAS_DIR"
fi
if [ -d "$DAC_SDTM_DIR" ]; then
    echo "  - SDTM:      $DAC_SDTM_DIR"
fi
echo "  - Logs:      $DAC_LOGS_DIR"
echo "  - Change Log: $CHANGE_LOG_FILE"
echo "=========================================="

# Write the change log footer
{
    printf "\n"
    printf -- "=======================================================\n"
    printf " Pipeline Complete\n"
    printf " End Time: %s\n" "$(date '+%Y-%m-%d %H:%M:%S')"
    printf "=======================================================\n"
} >> "$CHANGE_LOG_FILE"

echo "Change log written to: $CHANGE_LOG_FILE"

exit 0

