# SAS Data Preprocessing Pipeline — CLI Version

Automated pipeline that standardizes SAS dataset names, converts between SAS formats, exports data to CSV, and generates a suite of documentation files — all driven by a single Bash entry-point script.

---

## Contents

| File | Role |
|------|------|
| `dataPreprocessing20260320.sh` | **Entry point** — orchestrates all 7 steps |
| `rename_study_domains_cli20260320.sas` | Step 1 — standardize dataset names |
| `SAStoXPTcli20260320.sas` | Step 2 — convert SAS ↔ XPT |
| `SAStoCSVcli20260320.sas` | Step 3 — export SAS datasets to CSV |
| `variable_info_cli20260320.sas` | Step 4 — variable-level information workbook |
| `data_specs_cli20260320.sas` | Step 5 — data specifications workbook |
| `library_info_cli20260320.sas` | Step 6 — library summary workbook |
| `metadata_to_dict_cli20260320.py` | Step 7 — build combined trial dictionary |

---

## Requirements

- **SAS Foundation 9.4** installed at its default location
- **Python 3.8+** with `pandas` and `openpyxl`
- **Bash shell** — Git Bash on Windows or any POSIX-compatible shell
- All 7 script files must be kept in the **same directory**

---

## Quick Start

```bash
# Minimal — output defaults to the grandparent of the input directory
# (e.g. -i "C:/studies/SampleStudy/rawdata" → output goes to "C:/studies/SampleStudy")
./dataPreprocessing20260320.sh -i "C:/studies/SampleStudy/rawdata"

# Specify input and output directories explicitly
./dataPreprocessing20260320.sh -i "C:/studies/SampleStudy/rawdata" -o "C:/data/output"

# Full example with optional flags
./dataPreprocessing20260320.sh \
  -i "C:/studies/SampleStudy/rawdata" \
  -o "C:/data/output" \
  --trial-name=SampleStudy \
  --format=wide \
  --index=USUBJID \
  --log=1
```

---

## Flags and Options

### Required

| Flag | Description |
|------|-------------|
| `-i <input_dir>` | Path to the directory containing `.sas7bdat` or `.xpt` files |

### Optional

| Flag | Default | Description |
|------|---------|-------------|
| `-o <output_dir>` | Grandparent of input directory | Root directory where all output subdirectories are created |
| `-H`, `--detailed-help` | — | Show per-flag detail: which scripts use it, which files it affects |
| `-h`, `--help` | — | Show concise usage summary |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--trial-name=NAME` | Grandparent folder name of input directory | Trial identifier used in dictionary output filenames (uppercased automatically) |
| `--format=long\|condensed\|wide` | `long` | Layout of variables in each dataset summary tab of the data specs workbook |
| `--order=varnum\|name` | `varnum` | Variable ordering in dataset summary tabs — by position (`varnum`) or alphabetically (`name`) |
| `--index=VAR[,VAR...]` | none | One or more index variables (e.g. `USUBJID`) used to count distinct subjects per dataset |
| `--cat-threshold=N` | `10` | Maximum distinct levels before per-level frequencies are replaced with distribution statistics |
| `--where=CLAUSE` | none | SAS WHERE clause to subset which datasets are included in the specifications document. Example: `--where=%str(memname in ('AE','CM','DM'))` |
| `--debug=0\|1` | `0` | `1` — show SAS NOTES in log and retain temporary WORK datasets for inspection |
| `--log=0\|1` | `1` | `0` — suppress SAS `.log` files; `1` — save them to `DAC_Logs/` |
| `--lst=0\|1` | `0` | `1` — save SAS listing (`.lst`) files to `DAC_Logs/` |

---

## Output Structure

```
output_directory/
├── DAC_<ParentFolderName>/         - Standardized SAS datasets (step 1, only if any dataset was renamed)
├── DAC_XPT/                    - XPT exports from .sas7bdat input (step 2)
│                                 or standardized XPT copies (step 1)
├── DAC_SDTM/                   - SAS7BDAT datasets converted from .xpt input (step 2)
├── DAC_CSV/                    - CSV exports of all SAS datasets (step 3)
├── DAC_Documents/              - All documentation files (steps 4–7)
│   ├── variable_info_<GGG>_<GG>_<G>_<YYYYMMDD>.xlsx
│   ├── data_specs_<GGG>_<GG>_<G>_<YYYYMMDD>.xlsx
│   ├── library_info_<GGG>_<GG>_<G>_<YYYYMMDD>.xlsx
│   ├── dictionary_<GGG>_<GG>_<G>_<YYYYMMDD>.csv
│   └── dictionary_<GGG>_<GG>_<G>_<YYYYMMDD>.xlsx
├── DAC_Logs/                   - SAS execution logs (created when --log=1 or --lst=1)
│   ├── rename_domains.log/.lst
│   ├── sas_to_xpt.log/.lst
│   ├── sas_to_csv.log/.lst
│   ├── variable_info.log/.lst
│   ├── data_specs.log/.lst
│   └── library_info.log/.lst
├── pipeline_vars.env           - Environment variables exported for downstream use
└── pipeline_change_log.txt     - Record of every file created by each pipeline step
```

> **Filename components** — `<GGG>`, `<GG>`, and `<G>` are the third-to-last, second-to-last, and last segments of the input directory path (alphanumeric characters only). For example, an input path of `C:/studies/SampleStudy/sas` produces `C_studies_SampleStudy` as the three-part prefix.

---

## Pipeline Steps

### Step 1 — `rename_study_domains_cli20260320.sas`

Scans the input directory and standardizes dataset names by extracting the portion **after the last underscore** (e.g., `BERKELEY_AE` → `AE`, `AE_PLACEBO` → `PLACEBO`). When any dataset requires renaming, **all** datasets of that type are copied to a new subfolder with standardized names:

- `.sas7bdat` files → `DAC_<ParentFolderName>/` (where `<ParentFolderName>` is the leaf segment of the input directory)
- `.xpt` files → `DAC_XPT/`

If no datasets need renaming, no new folder is created and the original input directory is used for all subsequent steps.

### Step 2 — `SAStoXPTcli20260320.sas`

Bidirectional format conversion:

- `.sas7bdat` input → XPT output in `DAC_XPT/`
- `.xpt` input → SAS7BDAT output in `DAC_SDTM/`

Both conversion types may occur in the same run. Dataset names longer than 8 characters are automatically truncated to comply with the XPT format specification (first 4 + last 4 characters; e.g., `berkeley_pbo_ae` → `berkl_ae`).

### Step 3 — `SAStoCSVcli20260320.sas`

Exports all `.sas7bdat` files in the (potentially redirected) input directory to CSV format, placing the output in `DAC_CSV/`.

### Step 4 — `variable_info_cli20260320.sas`

Generates a multi-sheet Excel workbook with one sheet per dataset. Each sheet lists the variables in that dataset with columns: **NUM, VARIABLE, TYPE, LEN, POS, LABEL**.

Output: `DAC_Documents/variable_info_<GGG>_<GG>_<G>_<YYYYMMDD>.xlsx`

### Step 5 — `data_specs_cli20260320.sas`

Generates a data specifications workbook containing:

1. **Library summary** — dataset names, row counts, variable counts, and unique index counts
2. **Cross-dataset variable summary** — variables that appear in multiple datasets with any label differences
3. **Per-dataset tabs** — variable-level specs (label, format, value frequencies or distribution statistics)

The `--format`, `--order`, `--index`, `--cat-threshold`, `--where`, and `--debug` options all apply to this step.

Output: `DAC_Documents/data_specs_<GGG>_<GG>_<G>_<YYYYMMDD>.xlsx`

### Step 6 — `library_info_cli20260320.sas`

Generates a library summary workbook listing each dataset's name, label, observation count, variable count, creation date, and last modification date.

Output: `DAC_Documents/library_info_<GGG>_<GG>_<G>_<YYYYMMDD>.xlsx`

### Step 7 — `metadata_to_dict_cli20260320.py`

Reads every sheet of the `variable_info` workbook produced in step 4 (header on row 3), normalizes column names, tags each row with a `SOURCE_TAB` column identifying its origin sheet, concatenates all sheets, and writes a combined trial dictionary.

Output: `DAC_Documents/dictionary_<GGG>_<GG>_<G>_<YYYYMMDD>.csv` and `.xlsx`

---

## Notes

- The pipeline exits immediately on the first error (`set -e`).
- If step 1 standardizes dataset names, `INPUT_DIR` is automatically redirected to `DAC_<ParentFolderName>` or `DAC_XPT` for all subsequent steps; the original path segments are preserved in documentation filenames via `NAME_DIR`.
- When `-o` is omitted, the output directory defaults to the **grandparent** of the input directory (e.g. `-i "C:/studies/SampleStudy/rawdata"` → output in `C:/studies/SampleStudy`).
- If any `DAC_*` subfolder already exists in the output directory, the pipeline lists them and prompts for confirmation before proceeding. Entering anything other than `y` or `yes` aborts the run without modifying any files.
- `pipeline_vars.env` is written after every run and exports `VARIABLE_INFO_FILE` and `TRIAL_NAME` for use by downstream scripts.
- `pipeline_change_log.txt` records every file created by each step, sorted by step, with timestamps.
