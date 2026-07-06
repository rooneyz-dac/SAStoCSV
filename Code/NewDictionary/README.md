# Create Dictionary — CLI Script

Generates data dictionaries (CSV and Excel) from a folder of SAS datasets (`.sas7bdat`).

This is a focused subset of the full `dataPreprocessing` pipeline. It runs only the two steps needed to produce dictionary output:

1. **`variable_info_cli20260320.sas`** — builds a multi-sheet Excel workbook with one sheet per dataset listing variable number, name, type, length, position, and label.
2. **`metadata_to_dict_cli20260320.py`** — combines all sheets of the variable info workbook into a single trial dictionary exported as both CSV and Excel.

---

## Contents

| File | Role |
|------|------|
| `createDictionary.sh` | **Entry point** — orchestrates both steps |
| `variable_info_cli20260320.sas` | Step 1 — variable-level information workbook |
| `metadata_to_dict_cli20260320.py` | Step 2 — build combined trial dictionary |

---

## Requirements

- **SAS Foundation 9.4** installed at `C:/Program Files/SASHome/SASFoundation/9.4/sas.exe`
- **Python 3.8+** with `pandas` and `openpyxl`
- **Bash shell** — Git Bash on Windows or any POSIX-compatible shell
- All three files must be kept in the **same directory**

---

## Quick Start

```bash
# Minimal — output defaults to E:/output/dict
./createDictionary.sh -i "C:/studies/SampleStudy/rawdata"

# Specify both input and output directories
./createDictionary.sh -i "C:/studies/SampleStudy/rawdata" -o "E:/output/dict"

# Windows-style backslash paths are also accepted
./createDictionary.sh -i "C:\studies\SampleStudy\rawdata" -o "E:\output\dict"

# Provide a custom trial name for the dictionary filename
./createDictionary.sh -i "C:/studies/SampleStudy/rawdata" -o "E:/output/dict" --trial-name=SampleStudy

# Suppress the SAS log file
./createDictionary.sh -i "C:/studies/SampleStudy/rawdata" -o "E:/output/dict" --log=0

# Save the SAS listing file alongside the log
./createDictionary.sh -i "C:/studies/SampleStudy/rawdata" -o "E:/output/dict" --lst=1
```

---

## Flags and Options

### Required

| Flag | Description |
|------|-------------|
| `-i <input_dir>` | Path to the directory containing `.sas7bdat` files |

### Optional

| Flag | Default | Description |
|------|---------|-------------|
| `-o <output_dir>` | `E:\output\dict` | Root directory where `DAC_Documents` and `DAC_Logs` are created |
| `-h`, `--help` | — | Show concise usage summary |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--trial-name=NAME` | Parent folder name of input directory | Trial identifier passed to the dictionary builder (uppercased automatically) |
| `--log=0\|1` | `1` | `1` — save the SAS `.log` file to `DAC_Logs/`; `0` — suppress it |
| `--lst=0\|1` | `0` | `1` — save the SAS `.lst` listing file to `DAC_Logs/`; `0` — suppress it |

---

## Output Structure

```
output_directory/
├── DAC_Documents/
│   ├── variable_info_<GGG>_<GG>_<G>_<YYYYMMDD>.xlsx
│   ├── dictionary_<GGG>_<GG>_<G>_<YYYYMMDD>.csv
│   └── dictionary_<GGG>_<GG>_<G>_<YYYYMMDD>.xlsx
└── DAC_Logs/
    ├── variable_info.log   (when --log=1)
    └── variable_info.lst   (when --lst=1)
```

> **Filename components** — `<GGG>`, `<GG>`, and `<G>` are the third-to-last, second-to-last, and last segments of the input directory path (alphanumeric characters only). For example, an input path of `C:/studies/SampleStudy/rawdata` produces `C_studies_SampleStudy` as the three-part prefix and `rawdata` as the `<G>` segment.

---

## Overwrite Warning

If `dictionary_*.csv` or `dictionary_*.xlsx` files already exist in `DAC_Documents`, the script will list them and prompt for confirmation before proceeding. Entering anything other than `y` or `yes` aborts the run without modifying any files.

---

## Notes

- Both forward slashes (`/`) and backslashes (`\`) are accepted in all path arguments; the script normalizes them automatically.
- The script exits immediately on the first error (`set -e`).
- The SAS executable path is hardcoded to `C:/Program Files/SASHome/SASFoundation/9.4/sas.exe`. Update the `SAS_EXE` variable in `createDictionary.sh` if SAS is installed elsewhere.
