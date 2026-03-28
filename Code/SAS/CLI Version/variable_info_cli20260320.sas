/*------------------------------------------------------------------*
| PROGRAM NAME : variable_info_cli20260320.sas
| SHORT DESC   : Creates a variable information document with CLI
|                parameter support
*------------------------------------------------------------------*
| CREATED BY   : Modified for CLI usage
| DATE CREATED : 2025-01-22
*------------------------------------------------------------------*
| VERSION UPDATES:
| 2025-01-22: Initial CLI release
|   - Added SYSPARM parsing for input and output directories
|   - Added automatic creation of DAC_Documents output subfolder
|   - Added path validation for input and output directories
|   - Output file name includes library name and date stamp
|   - Each dataset is placed on its own sheet in the Excel workbook
| 2026-03-28: Column formatting update
|   - Dropped FORMAT and INFORMAT columns from output to match the
|     column selection logic in metadata_to_dict_cli20260320.py
|   - Enforced column order: NUM, VARIABLE, TYPE, LEN, LABEL
|   - Switched from PROC PRINT to PROC REPORT to enable per-column
|     width formatting in the Excel workbook output
*------------------------------------------------------------------*
| PURPOSE
| Creates a variable-level information report for all datasets in a
| designated directory. For each dataset the report lists variable
| number, name, type, length, and label. Results are exported to a
| multi-sheet Excel workbook (.xlsx) in the DAC_Documents subfolder
| of the output directory, with one sheet per dataset.
| Column selection and ordering mirrors the logic in
| metadata_to_dict_cli20260320.py (NUM, VARIABLE, TYPE, LEN, LABEL).
|
| 1.0: REQUIRED SYSPARM PARAMETERS (pipe-delimited)
| INPUT_DIRECTORY  = Path to the folder containing SAS datasets
|                    (.sas7bdat) to be summarized
| OUTPUT_DIRECTORY = Path to the folder where the DAC_Documents
|                    subfolder and output file will be created.
|                    Must exist prior to execution.
|
| 2.0: OPTIONAL PARAMETERS
| None. All parameters are supplied via SYSPARM.
|
| USAGE:
|   sas -sysparm "input_directory|output_directory"
|       -sysin variable_info_cli20260320.sas
|
| OUTPUT STRUCTURE:
|   output_directory\
|   └── DAC_Documents\
|       └── variable_info_<GGG_PARENT>_<GG_PARENT>_<G_PARENT>_<YYYYMMDD>.xlsx
|           (one Excel sheet per dataset)
*------------------------------------------------------------------*
| OPERATING SYSTEM COMPATIBILITY
| SAS v9.4 or Higher: Yes
*------------------------------------------------------------------*
| MACRO CALL
|
| Not applicable. Script is invoked directly via the SAS command
| line using -sysin and -sysparm.
*------------------------------------------------------------------*
| EXAMPLES
|
| Basic usage:
|   sas -sysparm "C:\data\input|C:\data\output"
|       -sysin variable_info_cli20260320.sas
|
| Using a study-specific directory:
|   sas -sysparm "C:\studies\trial1\sas|C:\studies\trial1\output"
|       -sysin variable_info_cli20260320.sas
*------------------------------------------------------------------*/

%macro variable_info_cli;
    /**Parse SYSPARM for input and output paths**/
    %local sysparm_value indir outdir pipe_pos;
    %let sysparm_value = &sysparm;
    %put DEBUG: Raw SYSPARM value = &sysparm_value;

    %let pipe_pos = %sysfunc(findc(&sysparm_value, |));
    %put DEBUG: Pipe position = &pipe_pos;

    %if &pipe_pos > 0 %then %do;
        %let indir = %substr(&sysparm_value, 1, %eval(&pipe_pos - 1));
        %let outdir = %substr(&sysparm_value, %eval(&pipe_pos + 1));
    %end;
    %else %do;
        %put ERROR: Invalid SYSPARM format. Expected: input_directory|output_directory;
        %abort;
    %end;

    %put DEBUG: Input Directory = &indir;
    %put DEBUG: Output Directory = &outdir;

    /**Validate directories exist**/
    %local indir_exist outdir_exist;
    %let indir_exist = %sysfunc(fileexist(&indir));
    %let outdir_exist = %sysfunc(fileexist(&outdir));

    %put DEBUG: Input directory exists = &indir_exist;
    %put DEBUG: Output directory exists = &outdir_exist;

    %if &indir_exist = 0 %then %do;
        %put ERROR: Input directory does not exist: &indir;
        %abort;
    %end;

    %if &outdir_exist = 0 %then %do;
        %put ERROR: Output directory does not exist: &outdir;
        %abort;
    %end;

    /**Create DAC_Documents folder if it doesn't exist**/
    %local doc_dir doc_exist rc;
    %let doc_dir = &outdir\DAC_Documents;
    %let doc_exist = %sysfunc(fileexist(&doc_dir));

    %if &doc_exist = 0 %then %do;
        %put DEBUG: Creating directory &doc_dir;
        %let rc = %sysfunc(dcreate(DAC_Documents, &outdir));
        %if &rc = %then %do;
            %put ERROR: Failed to create directory &doc_dir;
            %abort;
        %end;
        %put DEBUG: Successfully created &doc_dir;
    %end;
    %else %do;
        %put DEBUG: Directory &doc_dir already exists;
    %end;

    /**Assign library to input directory**/
    libname INLIB "&indir";

    %if %sysfunc(libref(INLIB)) ^= 0 %then %do;
        %put ERROR: Failed to assign library to &indir;
        %abort;
    %end;

    /**Set output file path**/
    %local out_file g_parent gg_parent ggg_parent;
    %let g_parent = %scan(&indir, -1, \);
    %let gg_parent = %scan(&indir, -2, \);
    %let ggg_parent = %scan(&indir, -3, \);
    %let out_file = &doc_dir\variable_info_%sysfunc(compress(&ggg_parent,,ka))_%sysfunc(compress(&gg_parent,,ka))_%sysfunc(compress(&g_parent,,ka))_%sysfunc(today(),yymmddn8.).xlsx;
    %put DEBUG: Output file = &out_file;

    /**Get variable information**/
    ods output variables=allvarout;

    proc contents data=INLIB._all_ memtype=data;
    run;

    /**Sort data by dataset and variable number**/
    proc sort data=allvarout;
        by member num;
    run;

    /**Create Excel output file with separate sheets per dataset**/
    /**Column selection and ordering mirrors metadata_to_dict_cli20260320.py:**/
    /**  desired_columns = [NUM, VARIABLE, TYPE, LEN, LABEL]                 **/
    /**  FORMAT and INFORMAT are excluded to match Python output logic.       **/
    options nobyline;
    ods excel file="&out_file"
        options(sheet_name="#BYVAL(member)" embedded_titles='yes');

    proc report data=allvarout nowindows headline;
        columns num variable type len label;
        define num      / display 'Variable Number'
                          style(header)=[just=center]
                          style(column)=[cellwidth=0.9in just=center];
        define variable / display 'Variable Name'
                          style(column)=[cellwidth=1.5in];
        define type     / display 'Type'
                          style(header)=[just=center]
                          style(column)=[cellwidth=0.75in just=center];
        define len      / display 'Length'
                          style(header)=[just=center]
                          style(column)=[cellwidth=0.75in just=center];
        define label    / display 'Variable Label'
                          style(column)=[cellwidth=3in];
        by member;
        title "Variables in #BYVAL(member) Dataset";
    run;

    title;

    ods excel close;

    /**Clean up**/
    proc datasets library=work nolist nodetails;
        delete allvarout;
    quit;

    libname INLIB clear;

    %put NOTE: Variable information document created: &out_file;

%mend variable_info_cli;

%variable_info_cli;
