/*------------------------------------------------------------------*
| PROGRAM NAME : library_info_cli20260320.sas
| SHORT DESC   : Creates a library information document with CLI
|                parameter support
*------------------------------------------------------------------*
| CREATED BY   : Modified for CLI usage
| DATE CREATED : 2025-01-21
*------------------------------------------------------------------*
| VERSION UPDATES:
| 2025-01-21: Initial CLI release
|   - Added SYSPARM parsing for input and output directories
|   - Added automatic creation of DAC_Documents output subfolder
|   - Added path validation for input and output directories
|   - Output file name includes library name and date stamp
*------------------------------------------------------------------*
| PURPOSE
| Creates a library information summary for all datasets found in a
| designated directory. The output lists each dataset's name, label,
| observation count, variable count, creation date, and last
| modification date. Results are exported to an Excel workbook
| (.xlsx) in the DAC_Documents subfolder of the output directory.
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
|       -sysin library_info_cli20260320.sas
|
| OUTPUT STRUCTURE:
|   output_directory\
|   └── DAC_Documents\
|       └── library_info_<libname>_<YYYYMMDD>.xlsx
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
|       -sysin library_info_cli20260320.sas
|
| Using a study-specific directory:
|   sas -sysparm "C:\studies\trial1\sas|C:\studies\trial1\output"
|       -sysin library_info_cli20260320.sas
*------------------------------------------------------------------*/

%macro library_info_cli;
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
    %local out_file libname_text;
    %let libname_text = %scan(&indir, -1, \);
    %if %sysevalf(%superq(libname_text)=,boolean) %then %let libname_text = INPUT;
    %let out_file = &doc_dir\library_info_%sysfunc(compress(&libname_text,,'ka'))_%sysfunc(today(),yymmddn8.).xlsx;
    %put DEBUG: Output file = &out_file;

    /**Create table of SQL library data**/
    proc sql;
        create table LIBINFO as
        select * from dictionary.tables
        where libname="INLIB";
    quit;

    /**Count number of datasets**/
    %local ds_count;
    proc sql noprint;
        select count(memname) into :ds_count trimmed
        from dictionary.tables
        where libname="INLIB";
    quit;

    %put NOTE: Number of datasets in library: &ds_count;

    /**Create Excel output file**/
    ods excel file="&out_file"
        options(sheet_name="Library Info" frozen_headers='yes');

    /**Print summary**/
    title1 "Library Information for: &indir";
    title2 "Total Datasets: &ds_count";
    title3 "Generated: %sysfunc(today(),worddate.)";

    proc print data=LIBINFO label;
        var memname memlabel nobs nvar crdate modate;
        label memname='Dataset Name'
              memlabel='Dataset Label'
              nobs='Observations'
              nvar='Variables'
              crdate='Created Date'
              modate='Modified Date';
    run;

    title;

    ods excel close;

    /**Clean up**/
    proc datasets library=work nolist nodetails;
        delete LIBINFO;
    quit;

    libname INLIB clear;

    %put NOTE: Library information document created: &out_file;

%mend library_info_cli;

%library_info_cli;
