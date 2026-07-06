/*------------------------------------------------------------------*
| PROGRAM NAME : SAStoCSVcli20260320.sas
| SHORT DESC   : Batch converts all SAS datasets in a directory to
|                CSV format via command-line interface
*------------------------------------------------------------------*
| CREATED BY   : DAC Development Team
| DATE CREATED : 2025-11-21
*------------------------------------------------------------------*
| PURPOSE
| Exports all SAS datasets (.sas7bdat) found in a specified input
| directory to CSV format. Output CSV files are placed in a
| DAC_CSV subfolder within the specified output directory. An error
| log file (error_log.txt) is written to the DAC_Logs subdirectory upon
| completion summarizing any SAS errors encountered.
|
| 1.0: REQUIRED SYSPARM PARAMETERS (pipe-delimited)
| INPUT_DIRECTORY  = Path to the folder containing .sas7bdat files
|                    to be converted
| OUTPUT_DIRECTORY = Path to the folder where the DAC_CSV subfolder
|                    and error log will be created
|
| 2.0: OPTIONAL PARAMETERS
| None. All parameters are supplied via SYSPARM.
|
| USAGE:
|   sas -sysparm "input_directory|output_directory"
|       -sysin SAStoCSVcli20260320.sas
|
| OUTPUT STRUCTURE:
|   output_directory\
|   ├── DAC_CSV\              - CSV exports of all SAS datasets
|   └── DAC_Logs\
|       └── error_log.txt     - Error summary from the SAS run
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
|       -sysin SAStoCSVcli20260320.sas
|
| Using a study-specific output folder:
|   sas -sysparm "C:\studies\trial1\sas|C:\studies\trial1\output"
|       -sysin SAStoCSVcli20260320.sas
*------------------------------------------------------------------*/

/* 1) Parse SYSPARM to get input and output directories */
%let params = %sysfunc(dequote(&SYSPARM));

%if %length(&params) = 0 %then %do;
    %put ERROR: SYSPARM not provided. You must specify input and output directories.;
    %put ERROR: Usage: sas -sysin path\to\script.sas -SYSPARM "C:\path\to\data|C:\path\to\output";
    %abort cancel;
%end;

/* Split SYSPARM by pipe delimiter */
%let in_dir = %scan(&params, 1, |);
%let out_dir = %scan(&params, 2, |);

%if %length(&in_dir) = 0 %then %do;
    %put ERROR: Input directory not provided in SYSPARM.;
    %put ERROR: Usage: sas -sysin path\to\script.sas -SYSPARM "C:\path\to\data|C:\path\to\output";
    %abort cancel;
%end;

%if %length(&out_dir) = 0 %then %do;
    %put ERROR: Output directory not provided in SYSPARM.;
    %put ERROR: Usage: sas -sysin path\to\script.sas -SYSPARM "C:\path\to\data|C:\path\to\output";
    %abort cancel;
%end;

/* Create csv_dir for CSV output */
%let csv_dir = &out_dir.\DAC_CSV;

%put NOTE: Input directory = &in_dir;
%put NOTE: Output directory = &out_dir;
%put NOTE: CSV directory = &csv_dir;

/* 2) Point a libname to input directory (for *.sas7bdat) */
libname inlib "&in_dir";

/* 3) Create output folder if it does not exist */
options noxwait noxsync;
data _null_;
    length out $260 parent $260 rc $260;
    out = "&out_dir";

    /* Remove trailing backslash if present */
    if substr(out, length(out), 1) = '\' then
        out = substr(out, 1, length(out) - 1);

    if fileexist(out) then
        put "NOTE: Directory &out_dir already exists.";
    else do;
        /* Get parent directory for dcreate */
        pos = length(out) - index(reverse(out), '\') + 1;
        if pos > 0 and pos < length(out) then do;
            parent = substr(out, 1, pos - 1);
            folder = substr(out, pos + 1);
            rc = dcreate(folder, parent);
        end;
        else do;
            rc = dcreate(out, '');
        end;

        if rc ne '' then put "NOTE: Created directory &out_dir successfully.";
        else put "ERROR: Could not create directory &out_dir";
    end;
run;

/* 3a) Create DAC_CSV subfolder */
data _null_;
    length rc $260;
    if fileexist("&csv_dir") then
        put "NOTE: Directory &csv_dir already exists.";
    else do;
        rc = dcreate('DAC_CSV', "&out_dir");
        if rc ne '' then put "NOTE: Created directory &csv_dir successfully.";
        else put "ERROR: Could not create directory &csv_dir";
    end;
run;

/* 3b) Create DAC_Logs subfolder */
data _null_;
    length rc $260;
    if fileexist("&out_dir.\DAC_Logs") then
        put "NOTE: Directory &out_dir.\DAC_Logs already exists.";
    else do;
        rc = dcreate('DAC_Logs', "&out_dir");
        if rc ne '' then put "NOTE: Created directory &out_dir.\DAC_Logs successfully.";
        else put "ERROR: Could not create directory &out_dir.\DAC_Logs";
    end;
run;

/* 4) Set up error log file in DAC_Logs subdirectory */
filename errlog "&out_dir.\DAC_Logs\error_log.txt";

/* 5) Get list of all datasets in INDIR */
proc sql noprint;
    create table work._dslist as
    select memname
    from dictionary.tables
    where libname = upcase('INLIB')
    order by memname;
quit;

/* 6) Macro to export all datasets to CSV in csv_dir */
%macro export_all_to_csv;
    %local n i dsname nvar _err_count;
    %let _err_count = 0;

    proc sql noprint;
        select count(*) into :n trimmed from work._dslist;
    quit;

    %if &n = 0 %then %do;
        %put NOTE: No SAS datasets found in &in_dir;
        %return;
    %end;

    %do i = 1 %to &n;
        proc sql noprint;
            select memname into :dsname trimmed
            from work._dslist
            where monotonic() = &i;
        quit;

        /* Check that the dataset has at least one variable before exporting */
        proc sql noprint;
            select count(*) into :nvar trimmed
            from dictionary.columns
            where libname = upcase('INLIB')
              and memname = upcase("&dsname");
        quit;

        %if &nvar = 0 %then %do;
            %put WARNING: Skipping INLIB.&dsname - dataset has no variables defined.;
            %let _err_count = %eval(&_err_count + 1);
        %end;
        %else %do;
            %let csvfile = &csv_dir.\&dsname..csv;
            %put NOTE: Exporting INLIB.&dsname to &csvfile;
            proc export data=inlib.&dsname
                outfile="&csvfile"
                dbms=csv
                replace;
            run;
        %end;
    %end;

    /* Pass error count out of macro scope */
    %global _export_err_count;
    %let _export_err_count = &_err_count;
%mend export_all_to_csv;

%export_all_to_csv;

/* 7) Create error report */
data _null_;
    length dt $20 cnt $10;
    file errlog;
    dt = put(datetime(), datetime20.);
    cnt = symget('_export_err_count');

    put "SAS Error Report";
    put "Generated: " dt;
    put "----------------------------------------";

    if symget('_export_err_count') ne '0' then do;
        put "Warnings: " cnt " dataset(s) skipped due to no variables defined.";
        put "Check the SAS log for details (search for WARNING: Skipping).";
    end;
    else do;
        put "No errors detected in the SAS run.";
    end;
run;

/* 8) Clean up */
libname inlib clear;
filename errlog clear;
