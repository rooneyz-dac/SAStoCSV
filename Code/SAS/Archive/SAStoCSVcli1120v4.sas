
/* ---------------------------------------------------------
   SAStoCSVcli1120v7.sas
   Usage (Windows CMD):
   sas -sysin C:\path\to\SAStoCSVcli1120v7.sas -SYSPARM "C:\path\to\sas\data"
--------------------------------------------------------- */

/* 1) Get input directory from -SYSPARM, or error if not provided */
%let in_dir = %sysfunc(dequote(&SYSPARM));

%if %length(&in_dir) = 0 %then %do;
    %put ERROR: SYSPARM not provided. You must specify an input directory.;
    %put ERROR: Usage: sas -sysin path\to\script.sas -SYSPARM "C:\path\to\data";
    %abort cancel;
%end;

%put NOTE: Input directory = &in_dir;

/* 2) Point a libname to input directory (for *.sas7bdat) */
libname inlib "&in_dir";

/* 3) Derive parent directory and DAC_CSV output directory */
data _null_;
    length in $260 parent $260 out $260;
    in = pathname('inlib');
    in = translate(in, '\', '/');

    if substr(in, length(in), 1) = '\' then
        in = substr(in, 1, length(in) - 1);

    pos = length(in) - index(reverse(in), '\') + 1;

    if pos > 0 and pos < length(in) then
        parent = substr(in, 1, pos - 1);
    else
        parent = in;

    if substr(parent, length(parent), 1) = '\' then
        parent = substr(parent, 1, length(parent) - 1);

    out = cats(parent, '\DAC_CSV');

    call symputx('parent_dir', parent, 'G');
    call symputx('out_dir', out, 'G');
run;

%put NOTE: Parent directory = &parent_dir;
%put NOTE: Output directory = &out_dir;


/* 4) Create DAC_CSV folder if it does not exist */
options noxwait noxsync;
data _null_;
    rc = dcreate('DAC_CSV', "&parent_dir");
    if rc = '' then put "NOTE: Directory &out_dir already exists or created successfully.";
    else put "ERROR: Could not create directory &out_dir";
run;

/* 5) Set up error log file in output directory */
filename errlog "&out_dir.\error_log.txt";

/* 6) Get list of all datasets in INDIR */
proc sql noprint;
    create table work._dslist as
    select memname
    from dictionary.tables
    where libname = upcase('INLIB')
    order by memname;
quit;

/* 7) Macro to export all datasets to CSV in DAC_CSV */
%macro export_all_to_csv;
    %local n i dsname;
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
        %let csvfile = &out_dir.\&dsname..csv;
        %put NOTE: Exporting INLIB.&dsname to &csvfile;
        proc export data=inlib.&dsname
            outfile="&csvfile"
            dbms=csv
            replace;
        run;
    %end;
%mend export_all_to_csv;

%export_all_to_csv;

/* 8) Create error report based on SYSERR */
data _null_;
    length dt $20 syserr_val $10;
    file errlog;
    dt = put(datetime(), datetime20.);
    syserr_val = symget('SYSERR');

    put "SAS Error Report";
    put "Generated: " dt;
    put "----------------------------------------";

    if syserr_val ne '0' then do;
        put "Errors detected. SYSERR = " syserr_val;
        put "Check the SAS log for details.";
    end;
    else do;
        put "No errors detected in the SAS run.";
    end;
run;

/* 9) Clean up */
libname inlib clear;
filename errlog clear;




