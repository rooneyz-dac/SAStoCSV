/*------------------------------------------------------------------*
| PROGRAM NAME : rename_study_domains_cli20260320.sas
| SHORT DESC   : Standardizes SAS and XPT dataset names by removing
|                suffixes (e.g., AE_PLACEBO to AE) and placing
|                renamed copies in DAC_<ParentFolderName> or DAC_XPT subfolders
*------------------------------------------------------------------*
| CREATED BY   : DAC Development Team
| DATE CREATED : 2026-05-08
| VERSION      : 1.3
*------------------------------------------------------------------*
| VERSION UPDATES:
| 2026-05-08: Initial CLI release (v1.0)
|   - Added SYSPARM parsing for input and output directories
|   - Standardizes dataset names by keeping only the prefix before
|     the first underscore (e.g., AE_PLACEBO -> AE)
|   - Copies SAS7BDAT datasets to DAC_SAS subfolder when any file
|     in the input directory has a non-standard (suffixed) name
|   - Copies XPT datasets to DAC_XPT subfolder when any XPT file
|     in the input directory has a non-standard name
|   - Files without underscores are copied unchanged to the new
|     folder so the folder is a complete, self-consistent set
| 2026-05-12: SAS7BDAT logic updated to match OriginalSAS (v1.1)
|   - Replaced Windows DIR-based file listing with libname +
|     dictionary.tables discovery (aligns with
|     rename_study_domains_ZRlocalCopy.sas)
|   - ALL datasets are now copied to DAC_SAS first via proc copy,
|     then datasets with suffixes are renamed in-place in DAC_SAS
|     using proc datasets change (same approach as original script)
|   - Standard name derived with scan(memname, 1, '_') -- identical
|     to the original script
| 2026-05-12: Fix standard name extraction (v1.2)
|   - Changed scan position from 1 to -1 so the domain name is taken
|     from the LAST token after the final underscore (e.g.
|     BERKELEY_AE -> AE).  Using position 1 caused all datasets to
|     map to the study prefix (e.g. BERKELEY), triggering a duplicate-
|     name collision for every dataset in the library.
| 2026-05-19: Dynamic SAS output folder naming (v1.3)
|   - SAS output folder is now named DAC_<ParentFolderName> where
|     ParentFolderName is the leaf segment of the input directory
|     (e.g. if input is C:\data\rawdata, folder becomes DAC_rawdata).
|   - Handles both Windows backslash and Unix forward slash separators.
*------------------------------------------------------------------*
| PURPOSE
| Checks SAS (.sas7bdat) and XPT (.xpt) datasets in the specified
| input directory and standardizes their names by removing
| study-specific suffixes (e.g., converting AE_PLACEBO to AE).
| When any dataset requires renaming, ALL datasets of that type are
| copied with standardized names to a new subfolder:
|   - DAC_<ParentFolderName> for .sas7bdat files
|   - DAC_XPT for .xpt files
| This ensures that subsequent pipeline scripts operate on a
| complete, consistently named set of datasets.
|
| Naming rule: the standard name is the portion of the dataset name
| after the last underscore (scan(memname, -1, '_')).  E.g.:
|   BERKELEY_AE   -> AE
|   BERKELEY_SUPPAE -> SUPPAE
|   CM          -> CM  (unchanged; no folder created for this alone)
|
| 1.1: REQUIRED SYSPARM PARAMETERS (pipe-delimited)
| INPUT_DIRECTORY  = Path to folder containing .sas7bdat or .xpt
|                    files to be renamed
| OUTPUT_DIRECTORY = Path to folder where DAC_<ParentFolderName>
|                    and/or DAC_XPT subfolders will be created.
|                    Must exist prior to execution.
|
| USAGE:
|   sas -sysparm "input_directory|output_directory"
|       -sysin rename_study_domains_cli20260320.sas
|
| OUTPUT STRUCTURE:
|   output_directory\
|   ├── DAC_<ParentFolderName>\  - Standardized SAS7BDAT datasets
|   │                              (created only when any .sas7bdat
|   │                               dataset has a non-standard suffix)
|   └── DAC_XPT\   - Standardized XPT datasets
|                     (created only when any .xpt file has
|                      a non-standard suffix)
*------------------------------------------------------------------*
| OPERATING SYSTEM COMPATIBILITY
| SAS v9.4 or Higher: Yes
| Windows:            Yes
*------------------------------------------------------------------*
| NOTES
| - If two input datasets share the same standard name (e.g.,
|   AE_PLACEBO and AE_TREATMENT both map to AE), a WARNING is
|   issued and those datasets are skipped.  Resolve naming conflicts
|   in the source data before running the pipeline.
| - For XPT files the script queries the SAS transport library to
|   obtain the actual member name inside each file.  SAS XPT
|   format limits member names to 8 characters.
*------------------------------------------------------------------*
| EXAMPLES
|
| Standardize SAS dataset names:
|   sas -sysparm "C:\data\input|C:\data\output"
|       -sysin rename_study_domains_cli20260320.sas
|
| Standardize XPT dataset names:
|   sas -sysparm "C:\data\xpt_files|C:\data\output"
|       -sysin rename_study_domains_cli20260320.sas
*------------------------------------------------------------------*/

/* ----------------------------------------------------------------
   Helper macro: copy one XPT file to DAC_XPT with a standardized
   member name.  Called via %nrstr() inside a DATA step CALL EXECUTE
   so that each file is processed sequentially after the driving
   DATA step completes.

   Parameters:
     inpath        - full path to the source .xpt file
     outpath       - full path for the destination .xpt file
     standard_name - desired (standardized) SAS dataset name
---------------------------------------------------------------- */
%macro _copy_xpt_file(inpath=, outpath=, standard_name=);

    /* Open source XPT with SAS transport engine */
    libname _xinlib xport "&inpath";

    /* Get the actual member name stored inside the XPT file */
    proc sql noprint;
        select memname into :_xpt_memname trimmed
        from dictionary.tables
        where libname = '_XINLIB';
    quit;

    %if %length(&_xpt_memname) = 0 %then %do;
        %put WARNING: No dataset found in XPT file: &inpath -- skipping.;
        libname _xinlib clear;
        %return;
    %end;

    %let _xpt_memname_lc = %lowcase(&_xpt_memname);
    %let _std_lc         = %lowcase(&standard_name);

    %if %upcase(&_xpt_memname) ne %upcase(&standard_name) %then %do;
        /* Copy internal dataset to WORK under the standard name */
        data work.&_std_lc;
            set _xinlib.&_xpt_memname_lc;
        run;

        libname _xinlib clear;

        /* Write the WORK dataset to the new XPT file */
        libname _xoutlib xport "&outpath";
        proc copy in=work out=_xoutlib;
            select &_std_lc;
        run;
        libname _xoutlib clear;

        /* Clean up temporary WORK dataset */
        proc datasets lib=work nolist;
            delete &_std_lc;
        quit;
    %end;
    %else %do;
        /* Name already matches standard; copy file as-is */
        libname _xoutlib xport "&outpath";
        proc copy in=_xinlib out=_xoutlib; run;
        libname _xinlib  clear;
        libname _xoutlib clear;
    %end;

%mend _copy_xpt_file;


/* ================================================================
   Main macro
================================================================ */
%macro rename_study_domains;

    %let sysparm_value = &sysparm;
    %put NOTE: [rename_study_domains] Raw SYSPARM = &sysparm_value;

    /* ---- Parse SYSPARM: input_dir|output_dir ---- */
    %let pipe_pos = %sysfunc(findc(&sysparm_value, |));
    %if &pipe_pos > 0 %then %do;
        %let indir  = %substr(&sysparm_value, 1, %eval(&pipe_pos - 1));
        %let outdir = %substr(&sysparm_value, %eval(&pipe_pos + 1));
    %end;
    %else %do;
        %put ERROR: Invalid SYSPARM format. Expected: input_directory|output_directory;
        %abort;
    %end;

    %put NOTE: [rename_study_domains] Input Directory  = &indir;
    %put NOTE: [rename_study_domains] Output Directory = &outdir;

    /* ---- Derive output folder name from the input directory leaf segment ---- */
    /* Handles both Windows backslash and Unix forward slash separators.         */
    %let parent_folder_name = %scan(&indir, -1, \/);
    %let dac_sas_folder     = DAC_&parent_folder_name;
    %put NOTE: [rename_study_domains] SAS output folder name = &dac_sas_folder;

    /* ---- Validate directories ---- */
    %if %sysfunc(fileexist(&indir))  = 0 %then %do;
        %put ERROR: Input directory does not exist: &indir;
        %abort;
    %end;
    %if %sysfunc(fileexist(&outdir)) = 0 %then %do;
        %put ERROR: Output directory does not exist: &outdir;
        %abort;
    %end;


    /* ============================================================
       PART 1: Process SAS7BDAT files
       Uses dictionary.tables to discover library members, then
       renames with proc datasets change (logic from
       rename_study_domains_ZRlocalCopy.sas).
       ============================================================ */

    /* Point a libname at the input directory so SAS can query its
       member metadata via dictionary.tables */
    libname _inlib "&indir";

    /* Build a table of datasets that need renaming (have a suffix).
       Standard name = last token after the final underscore, e.g.
       BERKELEY_AE -> AE, BERKELEY_SUPPAE -> SUPPAE. */
    proc sql noprint;
        create table _work_sas as
        select
            memname,
            scan(memname, -1, '_') as standard_name length=32
        from dictionary.tables
        where upcase(libname) = '_INLIB'
          and index(memname, '_') > 0;

        select count(*) into :sas_rename_count trimmed from _work_sas;

        select count(*) into :sas_total trimmed
        from dictionary.tables
        where upcase(libname) = '_INLIB';
    quit;

    %put NOTE: Found &sas_total SAS7BDAT dataset(s) in library, &sas_rename_count require renaming.;

    %if &sas_rename_count > 0 %then %do;

        /* Check for duplicate standard names and warn/exclude */
        proc sql noprint;
            create table _work_sas_dups as
            select standard_name, count(*) as n
            from _work_sas
            group by standard_name
            having count(*) > 1;
            select count(*) into :sas_dup_count trimmed from _work_sas_dups;
        quit;

        %if &sas_dup_count > 0 %then %do;
            %put WARNING: Multiple SAS datasets map to the same standard name.;
            %put WARNING: The following datasets will be SKIPPED to avoid overwriting:;
            data _null_;
                set _work_sas_dups;
                put "  Standard name '" standard_name +(-1)
                    "' has " n "conflicting source datasets.";
            run;
            /* Exclude datasets with duplicate standard names */
            proc sql noprint;
                delete from _work_sas
                where standard_name in (select standard_name from _work_sas_dups);
            quit;
            /* Re-count after exclusion */
            proc sql noprint;
                select count(*) into :sas_rename_count trimmed from _work_sas;
            quit;
        %end;

        proc datasets lib=work nolist; delete _work_sas_dups; quit;

        %if &sas_rename_count > 0 %then %do;

            /* Create DAC_<ParentFolderName> output directory if needed */
            %let dac_sas_dir   = &outdir\&dac_sas_folder;
            %let dac_sas_exist = %sysfunc(fileexist(&dac_sas_dir));

            %if &dac_sas_exist = 0 %then %do;
                %put NOTE: Creating directory &dac_sas_dir;
                %let _rc = %sysfunc(dcreate(&dac_sas_folder, &outdir));
                %if &_rc = %then %do;
                    %put ERROR: Failed to create directory &dac_sas_dir;
                    libname _inlib clear;
                    %abort;
                %end;
            %end;
            %else %do;
                %put NOTE: Directory &dac_sas_dir already exists.;
            %end;

            /* Copy ALL datasets from input library to &dac_sas_folder so the
               output folder is a complete, self-consistent set.
               Original input files are never modified. */
            libname _outdac "&dac_sas_dir";
            proc copy in=_inlib out=_outdac memtype=data; run;
            libname _inlib clear;

            /* Rename datasets that have suffixes in &dac_sas_folder using
               proc datasets change (same approach as the original script) */
            data _null_;
                set _work_sas;
                cmd = 'proc datasets lib=_outdac nolist; change ' ||
                      strip(memname) || ' = ' || strip(standard_name) || '; quit;';
                call execute(cmd);
                put cmd;
            run;

            libname _outdac clear;

            %put NOTE: SAS dataset standardization complete.;
            %put NOTE: Standardized SAS files are in: &dac_sas_dir;

        %end; /* sas_rename_count > 0 after dedup */

    %end; /* sas_rename_count > 0 */
    %else %do;
        %put NOTE: All &sas_total SAS7BDAT dataset(s) already have standard names. No &dac_sas_folder folder created.;
        libname _inlib clear;
    %end;

    proc datasets lib=work nolist; delete _work_sas; quit;


    /* ============================================================
       PART 2: Process XPT files
       ============================================================ */

    filename _xptlst pipe "dir /b ""&indir\*.xpt"" 2>nul";

    data _work_xpt;
        infile _xptlst truncover;
        input filename $256.;
        if filename = '' then delete;
        length dsname $32 standard_name $32
               xpt_in_path $512 xpt_out_path $512;
        dsname        = upcase(scan(filename, 1, '.'));
        standard_name = upcase(scan(dsname,  -1, '_'));
        needs_rename  = (dsname ne standard_name);
        xpt_in_path   = "&indir\" || strip(filename);
        xpt_out_path  = "&outdir\DAC_XPT\" ||
                        strip(lowcase(standard_name)) || '.xpt';
    run;

    filename _xptlst clear;

    proc sql noprint;
        select count(*) into :xpt_total        trimmed from _work_xpt;
        select count(*) into :xpt_rename_count trimmed from _work_xpt
            where needs_rename = 1;
    quit;

    %put NOTE: Found &xpt_total XPT file(s), &xpt_rename_count require renaming.;

    %if &xpt_total > 0 and &xpt_rename_count > 0 %then %do;

        /* Check for duplicate standard names and warn/exclude */
        proc sql noprint;
            create table _work_xpt_dups as
            select standard_name, count(*) as n
            from _work_xpt
            group by standard_name
            having count(*) > 1;
            select count(*) into :xpt_dup_count trimmed from _work_xpt_dups;
        quit;

        %if &xpt_dup_count > 0 %then %do;
            %put WARNING: Multiple XPT datasets map to the same standard name.;
            %put WARNING: The following XPT datasets will be SKIPPED:;
            data _null_;
                set _work_xpt_dups;
                put "  Standard name '" standard_name +(-1)
                    "' has " n "conflicting source datasets.";
            run;
            proc sql noprint;
                delete from _work_xpt
                where standard_name in (select standard_name from _work_xpt_dups);
            quit;
            proc sql noprint;
                select count(*) into :xpt_total        trimmed from _work_xpt;
                select count(*) into :xpt_rename_count trimmed from _work_xpt
                    where needs_rename = 1;
            quit;
        %end;

        proc datasets lib=work nolist; delete _work_xpt_dups; quit;

        %if &xpt_rename_count > 0 %then %do;

            /* Create DAC_XPT output directory if needed */
            %let dac_xpt_dir   = &outdir\DAC_XPT;
            %let dac_xpt_exist = %sysfunc(fileexist(&dac_xpt_dir));

            %if &dac_xpt_exist = 0 %then %do;
                %put NOTE: Creating directory &dac_xpt_dir;
                %let _rc = %sysfunc(dcreate(DAC_XPT, &outdir));
                %if &_rc = %then %do;
                    %put ERROR: Failed to create directory &dac_xpt_dir;
                    %abort;
                %end;
            %end;
            %else %do;
                %put NOTE: Directory &dac_xpt_dir already exists.;
            %end;

            /* Generate a call to %_copy_xpt_file for each XPT file.
               Using %nrstr prevents premature macro resolution of the
               parameter values inside the DATA step. */
            data _null_;
                set _work_xpt;
                call execute(
                    '%_copy_xpt_file(inpath=' || strip(xpt_in_path)   ||
                    ', outpath='              || strip(xpt_out_path)  ||
                    ', standard_name='        || strip(standard_name) || ')'
                );
            run;

            %put NOTE: XPT dataset standardization complete.;
            %put NOTE: Standardized XPT files are in: &dac_xpt_dir;

        %end; /* xpt_rename_count > 0 after dedup */

    %end; /* xpt_total > 0 and xpt_rename_count > 0 */
    %else %if &xpt_total > 0 %then %do;
        %put NOTE: All &xpt_total XPT file(s) already have standard names. No DAC_XPT folder created.;
    %end;

    proc datasets lib=work nolist; delete _work_xpt; quit;

    %put NOTE: [rename_study_domains] Processing complete.;

%mend rename_study_domains;

%rename_study_domains;
