/*------------------------------------------------------------------*
| PROGRAM NAME : library_info_cli20260320.sas
| SHORT DESC   : Creates a library information document with CLI
|                parameter support
*------------------------------------------------------------------*
| CREATED BY   : DAC Development Team
| DATE CREATED : 2025-01-21
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
|       └── library_info_<GGG_PARENT>_<GG_PARENT>_<G_PARENT>_<YYYYMMDD>.xlsx
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
    /**Optional third segment: pipe-delimited key=value pairs (e.g. name_dir=...) **/
    %local sysparm_value indir outdir pipe_pos name_dir;
    %let sysparm_value = &sysparm;
    %let name_dir = ;
    %put DEBUG: Raw SYSPARM value = &sysparm_value;

    %let pipe_pos = %sysfunc(findc(&sysparm_value, |));
    %put DEBUG: Pipe position = &pipe_pos;

    %if &pipe_pos > 0 %then %do;
        %let indir = %substr(&sysparm_value, 1, %eval(&pipe_pos - 1));
        %local remaining pipe2_pos;
        %let remaining = %substr(&sysparm_value, %eval(&pipe_pos + 1));

        /* Check for additional pipe-separated optional key=value parameters */
        %let pipe2_pos = %sysfunc(findc(&remaining, |));
        %if &pipe2_pos > 0 %then %do;
            %let outdir = %substr(&remaining, 1, %eval(&pipe2_pos - 1));
            %local extra_params ep_count ep_i ep_curr ep_name eq_pos ep_value;
            %let extra_params = %substr(&remaining, %eval(&pipe2_pos + 1));
            %let ep_count = %sysfunc(countw(&extra_params, |));
            %do ep_i = 1 %to &ep_count;
                %let ep_curr = %scan(&extra_params, &ep_i, |);
                %let eq_pos = %sysfunc(findc(&ep_curr, =));
                %if &eq_pos > 0 %then %do;
                    %let ep_name = %qupcase(%substr(&ep_curr, 1, %eval(&eq_pos - 1)));
                    %if &eq_pos < %length(&ep_curr) %then %do;
                        %let ep_value = %substr(&ep_curr, %eval(&eq_pos + 1));
                    %end;
                    %else %do;
                        %let ep_value = ;
                    %end;
                    %if &ep_name = NAME_DIR %then %do;
                        %let name_dir = &ep_value;
                    %end;
                %end;
            %end;
        %end;
        %else %do;
            %let outdir = &remaining;
        %end;
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
        %if %length(&rc) = 0 %then %do;
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
    /**When name_dir is provided (XPT-to-SAS conversion scenario), use it for   **/
    /**deriving the path segments in the output filename so the name reflects    **/
    /**the original study directory rather than the DAC_SDTM conversion folder. **/
    /**Use both \ and / as path delimiters to handle Windows and Unix paths.    **/
    %local out_file g_parent gg_parent ggg_parent name_path;
    %if %length(%superq(name_dir)) > 0 %then %let name_path = &name_dir;
    %else %let name_path = &indir;
    %let g_parent = %scan(&name_path, -1, \/);
    %let gg_parent = %scan(&name_path, -2, \/);
    %let ggg_parent = %scan(&name_path, -3, \/);
    %let out_file = &doc_dir\library_info_%sysfunc(compress(&ggg_parent,'_',ka))_%sysfunc(compress(&gg_parent,'_',ka))_%sysfunc(compress(&g_parent,'_',ka))_%sysfunc(today(),yymmddn8.).xlsx;
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
