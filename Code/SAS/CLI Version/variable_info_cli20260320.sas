/*------------------------------------------------------------------*
| PROGRAM NAME : variable_info_cli20260320.sas
| SHORT DESC   : Creates a variable information document with CLI
|                parameter support
*------------------------------------------------------------------*
| CREATED BY   : DAC Development Team
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
|   - Dropped FORMAT and INFORMAT columns from output
|   - Enforced column order: NUM, VARIABLE, TYPE, LEN, POS, LABEL
|   - Switched from PROC PRINT to PROC REPORT to enable per-column
|     width formatting in the Excel workbook output
|   - Centered alignment on all columns (header and data cells)
|   - Added variant column-name renaming: variant column names such as
|     LENGTH, VARIABLE_NAME, VAR_NUM, etc. are normalized to the
|     standard desired names (NUM, VARIABLE, TYPE, LEN, POS, LABEL) before
|     any columns are dropped, with DEBUG/NOTE logging for renames
|   - Added dynamic column-drop logic: any column not in the desired
|     list is removed from the dataset before output, with DEBUG
|     notification printed to the console and a highlighted NOTE
|     written to the SAS log
| 2026-05-18: Bug fix - empty drop_cols guard
|   - Wrapped %sysfunc(strip(&drop_cols)) in a %length > 0 guard to
|     prevent "too few arguments" ERROR when all columns are already
|     in the desired list and drop_cols is empty
| 2026-05-19: Fixed dcreate return-value check
|   - Replaced `%if &rc = %then` with `%if %length(&rc) = 0 %then`
|     for the DAC_Documents directory-creation guard. dcreate returns
|     the new directory name on success, causing %EVAL to abort with
|     a numeric-operand error when the directory was successfully
|     created.
| 2026-03-29: Column name variant fix
|   - Added OPTIONS VALIDVARNAME=V7 to force standard V7 column names
|     in PROC CONTENTS ODS output (prevents space-containing names
|     like 'Variable Number' that occur under VALIDVARNAME=ANY)
|   - Added space-containing column name variants to the rename map
|     as defense-in-depth for environments that override VALIDVARNAME:
|       NUM      <- VARIABLE NUMBER, VAR NUM, VAR NO
|       VARIABLE <- VARIABLE NAME, VAR NAME
|       TYPE     <- VAR TYPE, DATA TYPE, VARIABLE TYPE
|       LEN      <- (no space variants needed)
|       POS      <- POSITION, START POSITION, COLUMN POSITION, START
|       LABEL    <- VARIABLE LABEL, VAR LABEL
|   - Updated rename SQL to use SAS name literals ('name'n) for any
|     column names containing spaces, so PROC DATASETS RENAME works
|     correctly under VALIDVARNAME=ANY
| 2026-05-20: Empty-library guard
|   - Added dataset count check immediately after the INLIB libname
|     assignment. When 0 SAS datasets are found in the input directory,
|     the macro now logs a NOTE and returns gracefully instead of
|     crashing with "ERROR: File WORK.ALLVAROUT.DATA does not exist."
|     (which occurred because proc contents ODS capture does not create
|     the ALLVAROUT output object when the library has no members).
| 2026-05-20: Strip library prefix from MEMBER column
|   - Added a data step after column cleanup to replace MEMBER values
|     like "INLIB.AE" with just the dataset name "AE" by taking the
|     last dot-delimited token (scan(member,-1,'.')). This affects
|     Excel sheet names, BY-group titles, and any downstream consumers.
*------------------------------------------------------------------*
| PURPOSE
| Creates a variable-level information report for all datasets in a
| designated directory. For each dataset the report lists variable
| number, name, type, length, position, and label. Results are exported to a
| multi-sheet Excel workbook (.xlsx) in the DAC_Documents subfolder
| of the output directory, with one sheet per dataset.
| Output columns: NUM, VARIABLE, TYPE, LEN, POS, LABEL.
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
    /**Force V7 variable naming so PROC CONTENTS ODS output uses standard**/
    /**column names (Num, Variable, Type, Len, Pos, Label) instead of the **/
    /**space-containing names produced under VALIDVARNAME=ANY             **/
    /**(e.g. 'Variable Number' instead of 'Num').                        **/
    options validvarname=v7;

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

    /**Guard: exit gracefully when the library contains no SAS datasets**/
    /**proc contents uses ODS capture which does not create the ALLVAROUT  **/
    /**output object when the library has no members, causing all downstream**/
    /**steps to fail with "File WORK.ALLVAROUT.DATA does not exist."        **/
    %local ds_count;
    proc sql noprint;
        select count(*) into :ds_count trimmed
        from dictionary.tables
        where upcase(libname) = 'INLIB'
          and memtype = 'DATA';
    quit;

    %if &ds_count = 0 %then %do;
        %put NOTE: No SAS datasets found in &indir -- variable information document will not be created.;
        libname INLIB clear;
        %return;
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
    %let out_file = &doc_dir\variable_info_%sysfunc(compress(&ggg_parent,'_',ka))_%sysfunc(compress(&gg_parent,'_',ka))_%sysfunc(compress(&g_parent,'_',ka))_%sysfunc(today(),yymmddn8.).xlsx;
    %put DEBUG: Output file = &out_file;

    /**Get variable information**/
    ods output variables=allvarout;

    proc contents data=INLIB._all_ memtype=data;
    run;

    /**Sort data by dataset and variable number**/
    proc sort data=allvarout;
        by member num;
    run;

    /**Rename variant column names to standard desired names**/
    /**If a variant column name is found AND the target standard name does    **/
    /**not already exist, the column is renamed to the standard name.        **/
    %local rename_pairs renamed_log;
    %let rename_pairs = ;
    %let renamed_log = ;

    data _col_map_;
        length variant $32 standard $32;
        /* NUM variants (underscore-separated) */
        variant='VARIABLE_NUMBER'; standard='NUM'; output;
        variant='VAR_NUM';         standard='NUM'; output;
        variant='VAR_NO';          standard='NUM'; output;
        variant='NUMBER';          standard='NUM'; output;
        variant='NO';              standard='NUM'; output;
        /* NUM variants (space-separated — VALIDVARNAME=ANY) */
        variant='VARIABLE NUMBER'; standard='NUM'; output;
        variant='VAR NUM';         standard='NUM'; output;
        variant='VAR NO';          standard='NUM'; output;
        /* VARIABLE variants (underscore-separated) */
        variant='VARIABLE_NAME';   standard='VARIABLE'; output;
        variant='VAR_NAME';        standard='VARIABLE'; output;
        variant='NAME';            standard='VARIABLE'; output;
        /* VARIABLE variants (space-separated — VALIDVARNAME=ANY) */
        variant='VARIABLE NAME';   standard='VARIABLE'; output;
        variant='VAR NAME';        standard='VARIABLE'; output;
        /* TYPE variants (underscore-separated) */
        variant='VAR_TYPE';        standard='TYPE'; output;
        variant='DATA_TYPE';       standard='TYPE'; output;
        variant='VARIABLE_TYPE';   standard='TYPE'; output;
        /* TYPE variants (space-separated — VALIDVARNAME=ANY) */
        variant='VAR TYPE';        standard='TYPE'; output;
        variant='DATA TYPE';       standard='TYPE'; output;
        variant='VARIABLE TYPE';   standard='TYPE'; output;
        /* LEN variants */
        variant='LENGTH';          standard='LEN'; output;
        variant='SIZE';            standard='LEN'; output;
        /* POS variants (underscore-separated) */
        variant='POSITION';        standard='POS'; output;
        variant='START_POSITION';  standard='POS'; output;
        variant='COLUMN_POSITION'; standard='POS'; output;
        variant='START';           standard='POS'; output;
        /* POS variants (space-separated — VALIDVARNAME=ANY) */
        variant='START POSITION';  standard='POS'; output;
        variant='COLUMN POSITION'; standard='POS'; output;
        /* LABEL variants (underscore-separated) */
        variant='VARIABLE_LABEL';  standard='LABEL'; output;
        variant='VAR_LABEL';       standard='LABEL'; output;
        variant='DESCRIPTION';     standard='LABEL'; output;
        variant='DESC';            standard='LABEL'; output;
        /* LABEL variants (space-separated — VALIDVARNAME=ANY) */
        variant='VARIABLE LABEL';  standard='LABEL'; output;
        variant='VAR LABEL';       standard='LABEL'; output;
    run;

    proc sql noprint;
        /* Build rename pairs: only rename if variant exists and target does not.
           Use the actual column name from dictionary.columns (b.name) for the
           left side of the rename pair.  Wrap names containing spaces in SAS
           name literals ('name'n) so PROC DATASETS RENAME works correctly
           under VALIDVARNAME=ANY. */
        select case
                   when indexc(strip(b.name), ' ') > 0
                   then cats("'", strip(b.name), "'n=", a.standard)
                   else cats(strip(b.name), '=', a.standard)
               end
            into :rename_pairs separated by ' '
        from _col_map_ a
        inner join dictionary.columns b
            on upcase(b.name) = upcase(a.variant)
            and b.libname = 'WORK' and b.memname = 'ALLVAROUT'
        where upcase(a.standard) not in (
            select upcase(name) from dictionary.columns
            where libname = 'WORK' and memname = 'ALLVAROUT'
        );

        /* Build human-readable log of renames */
        select catx(' -> ', strip(b.name), a.standard)
            into :renamed_log separated by ', '
        from _col_map_ a
        inner join dictionary.columns b
            on upcase(b.name) = upcase(a.variant)
            and b.libname = 'WORK' and b.memname = 'ALLVAROUT'
        where upcase(a.standard) not in (
            select upcase(name) from dictionary.columns
            where libname = 'WORK' and memname = 'ALLVAROUT'
        );
    quit;

    %if %length(&rename_pairs) > 0 %then %do;
        proc datasets library=work nolist nodetails;
            modify allvarout;
            rename &rename_pairs;
        quit;
        %put DEBUG: Renamed variant columns to standard names: &renamed_log;
        %put NOTE: ************************************************************;
        %put NOTE: *** Columns renamed to standard names: &renamed_log ***;
        %put NOTE: ************************************************************;
    %end;
    %else %do;
        %put DEBUG: No variant column names found - all columns already use standard names;
    %end;

    proc datasets library=work nolist nodetails;
        delete _col_map_;
    quit;

    /**Identify and drop columns not in the desired list**/
    /**Desired output columns: NUM, VARIABLE, TYPE, LEN, POS, LABEL       **/
    /**MEMBER is retained for BY-group processing but excluded from output.**/
    %local drop_cols;
    proc contents data=allvarout out=_colinfo_(keep=name) noprint;
    run;

    proc sql noprint;
        select upcase(name) into :drop_cols separated by ' '
        from _colinfo_
        where upcase(name) not in ('NUM', 'VARIABLE', 'TYPE', 'LEN', 'POS', 'LABEL', 'MEMBER');
    quit;

    %if %length(&drop_cols) > 0 %then
        %let drop_cols = %sysfunc(strip(&drop_cols));

    %if %length(&drop_cols) > 0 %then %do;
        %put DEBUG: Dropping columns not in desired list: &drop_cols;
        %put NOTE: ************************************************************;
        %put NOTE: *** Columns dropped from output: &drop_cols ***;
        %put NOTE: ************************************************************;
        data allvarout;
            set allvarout(drop=&drop_cols);
        run;
    %end;
    %else %do;
        %put DEBUG: No columns to drop - all columns are in the desired list;
    %end;

    proc datasets library=work nolist nodetails;
        delete _colinfo_;
    quit;

    /**Strip library prefix from MEMBER (e.g. "INLIB.AE" -> "AE")**/
    /**proc contents includes the libname in the MEMBER column; removing **/
    /**it keeps sheet names and titles clean (just the dataset name).    **/
    data allvarout;
        set allvarout;
        member = scan(member, -1, '.');
    run;

    /**Create Excel output file with separate sheets per dataset**/
    /**Output columns: NUM, VARIABLE, TYPE, LEN, POS, LABEL              **/
    options nobyline;
    ods excel file="&out_file"
        options(sheet_name="#BYVAL(member)" embedded_titles='yes');

    proc report data=allvarout nowindows headline;
        columns num variable type len pos label;
        define num      / display 'Num'
                          style(header)=[just=center]
                          style(column)=[cellwidth=0.9in just=center];
        define variable / display 'Variable'
                          style(header)=[just=center]
                          style(column)=[cellwidth=1.5in just=center];
        define type     / display 'Type'
                          style(header)=[just=center]
                          style(column)=[cellwidth=0.75in just=center];
        define len      / display 'Len'
                          style(header)=[just=center]
                          style(column)=[cellwidth=0.75in just=center];
        define pos      / display 'Pos'
                          style(header)=[just=center]
                          style(column)=[cellwidth=0.9in just=center];
        define label    / display 'Label'
                          style(header)=[just=center]
                          style(column)=[cellwidth=3in just=center];
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
