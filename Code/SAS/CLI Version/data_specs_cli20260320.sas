/*------------------------------------------------------------------*
| MACRO NAME  : data_specs_cli20260320
| SHORT DESC  : Creates a specifications document on a library of
|               datasets with CLI parameter support
*------------------------------------------------------------------*
| CREATED BY  : DAC Development Team
| ORIGINAL    : Meyers, Jeffrey (07/28/2016)
*------------------------------------------------------------------*
| VERSION UPDATES:
| 2025-01-21: Modified to accept CLI parameters via SYSPARM
|   - Added input/output path parsing from SYSPARM
|   - Added automatic creation of DAC_Documents folder
|   - Added path validation
| 2026-03-20: Extended SYSPARM to support optional macro parameters
|   - Optional params passed as key=value pairs after output dir
| See original data_specs macro for prior version history.
*------------------------------------------------------------------*
| PURPOSE
| Create a data specifications sheet for all datasets in a
| designated directory with CLI parameter support for automated
| workflows. Allows for a quick look at the data within a library
| or allows for a simple data dictionary to be built.
| The macro outputs a report with three types of tables. The first
| is a listing of the datasets included in the library along with
| the number of rows, number of unique indexes, and number of
| variables. The second table shows the number of variables that
| exist across multiple datasets along with any different labels
| given to them. The last type of table is built for each dataset
| on a separate tab listing the variables and specs (label, format,
| and values) for each.
|
| 1.0: REQUIRED SYSPARM PARAMETERS (pipe-delimited)
| INPUT_DIRECTORY  = Path to the folder containing SAS datasets
|                    (.sas7bdat) to be summarized
| OUTPUT_DIRECTORY = Path to the folder where the DAC_Documents
|                    subfolder and output file will be created.
|                    Must exist prior to execution.
|
| 2.0: OPTIONAL SYSPARM PARAMETERS (pipe-delimited key=value pairs)
| INDEX         = Allows the designation of multiple index variables
|                 to determine number of patients or other interest
|                 within a dataset.
| CAT_THRESHOLD = Determines the number of variable levels that can
|                 exist before the program will not output the
|                 frequency and percentage of each level. If a
|                 numeric variable exceeds this many levels then
|                 distribution statistics will be used instead.
|                 Must be greater than or equal to 0. Default: 10.
| FORMAT        = For the summary of individual datasets this
|                 determines if the variables will be listed in
|                 CONDENSED (one row per variable), LONG (one
|                 column), or WIDE (one row) format. Options are
|                 LONG, CONDENSED, and WIDE. Default is LONG.
| ORDER         = Determines how the variables will be ordered in
|                 the dataset summary tabs. VARNUM will order by
|                 the variable order in the dataset and NAME will
|                 order alphabetically. Options are VARNUM and NAME.
|                 Default is VARNUM.
| WHERE         = Allows a WHERE clause to be used on the dictionary
|                 table to subset the datasets included.
| DEBUG         = Determines if notes are on/off and if temporary
|                 datasets are cleaned up. Options are 0 (no notes,
|                 datasets cleaned up) and 1 (notes shown, datasets
|                 left behind). Default is 0.
|
| USAGE:
|   sas -sysparm "input_directory|output_directory[|key=value|...]"
|       -sysin data_specs_cli20260320.sas
|
| OUTPUT STRUCTURE:
|   output_directory\
|   └── DAC_Documents\
|       └── data_specs_<GGG_PARENT>_<GG_PARENT>_<G_PARENT>_<YYYYMMDD>.xlsx
|           (summary sheet + one sheet per dataset)
*------------------------------------------------------------------*
| OPERATING SYSTEM COMPATIBILITY
| SAS v9.4 or Higher: Yes
*------------------------------------------------------------------*
| MACRO CALL
|
| Not applicable. Script is invoked directly via the SAS command
| line using -sysin and -sysparm. Optional parameters are passed
| as pipe-delimited key=value pairs within SYSPARM.
|
| Equivalent direct macro call (for reference):
|   %data_specs_cli(
|       index=,
|       cat_threshold=10,
|       where=,
|       format=long,
|       order=varnum,
|       debug=0
|   );
*------------------------------------------------------------------*
| EXAMPLES
|
| Basic usage (default options):
|   sas -sysparm "C:\data\input|C:\data\output"
|       -sysin data_specs_cli20260320.sas
|
| Wide format, alphabetical variable order:
|   sas -sysparm "C:\data\input|C:\data\output|format=wide|order=name"
|       -sysin data_specs_cli20260320.sas
|
| With index variable and custom category threshold:
|   sas -sysparm "C:\data\input|C:\data\output|index=USUBJID|cat_threshold=5"
|       -sysin data_specs_cli20260320.sas
|
| Full options example with debug enabled:
|   sas -sysparm "C:\data\input|C:\data\output|index=USUBJID|format=condensed|order=name|cat_threshold=15|debug=1"
|       -sysin data_specs_cli20260320.sas
*------------------------------------------------------------------*
| This program is free software; you can redistribute it and/or
| modify it under the terms of the GNU General Public License as
| published by the Free Software Foundation; either version 2 of
| the License, or (at your option) any later version.
|
| This program is distributed in the hope that it will be useful,
| but WITHOUT ANY WARRANTY; without even the implied warranty of
| MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
| General Public License for more details.
*------------------------------------------------------------------*/

%macro data_specs_cli(
    /*Optional Parameters*/
    index=,cat_threshold=10,where=,format=long,order=varnum,
    debug=0);

    /**Save current options to reset after macro runs**/
    %local _mergenoby _notes _quotelenmax _starttime _listing;
    %let _starttime=%sysfunc(time());
    %let _notes=%sysfunc(getoption(notes));
    %let _mergenoby=%sysfunc(getoption(mergenoby));
    %let _quotelenmax=%sysfunc(getoption(quotelenmax));

    /*Set Options*/
    options NOQUOTELENMAX nonotes mergenoby=nowarn;

    /**Parse SYSPARM for input and output paths**/
    %local sysparm_value indir outdir pipe_pos;
    %let sysparm_value = &sysparm;
    %put DEBUG: Raw SYSPARM value = &sysparm_value;

    %let pipe_pos = %sysfunc(findc(&sysparm_value, |));
    %put DEBUG: Pipe position = &pipe_pos;

    %if &pipe_pos > 0 %then %do;
        %let indir = %substr(&sysparm_value, 1, %eval(&pipe_pos - 1));
        %local remaining pipe2_pos;
        %let remaining = %substr(&sysparm_value, %eval(&pipe_pos + 1));

        /* Check for additional pipe-separated optional parameters */
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
                    %if %eval(&eq_pos) < %length(&ep_curr) %then
                        %let ep_value = %substr(&ep_curr, %eval(&eq_pos + 1));
                    %else
                        %let ep_value = ;
                    %if &ep_name = INDEX %then %let index = &ep_value;
                    %else %if &ep_name = CAT_THRESHOLD %then %do;
                        %if %sysevalf(%superq(ep_value)^=,boolean) %then %let cat_threshold = &ep_value;
                    %end;
                    %else %if &ep_name = FORMAT %then %do;
                        %if %sysevalf(%superq(ep_value)^=,boolean) %then %let format = &ep_value;
                    %end;
                    %else %if &ep_name = ORDER %then %do;
                        %if %sysevalf(%superq(ep_value)^=,boolean) %then %let order = &ep_value;
                    %end;
                    %else %if &ep_name = WHERE %then %let where = &ep_value;
                    %else %if &ep_name = DEBUG %then %do;
                        %if %sysevalf(%superq(ep_value)^=,boolean) %then %let debug = &ep_value;
                    %end;
                %end;
            %end;
            %put DEBUG: Parsed extra params: index=&index cat_threshold=&cat_threshold format=&format order=&order debug=&debug;
        %end;
        %else %do;
            %let outdir = &remaining;
        %end;
    %end;
    %else %do;
        %put ERROR: Invalid SYSPARM format. Expected: input_directory|output_directory;
        %goto errhandl;
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
        %goto errhandl;
    %end;

    %if &outdir_exist = 0 %then %do;
        %put ERROR: Output directory does not exist: &outdir;
        %goto errhandl;
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
            %goto errhandl;
        %end;
        %put DEBUG: Successfully created &doc_dir;
    %end;
    %else %do;
        %put DEBUG: Directory &doc_dir already exists;
    %end;

    /**Assign library to input directory**/
    libname inlib "&indir";

    %if %sysfunc(libref(inlib)) ^= 0 %then %do;
        %put ERROR: Failed to assign library to &indir;
        %goto errhandl;
    %end;

    /**Set output file path**/
    %local out_file g_parent gg_parent ggg_parent libname_text;
    %let g_parent = %scan(&indir, -1, \);
    %let gg_parent = %scan(&indir, -2, \);
    %let ggg_parent = %scan(&indir, -3, \);
    %let libname_text = &g_parent;
    %let out_file = &doc_dir\data_specs_%sysfunc(compress(&ggg_parent,,ka))_%sysfunc(compress(&gg_parent,,ka))_%sysfunc(compress(&g_parent,,ka))_%sysfunc(today(),yymmddn8.).xlsx;
    %put DEBUG: Output file = &out_file;

    /**See if the listing output is turned on**/
    proc sql noprint;
        select 1 into :_listing separated by '' from sashelp.vdest where upcase(destination)='LISTING';
    quit;

    /**Process Error Handling**/
    %if &sysver < 9.4 %then %do;
        %put ERROR: SAS must be version 9.4 or later;
        %goto errhandl;
    %end;

    %local nerror;
    %let nerror=0;

    /**Error Handling on Global Parameters**/
    %macro _gparmcheck(parm, parmlist);
        %local _test _z;
        %let _test=0;
        /**Check if values are in approved list**/
        %do _z=1 %to %sysfunc(countw(&parmlist,|,m));
            %if %qupcase(%superq(&parm))=%qupcase(%scan(&parmlist,&_z,|,m)) %then %let _test=1;
        %end;
        %if &_test ^= 1 %then %do;
            /**If not then throw error**/
            %put ERROR: (Global: %qupcase(&parm)): %superq(&parm) is not a valid value;
            %put ERROR: (Global: %qupcase(&parm)): Possible values are &parmlist;
            %let nerror=%eval(&nerror+1);
        %end;
    %mend;

    /*Data display format*/
    %_gparmcheck(format,condensed|wide|long)
    /*Variable Order*/
    %_gparmcheck(order,varnum|name)
    /*Debug switch*/
    %_gparmcheck(debug,0|1)

    /**Error Handling on Global Model Numeric Variables**/
    %macro _gnumcheck(parm, min);
        /**Check if missing**/
        %if %sysevalf(%superq(&parm)^=,boolean) %then %do;
            %if %sysfunc(notdigit(%sysfunc(compress(%superq(&parm),-.)))) > 0 %then %do;
                /**Check if character value**/
                %put ERROR: (Global: %qupcase(&parm)) Must be numeric.  %qupcase(%superq(&parm)) is not valid.;
                %let nerror=%eval(&nerror+1);
            %end;
            %else %if %superq(&parm) < &min %then %do;
                /**Makes sure number is not less than the minimum**/
                %put ERROR: (Global: %qupcase(&parm)) Must be greater than %superq(min). %qupcase(%superq(&parm)) is not valid.;
                %let nerror=%eval(&nerror+1);
            %end;
        %end;
        %else %do;
            /**Throw Error**/
            %put ERROR: (Global: %qupcase(&parm)) Cannot be missing;
            %put ERROR: (Global: %qupcase(&parm)) Possible values are numeric values greater than or equal to %superq(min);
            %let nerror=%eval(&nerror+1);
        %end;
    %mend;

    /*Categories cut-off*/
    %_gnumcheck(cat_threshold,0);

    /*** If any errors exist, stop macro and send to end ***/
    %if &nerror > 0 %then %do;
        %put ERROR: &nerror pre-run errors listed;
        %put ERROR: Macro DATA_SPECS_CLI will cease;
        %goto errhandl;
    %end;

    %if &debug=1 %then %do;
        options notes mprint;
    %end;

    /*Creates dictionary table for library*/
    proc contents data=inlib._all_
        out=_specs_ (%if %sysevalf(%superq(where)^=,boolean) %then %do;
                     where=(&where)
                     %end;) noprint;
    run;

    /*Sorts by VARNUM or NAME*/
    proc sort data=_specs_;
        by memname %superq(order);
    run;

    /**Set up Data sets to insert into**/
    proc sql;
        /**Data set for specifications**/
        create table _data_specs_
            (data_id num,
            dsn char(300) 'Data Set Name (Label)',
            var_name char(40) 'Name',
            type char(10) 'Type',
            length char(20) 'Length',
            format char(40) 'Format',
            informat char(40) 'Informat',
            label char(256) 'Label',
            cat_values char(10000) 'Category Values',
            spec char(30) 'Specification',
            value char(10000) 'Value');
        /**Data set for high level summary of library**/
        create table _libn_summary_
            (dsn char(300) 'Data Set Name',
            obs num 'Observations',
            unique_index num "Unique Index Values (%qupcase(%sysfunc(tranwrd(&index,|,%str( or )))))",
            variables num 'Number of Variables');
        /**Data set for variables in multiple datasets**/
        create table _var_summary_
            (var_name char(40) 'Variable Name',
            dsn_list char(10000) 'Datasets Containing Variable',
            labels char(10000) 'Variable Label(s)');
    quit;

    /*Gets list of datasets within library*/
    %local datalist;
    proc sql noprint;
        select distinct memname into :datalist separated by ' '
            from _specs_;
    quit;

    /*Loops between datasets in library*/
    %local i j varlist curr_index n_numeric;
    %do i = 1 %to %sysfunc(countw(&datalist,%str( )));
        %put Progress: %scan(&datalist,&i,%str( )): &i of %sysfunc(countw(&datalist,%str( )));
        %local varlist&i;
        %let varlist&i=;

        /*Creates list of all variables in data set*/
        proc sql noprint;
            select name into :varlist&i separated by ' '
                from _specs_ where upcase(memname)=upcase("%scan(&datalist,&i,%str( ))");
        quit;

        %let curr_index=;
        /*Determine if index variables are present*/
        %local index_varn index_set j k l;
        %let index_varn=;%let index_set=;
        %if %sysevalf(%superq(index)^=,boolean) %then %do j = 1 %to %sysfunc(countw(&&varlist&i,%str( )));
            %do k=1 %to %sysfunc(countw(%superq(index),|));
                %if %sysevalf(%superq(index_set)=,boolean) or %sysevalf(%superq(index_set)=&k,boolean) %then %do l = 1 %to %sysfunc(countw(%scan(&index,&k,|),%str( )));
                    %if %sysevalf(%qupcase(%scan(&&varlist&i,&j,%str( )))=%qupcase(%qscan(%qscan(&index,&k,|),&l,%str( ))),boolean) %then %let index_varn=&index_varn &j;
                %end;
            %end;
        %end;

        /*Creates macro variables for variable format and type*/
        data __temp;
            set inlib.%scan(&datalist,&i,%str( ));
            if _n_=1 then do;
                %do j = 1 %to %sysfunc(countw(&&varlist&i,%str( )));
                    %local v&j.format v&j.type;
                    call symput("v&j.format",vformat(%scan(&&varlist&i,&j,%str( ))));
                    call symput("v&j.type",vtype(%scan(&&varlist&i,&j,%str( ))));
                %end;
            end;
            rename %do j = 1 %to %sysfunc(countw(&&varlist&i,%str( ))); %scan(&&varlist&i,&j,%str( ))=_var&j %end;;
        run;

        %if %sysevalf(%superq(index_varn)^=,boolean) %then %do;
            proc sort data=__temp;
                by %do j=1 %to %sysfunc(countw(&index_varn,%str( ))); _var%scan(&index_varn,&j,%str( )) %end;;
            run;

            data __temp;
                set __temp;
                by %do j=1 %to %sysfunc(countw(&index_varn,%str( ))); _var%scan(&index_varn,&j,%str( )) %end;;
                if first._var%scan(&index_varn,%eval(%sysfunc(countw(&index_varn,%str( )))-1+1),%str( )) then _index_+1;
            run;
        %end;

        /*Creates variable lists*/
        proc sql noprint;
            /*Finds number of numeric values*/
            select sum(ifn(type=1,1,0)) into :n_numeric separated by ' '
                from _specs_ where upcase(memname)=upcase("%scan(&datalist,&i,%str( ))");

            /*Inserts current dataset specs into table*/
            insert into _data_specs_
                select &i,
                strip(memname)||
                   case(memlabel)
                       when '' then ''
                else ' ('||strip(memlabel)||')' end,
                name,
                case(type)
                  when 1 then 'Numeric'
                  when 2 then 'Character'
                else '' end,
                strip(put(length,12.)),
                format,
                informat,
                label,
                '',
                '',
                ''
                from _specs_ where upcase(memname)=upcase("%scan(&datalist,&i,%str( ))");

            /*Inserts high level dataset specs into library table*/
            insert into _libn_summary_
                select a.dsn,
                a.nobs,
                b.unique_index,
                a.n
                from (select distinct &i as data_id, strip(memname)||
                   case(memlabel)
                       when '' then ''
                    else ' ('||strip(memlabel)||')' end as dsn,nobs,count(distinct name) as n from _specs_
                         where upcase(memname)=upcase("%scan(&datalist,&i,%str( ))") group by nobs) as a left join
                  %if %sysevalf(%superq(index_varn)^=,boolean) %then %do;
                     (select distinct &i as data_id,"%scan(&datalist,&i,%str( ))" as dsn,count(distinct _index_) as unique_index
                        from __temp) as b
                  %end;
                  %else %do;
                     (select distinct &i as data_id,"%scan(&datalist,&i,%str( ))" as dsn,0 as unique_index
                          from inlib.%scan(&datalist,&i,%str( ))) as b
                  %end;
                  on a.data_id=b.data_id;

            /*Initializes and creates macro variables for levels of each variable*/
            %local n nothing;
            %do j = 1 %to %sysfunc(countw(&&varlist&i,%str( )));
               %local n_&j levels_&j;
               %let n_&j=;%let levels_&j=;
            %end;

            select %do j = 1 %to %sysfunc(countw(&&varlist&i,%str( )));
                count(distinct _var&j),
                %end;
                count(*) into
                %do j = 1 %to %sysfunc(countw(&&varlist&i,%str( )));
                  :n_&j,
                %end; :n from __temp;
        quit;

        %local n_frq n_dist n_miss;
        %let n_frq=;%let n_dist=;%let n_miss=;
        %if &n > 0 %then %do;
            /**Determine how each variable should be summarized**/
            %do j = 1 %to %sysfunc(countw(&&varlist&i,%str( )));
                %if %superq(v&j.type)=N %then %do;
                    %if %superq(n_&j) <= &cat_threshold %then %let n_frq=&n_frq &j;
                    %else %let n_dist=&n_dist &j;
                %end;
                %else %do;
                    %if %superq(n_&j) <= &cat_threshold %then %let n_frq=&n_frq &j;
                    %else %let n_miss=&n_miss &j;
                %end;
            %end;

            ods select none;
            ods noresults;

            /*Creates frequencies for all variables under the threshold*/
            %if %sysevalf(%superq(n_frq)^=,boolean) %then %do;
                proc freq data=__temp;
                    table %do j = 1 %to %sysfunc(countw(&n_frq,%str( )));
                              _var%scan(&n_frq,&j,%str( ))
                          %end; / missing;
                    ods output onewayfreqs=_frqs_;
                run;
            %end;

            /*Creates distributions for all numeric variables above the threshold*/
            %if %sysevalf(%superq(n_dist)^=,boolean) %then %do;
                proc means data=__temp missing noprint;
                    var %do j = 1 %to %sysfunc(countw(&n_dist,%str( )));
                             _var%scan(&n_dist,&j,%str( ))
                         %end;;
                    output out=_distribution_ n= nmiss= mean= std= median= min= max= / autoname;
                run;
            %end;

            ods select all;
            ods results;
        %end;

        proc sql noprint;
            /*Creates macro variables for values of variables*/
            %do j = 1 %to %sysfunc(countw(&&varlist&i,%str( )));
                /*If number of distinct levels are less than the threshold*/
                %if &n > 0 and %superq(n_&j) <= &cat_threshold %then %do;
                    select case (missing(f__var&j))
                        when 1 then 'Missing'
                        else case
                             when %if %superq(v&j.type)=N %then %do; strip(put(_var&j,best12.)) %end;
                                  %else %do; strip(_var&j) %end; = strip(f__var&j) then strip(f__var&j)
                             else %if %superq(v&j.type)=N %then %do; strip(put(_var&j,best12.)) %end;
                                  %else %do; strip(_var&j) %end; ||' (' || strip(f__var&j)||')' end end
                        ||': '||strip(put(frequency,comma12.))||' ('||strip(put(percent,12.1))||'%)'
                        into :levels_&j separated by '^n '
                        from _frqs_ where upcase(table)=upcase("Table _var&j");
                %end;
                %else %if &n > 0 and %superq(v&j.type)=N and %sysevalf(%superq(n_dist)^=,boolean) %then %do;
                    /*If number of distinct levels are greater than the threshold and variable is numeric*/
                    select 'N (N Missing): '||strip(put(_var&j._n,comma12.0))||' ('||strip(put(_var&j._nmiss,comma12.0))||')^n '||
                        'Median: '||strip(Put(_var&j._median,&&v&j.format))||'^n '||
                        'Range: '||strip(put(_var&j._min,&&v&j.format))||' - '||strip(put(_var&j._max,&&v&j.format))
                        into :levels_&j separated by '' from _distribution_;
                %end;
                /*Otherwise leave blank*/
                %else %do;
                    select 'N (N Missing): ' ||strip(put(sum(^missing(_var&j)),12.0)) ||' ('||
                        strip(put(sum(missing(_var&j)),12.0)) ||')' into :levels_&j separated by ''
                        from __temp;
                %end;
            %end;

            /*Delete temporary datasets*/
            %if %sysevalf(%superq(n_frq)^=,boolean) %then %do;
                drop table _frqs_;
            %end;
            %if %sysevalf(%superq(n_dist)^=,boolean) %then %do;
                drop table _distribution_;
            %end;

            /*Update formats and values into the specifications data set*/
            update _data_specs_
                set format=case (upcase(var_name))
                %do j = 1 %to %sysfunc(countw(&&varlist&i,%str( )));
                  when upcase("%scan(&&varlist&i,&j,%str( ))") then "%superq(v&j.format)"
                %end;
                else '' end,
                cat_values=case (upcase(var_name))
                %do j = 1 %to %sysfunc(countw(&&varlist&i,%str( )));
                  when upcase("%scan(&&varlist&i,&j,%str( ))") then "%superq(levels_&j)"
                %end;
                else '' end
                where strip(scan(upcase(dsn),1,'('))=upcase("%scan(&datalist,&i,%str( ))");
            drop table __temp;
        quit;
    %end;

    /*Transposes some specifications to long form*/
    data _data_specs_;
        set _data_specs_;

        spec='Variable';value=strip(var_name);output;
        spec='Label';value=strip(label);output;
        spec='Format';
        if upcase(type)='NUMERIC' then value='Numeric with format '||strip(format);
        else if upcase(type)='CHARACTER' then value='Character string of length '||strip(length)||' and format '||
            strip(format);
        output;
        %if %sysevalf(%qupcase(&format)=LONG, boolean) %then %do; if ^missing(cat_values) then do; %end;
            spec='Values';value=strip(cat_values);output;
        %if %sysevalf(%qupcase(&format)=LONG, boolean) %then %do; end; %end;
    run;

    /*Transpose into condensed form*/
    proc transpose data=_data_specs_ out=_data_specs_condensed (drop=_name_ _label_);
        by data_id notsorted var_name;
        id spec;
        var value;
    run;

    /*Merges variables into wide dataset*/
    %if %sysevalf(%qupcase(&format)=WIDE, boolean) %then %do;
        %do i = 1 %to %sysfunc(countw(&datalist,%str( )));
            data _data_specs_wide&i;
                merge
                    %do j = 1 %to %sysfunc(countw(&&varlist&i,%str( )));
                        _data_specs_ (keep=data_id dsn spec value var_name where=(data_id=&i and upcase(var_name)=upcase("%scan(%superq(varlist&i),&j,%str( ))"))
                                      rename=(var_name=var_name&j value=value&j))
                    %end;;
            run;
        %end;
        data _data_specs_wide;
            set %do i = 1 %to %sysfunc(countw(&datalist,%str( ))); _data_specs_wide&i %end;;
        run;
    %end;

    /*Completes variables across multiple datasets table*/
    proc sql noprint;
        %local multilist;
        %let multilist=;
        /*Find variables in multiple datasets*/
        select distinct upcase(name) into :multilist separated by ' '
            from _specs_ where upcase(name) ^in("%sysfunc(tranwrd(%upcase(&index),%str( ), " "))")
            group by upcase(name) having count(*)>1;

        /*Find Dataset name and label for variables*/
        %do i = 1 %to %sysfunc(countw(&multilist,%str( )));
            %local dlist&i dlabels&i;
            %let dlist&i=;%let dlabels&i=;
            select memname into :dlist&i separated by ', '
                from _specs_ where upcase(name)="%scan(&multilist,&i,%str( ))";
            select distinct strip(label) into :dlabels&i separated by '^n'
                from _specs_ where upcase(name)="%scan(&multilist,&i,%str( ))";
        %end;

        /*Insert values into dataset*/
        %if %sysevalf(%superq(multilist)^=,boolean) %then %do;
            insert into _var_summary_
                %do i = 1 %to %sysfunc(countw(&multilist,%str( )));
                    set var_name="%scan(&multilist,&i,%str( ))",
                        dsn_list="%superq(dlist&i)",
                        labels="%superq(dlabels&i)"
                %end;
                ;
        %end;
    quit;

    /*Print out datasets*/
    ods escapechar='^';
    %if &_listing=1 %then %do;
        ods listing close;
    %end;
    ods noresults;

    /*First tab has library summary and variables in multiple datasets summary*/
    ods excel file="&out_file" options(sheet_name="Library %qupcase(&libname_text) - Summary" sheet_interval='none' flow='tables');
    proc report data=_libn_summary_ nowd
        style(header)={background=white just=c fontfamily='Times New Roman' fontsize=9pt}
        style(column)={background=white just=left vjust=top fontfamily='Times New Roman' fontsize=9pt};
        columns ("Summary of Datasets within Library - %qupcase(&libname_text)" dsn obs unique_index variables);

        define dsn / display width=50;
        define unique_index / display width=50 center;
        define obs / display width=50 center;
        define variables / display width=50 center;

        compute variables;
            shade+1;
            if mod(shade,2)=0 then call define(_row_,'style/merge','style={background=greyef}');
            if substr('Dataset - '||strip(dsn),1,28)=substr('Dataset - '||strip(tempname),1,28) then do;
               if tempnamen=. then tempnamen=2;
               else tempnamen+1;
               urlstring="#'"||substr('Dataset - '||strip(dsn),1,28)||' '||strip(put(tempnamen,12.0))||"'!A3";
            end;
            else do;
               urlstring="#'"||substr('Dataset - '||strip(dsn),1,28)||"'!A3";
               tempnamen=.;
            end;
            if ^missing(dsn) then call define('dsn','url',urlstring);
        endcomp;
    run;

    proc report data=_var_summary_ nowd
        style(header)={background=white just=c fontfamily='Times New Roman' fontsize=9pt}
        style(column)={background=white just=left vjust=top fontfamily='Times New Roman' fontsize=9pt};
        columns ("Variables that Exist Across Multiple Datasets" var_name dsn_list labels);

        define dsn_list / display width=50;
        define var_name / display width=50;
        define labels / display width=50;

        compute labels;
            shade+1;
            if mod(shade,2)=0 then call define(_row_,'style/merge','style={background=greyef}');
        endcomp;
    run;

    %local topref;
    %let topref=%str(=HYPERLINK("#'Library %qupcase(&libname_text) - Summary'!A4","Click to Return to Top Summary"));

    /*Print one tab for each dataset: either long or wide form*/
    %do i = 1 %to %sysfunc(countw(&datalist,%str( )));
        %if %sysevalf(%qupcase(&format)=LONG, boolean) %then %do;
            ods excel options(sheet_name="Dataset - %scan(&datalist,&i,%str( ))" sheet_interval='table' frozen_headers='2');
            proc report data=_data_specs_ nowd
                style(header)={background=white just=c fontfamily='Times New Roman' fontsize=9pt}
                style(column)={background=white just=left vjust=top fontfamily='Times New Roman' fontsize=9pt};
                columns ("Dataset Name: %scan(&datalist,&i,%str( ))" ("%superq(topref)" var_name spec value));

                define var_name / order order=data width=50 noprint;
                define spec / display width=50;
                define value / display width=50;

                compute value;
                    if ^missing(var_name) then do;
                        shade+1;
                        call define(_row_,'style/merge','style={bordertopwidth=0.5pt bordertopcolor=black bordertopstyle=solid}');
                    end;
                    if mod(shade,2)=0 then call define(_row_,'style/merge','style={background=greyef}');
                endcomp;
                where data_id=&i;
            run;
        %end;
        %else %if %sysevalf(%qupcase(&format)=CONDENSED, boolean) %then %do;
            ods excel options(sheet_name="Dataset - %scan(&datalist,&i,%str( ))" sheet_interval='table' frozen_headers='2');
            proc report data=_data_specs_condensed nowd
                style(header)={background=white just=c fontfamily='Times New Roman' fontsize=9pt}
                style(column)={background=white just=left vjust=top fontfamily='Times New Roman' fontsize=9pt};
                columns ("Dataset Name: %scan(&datalist,&i,%str( ))"  ("%superq(topref)" var_name label format values));

                define var_name / order order=data width=50 'Variable';
                define label / display width=50;
                define format / display width=50;
                define values / display width=50;

                compute values;
                    shade+1;
                    call define(_row_,'style/merge','style={bordertopwidth=0.5pt bordertopcolor=black bordertopstyle=solid}');
                    if mod(shade,2)=0 then call define(_row_,'style/merge','style={background=greyef}');
                endcomp;
                where data_id=&i;
            run;
        %end;
        %else %if %sysevalf(%qupcase(&format)=WIDE, boolean) %then %do;
            ods excel options(sheet_name="Dataset - %scan(&datalist,&i,%str( ))" sheet_interval='table' frozen_headers='1' frozen_rowheaders='1');
            proc report data=_data_specs_wide nowd
                style(header)={background=white just=c fontfamily='Times New Roman' fontsize=9pt}
                style(column)={background=white just=left vjust=top fontfamily='Times New Roman' fontsize=9pt};
                columns ("Dataset Name: %scan(&datalist,&i,%str( ))"  ("%superq(topref)" spec
                %do j = 1 %to %sysfunc(countw(&&varlist&i,%str( )));
                    value&j
                %end;));

                define spec / display width=50 ' ';
                %do j = 1 %to %sysfunc(countw(&&varlist&i,%str( )));
                    define value&j / display width=50 ' ' style={cellwidth=2in};
                %end;

                compute spec;
                    shade+1;
                    if mod(shade,2)=0 then call define(_row_,'style/merge','style={background=greyef}');
                endcomp;
                where data_id=&i;
            run;
        %end;
    %end;

    ods excel close;
    ods results;

    %if &_listing=1 %then %do;
        ods listing;
    %end;

    /**Delete temporary datasets**/
    proc datasets nolist nodetails;
        %if &debug=0 %then %do;
            delete _data_specs_ _data_specs_condensed _data_specs_wide _libn_summary_ _specs_ _var_summary_
                %do i = 1 %to %sysfunc(countw(&datalist,%str( ))); _data_specs_wide&i %end;;
        %end;
    quit;

    libname inlib clear;

    %put NOTE: Data specifications document created: &out_file;

    %errhandl:
    /**Reload previous Options**/
    options mergenoby=&_mergenoby &_notes &_quotelenmax;
    %put DATA_SPECS_CLI has finished processing, runtime: %sysfunc(putn(%sysevalf(%sysfunc(TIME())-&_starttime.),mmss.));
%mend;

%data_specs_cli;
