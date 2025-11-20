/* ----------------------------------------
Code exported from SAS Enterprise Guide
DATE: Friday, October 31, 2025     TIME: 10:10:32 AM
PROJECT: DataPreproccessing
PROJECT PATH: C:\Users\rooneyz\Documents\Code\DataPreproccessing.egp
---------------------------------------- */

%macro _eg_hidenotesandsource;
	%global _egnotes;
	%global _egsource;
	
	%let _egnotes=%sysfunc(getoption(notes));
	options nonotes;
	%let _egsource=%sysfunc(getoption(source));
	options nosource;
%mend _eg_hidenotesandsource;
/* Step 1: Create empty datasets for results and summary */
data comparison_results;
    length filename $100. variable $32. base_value $200. compare_value $200.;
    stop;
run;

data comparison_summary;
    length filename $100. status $20. base_obs 8 compare_obs 8 diff_count 8;
    stop;
run;

/* Step 2: Get list of CSV files */
filename csvlist pipe "dir /b &folder1.\*.csv";
data csv_files;
    length filename $100.;
    infile csvlist truncover;
    input filename $;
run;

/* Step 3: Enhanced macro to compare CSV files and track summary */
%macro compare_csvs(file);
    %local base_obs comp_obs diff_obs;

    /* Import both files */
    proc import datafile="&folder1.\&file" out=work.f1_data dbms=csv replace;
        guessingrows=max;
    run;

    proc import datafile="&folder2.\&file" out=work.f2_data dbms=csv replace;
        guessingrows=max;
    run;

    /* Get observation counts */
    %let dsid = %sysfunc(open(work.f1_data));
    %let base_obs = %sysfunc(attrn(&dsid, nobs));
    %let rc = %sysfunc(close(&dsid));

    %let dsid = %sysfunc(open(work.f2_data));
    %let comp_obs = %sysfunc(attrn(&dsid, nobs));
    %let rc = %sysfunc(close(&dsid));

    /* Compare the datasets */
    proc compare base=work.f1_data compare=work.f2_data
                 out=work.diffs outnoequal noprint;
    run;

    /* Count differences */
    %let diff_obs = 0;
    %if %sysfunc(exist(work.diffs)) %then %do;
        %let dsid = %sysfunc(open(work.diffs));
        %let diff_obs = %sysfunc(attrn(&dsid, nobs));
        %let rc = %sysfunc(close(&dsid));

        /* Add filename to differences */
        data work.diffs;
            set work.diffs;
            length filename $100.;
            filename = "&file";
        run;

        /* Append to results */
        proc append base=comparison_results data=work.diffs force;
        run;
    %end;

    /* Add summary record */
    data work.summary_rec;
        length filename $100. status $20. base_obs 8 compare_obs 8 diff_count 8;
        filename = "&file";
        base_obs = &base_obs;
        compare_obs = &comp_obs;
        diff_count = &diff_obs;
        if diff_count > 0 then status = "DIFFERENCES FOUND";
        else if base_obs ne compare_obs then status = "ROW COUNT MISMATCH";
        else status = "IDENTICAL";
    run;

    proc append base=comparison_summary data=work.summary_rec force;
    run;

    /* Clean up temporary datasets */
    proc datasets library=work nolist;
        delete f1_data f2_data diffs summary_rec;
    quit;
%mend;

/* Step 4: Loop through all CSV files */
data _null_;
    set csv_files;
    call execute('%compare_csvs(' || trim(filename) || ')');
run;

/* Step 5: Always export detailed report with multiple sheets */
%macro export_results;
    %let dsid = %sysfunc(open(comparison_summary));
    %let nobs = %sysfunc(attrn(&dsid, nobs));
    %let rc = %sysfunc(close(&dsid));

    %if &nobs > 0 %then %do;
        /* Export summary sheet */
        proc export data=comparison_summary
            outfile="&output_file"
            dbms=xlsx
            replace;
            sheet="Summary";
        run;

        /* Export detailed differences if any exist */
        %let dsid = %sysfunc(open(comparison_results));
        %let diff_nobs = %sysfunc(attrn(&dsid, nobs));
        %let rc = %sysfunc(close(&dsid));

        %if &diff_nobs > 0 %then %do;
            proc export data=comparison_results
                outfile="&output_file"
                dbms=xlsx;
                sheet="Detailed Differences";
            run;
        %end;

        %put NOTE: Comparison report exported to &output_file;
        %put NOTE: Total files compared: &nobs;
        %put NOTE: Files with differences: &diff_nobs;
    %end;
    %else %do;
        /* Create empty summary if no files processed */
        data empty_summary;
            length message $200.;
            message = "No CSV files found to compare";
        run;

        proc export data=empty_summary
            outfile="&output_file"
            dbms=xlsx
            replace;
            sheet="Summary";
        run;

        %put WARNING: No files found to compare;
    %end;
%mend;

%export_results;

%macro _eg_restorenotesandsource;
	%global _egnotes;
	%global _egsource;
	
	options &_egnotes;
	options &_egsource;
%mend _eg_restorenotesandsource;


/* ---------------------------------- */
/* MACRO: enterpriseguide             */
/* PURPOSE: define a macro variable   */
/*   that contains the file system    */
/*   path of the WORK library on the  */
/*   server.  Note that different     */
/*   logic is needed depending on the */
/*   server type.                     */
/* ---------------------------------- */
%macro enterpriseguide;
%global sasworklocation;
%local tempdsn unique_dsn path;

%if &sysscp=OS %then %do; /* MVS Server */
	%if %sysfunc(getoption(filesystem))=MVS %then %do;
        /* By default, physical file name will be considered a classic MVS data set. */
	    /* Construct dsn that will be unique for each concurrent session under a particular account: */
		filename egtemp '&egtemp' disp=(new,delete); /* create a temporary data set */
 		%let tempdsn=%sysfunc(pathname(egtemp)); /* get dsn */
		filename egtemp clear; /* get rid of data set - we only wanted its name */
		%let unique_dsn=".EGTEMP.%substr(&tempdsn, 1, 16).PDSE"; 
		filename egtmpdir &unique_dsn
			disp=(new,delete,delete) space=(cyl,(5,5,50))
			dsorg=po dsntype=library recfm=vb
			lrecl=8000 blksize=8004 ;
		options fileext=ignore ;
	%end; 
 	%else %do; 
        /* 
		By default, physical file name will be considered an HFS 
		(hierarchical file system) file. 
		*/
		%if "%sysfunc(getoption(filetempdir))"="" %then %do;
			filename egtmpdir '/tmp';
		%end;
		%else %do;
			filename egtmpdir "%sysfunc(getoption(filetempdir))";
		%end;
	%end; 
	%let path=%sysfunc(pathname(egtmpdir));
    %let sasworklocation=%sysfunc(quote(&path));  
%end; /* MVS Server */
%else %do;
	%let sasworklocation = "%sysfunc(getoption(work))/";
%end;
%if &sysscp=VMS_AXP %then %do; /* Alpha VMS server */
	%let sasworklocation = "%sysfunc(getoption(work))";                         
%end;
%if &sysscp=CMS %then %do; 
	%let path = %sysfunc(getoption(work));                         
	%let sasworklocation = "%substr(&path, %index(&path,%str( )))";
%end;
%mend enterpriseguide;

%enterpriseguide


/* Conditionally delete set of tables or views, if they exists          */
/* If the member does not exist, then no action is performed   */
%macro _eg_conditional_dropds /parmbuff;
	
   	%local num;
   	%local stepneeded;
   	%local stepstarted;
   	%local dsname;
	%local name;

   	%let num=1;
	/* flags to determine whether a PROC SQL step is needed */
	/* or even started yet                                  */
	%let stepneeded=0;
	%let stepstarted=0;
   	%let dsname= %qscan(&syspbuff,&num,',()');
	%do %while(&dsname ne);	
		%let name = %sysfunc(left(&dsname));
		%if %qsysfunc(exist(&name)) %then %do;
			%let stepneeded=1;
			%if (&stepstarted eq 0) %then %do;
				proc sql;
				%let stepstarted=1;

			%end;
				drop table &name;
		%end;

		%if %sysfunc(exist(&name,view)) %then %do;
			%let stepneeded=1;
			%if (&stepstarted eq 0) %then %do;
				proc sql;
				%let stepstarted=1;
			%end;
				drop view &name;
		%end;
		%let num=%eval(&num+1);
      	%let dsname=%qscan(&syspbuff,&num,',()');
	%end;
	%if &stepstarted %then %do;
		quit;
	%end;
%mend _eg_conditional_dropds;


/* save the current settings of XPIXELS and YPIXELS */
/* so that they can be restored later               */
%macro _sas_pushchartsize(new_xsize, new_ysize);
	%global _savedxpixels _savedypixels;
	options nonotes;
	proc sql noprint;
	select setting into :_savedxpixels
	from sashelp.vgopt
	where optname eq "XPIXELS";
	select setting into :_savedypixels
	from sashelp.vgopt
	where optname eq "YPIXELS";
	quit;
	options notes;
	GOPTIONS XPIXELS=&new_xsize YPIXELS=&new_ysize;
%mend _sas_pushchartsize;

/* restore the previous values for XPIXELS and YPIXELS */
%macro _sas_popchartsize;
	%if %symexist(_savedxpixels) %then %do;
		GOPTIONS XPIXELS=&_savedxpixels YPIXELS=&_savedypixels;
		%symdel _savedxpixels / nowarn;
		%symdel _savedypixels / nowarn;
	%end;
%mend _sas_popchartsize;


ODS PROCTITLE;
OPTIONS DEV=PNG;
GOPTIONS XPIXELS=0 YPIXELS=0;
FILENAME EGSRX TEMP;
ODS tagsets.sasreport13(ID=EGSRX) FILE=EGSRX
    STYLE=HTMLBlue
    STYLESHEET=(URL="file:///C:/Program%20Files%20(x86)/SASHome/x86/SASEnterpriseGuide/7.1/Styles/HTMLBlue.css")
    NOGTITLE
    NOGFOOTNOTE
    GPATH=&sasworklocation
    ENCODING=UTF8
    options(rolap="on")
;

/*   START OF NODE: sas_csv_compare1030   */
%_eg_hidenotesandsource;


GOPTIONS ACCESSIBLE;
%_eg_restorenotesandsource;

%let folder1 =C:\Users\rooneyz\Documents\Code\SAS_Out\DAC_CSV;
%let folder2 =C:\Users\rooneyz\Documents\TestData\raw_data_csv\Enyo_v2_CSV;
%let output_file =C:\Users\rooneyz\Documents\Code\SAS_Out\sas_comparison_report.xlsx;


/* Step 1: Create an empty dataset to store comparison results */
data comparison_results;
    length filename $100. variable $32. base_value $200. compare_value $200.;
    stop;
run;

/* Step 2: Get list of CSV files */
filename csvlist pipe "dir /b &folder1.\*.csv";
data csv_files;
    length filename $100.;
    infile csvlist truncover;
    input filename $;
run;

/* Step 3: Macro to compare CSV files */
%macro compare_csvs(file);
    /* Import both files */
    proc import datafile="&folder1.\&file" out=work.f1_data dbms=csv replace;
        guessingrows=max;
    run;
    
    proc import datafile="&folder2.\&file" out=work.f2_data dbms=csv replace;
        guessingrows=max;
    run;
    
    /* Compare the datasets */
    proc compare base=work.f1_data compare=work.f2_data 
                 out=work.diffs outnoequal noprint;
    run;
    
    /* Check if differences exist */
    %if %sysfunc(exist(work.diffs)) %then %do;
        /* Add filename to differences */
        data work.diffs;
            set work.diffs;
            length filename $100.;
            filename = "&file";
        run;
        
        /* Append to results */
        proc append base=comparison_results data=work.diffs force;
        run;
    %end;
    
    /* Clean up temporary datasets */
    proc datasets library=work nolist;
        delete f1_data f2_data diffs;
    quit;
%mend;

/* Step 4: Loop through all CSV files */
data _null_;
    set csv_files;
    call execute('%compare_csvs(' || trim(filename) || ')');
run;

/* Step 5: Export results if any differences found */
%macro export_results;
    %let dsid = %sysfunc(open(comparison_results));
    %let nobs = %sysfunc(attrn(&dsid, nobs));
    %let rc = %sysfunc(close(&dsid));
    
    %if &nobs > 0 %then %do;
        proc export data=comparison_results
            outfile="&output_file"
            dbms=xlsx
            replace;
        run;
        
        %put NOTE: Comparison results exported to &output_file;
        %put NOTE: &nobs differences found;
    %end;
    %else %do;
        %put NOTE: No differences found between the datasets;
    %end;
%mend;

%export_results;




%_eg_hidenotesandsource;

GOPTIONS NOACCESSIBLE;
%LET _CLIENTTASKLABEL=;
%LET _CLIENTPROCESSFLOWNAME=;
%LET _CLIENTPROJECTPATH=;
%LET _CLIENTPROJECTPATHHOST=;
%LET _CLIENTPROJECTNAME=;
%LET _SASPROGRAMFILE=;
%LET _SASPROGRAMFILEHOST=;

%_eg_restorenotesandsource;

;*';*";*/;quit;run;
ODS _ALL_ CLOSE;
