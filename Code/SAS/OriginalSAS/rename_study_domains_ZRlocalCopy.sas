/***********************************************************************************
* PROGRAM NAME :rename_study_domains
* DESCRIPTION :  One-time renaming of STDY4A datasets to remove suffixes (e.g., AE_PLACEBO to AE)

* CALLED BY :  
* CALLS TO :   
* PROGRAMMER :  Divyasree Mannem
* DATE WRITTEN :  12/25/2025
************************************************************************************/

%let folder_path = &edata2.\MASHPDB\AKERO\Balanced_20251205\SDTM\SAS;

libname study "&folder_path.";


/*proc sql;
    create table rename_cmds as
    select
        memname,
        cats(
            'proc datasets lib=STDY4A nolist; ',
            'change ', ' ', strip(memname), ' = ',  strip(scan(memname, 1, '_')), 
				'; quit;'
        ) as cmd length=400
    from dictionary.tables
    where upcase(libname) = 'STDY4A'
      and index(memname, '_') > 0;
quit;*/

proc sql;
    create table rename_cmds as
    select
        memname,
        ('proc datasets lib=STUDY nolist; change ' ||
		 strip(memname) || ' = ' ||  strip(scan(memname, 1, '_')) ||
				'; quit;'
        ) as cmd length=400
    from dictionary.tables
    where upcase(libname) = 'STUDY'
      and index(memname, '_') > 0;
quit;

proc print data=rename_cmds noobs;
    title "Review: Dataset Rename Commands for &folder_path";
run;

data _null_;
  set rename_cmds;
   call execute(cmd); 
  put cmd;
run;
