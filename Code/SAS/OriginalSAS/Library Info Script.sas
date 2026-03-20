/*Library Declarations*/
Libname LIBNAME "FILEPATH";

/*Create table of SQL library data*/
proc sql;
 create table LIBINFO as
 select * from dictionary.tables
 where libname="LIBNAME";
quit;

/*Count number of datasets*/
proc sql;
 select count(memname) label="# of datasets in LIBNAME" from dictionary.tables
  where libname="LIBNAME";
quit;

/*Uncomment to create Excel output file*/
/*ods excel file="OUTPUTFILEPATH.xlsx"; */

/*Print table*/
proc print data=LIBINFO;
run; 
