/*Library Declarations*/
Libname LIBNAME "FILEPATH";

/*Show details in log*/
*ODS Trace on;

/*Select only variable tables*/
ODS OUTPUT variables=allvarout;

/*Get variable tables*/
proc contents data=LIBNAME._all_ memtype=data;
run;

/*Sort data*/
proc sort data=allvarout;
	by member num;
run;

/*Declare output path and options*/
options nobyline;
ODS EXCEL FILE="OUTPUTFILEPATH Variable Info.xlsx"
options(sheet_name="#BYVAL(member)" embedded_titles='yes');

/*Print table*/
proc print data=allvarout noobs;
	by member;
	pageby member;
	title "Variables in #BYVAL(member) data set";
run;

/*Close ODS*/
ODS EXCEL CLOSE;


