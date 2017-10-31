#!/bin/ksh
#
## Creation d'un fichier .des pour regarder 
## des séries temporelles annuelles

## $1 path of the files (eg /home/karen/nc/file_*.nc) 
## $2 name of the output file

## variables à modifier - Début
## CASE et variables à changer
VAR='U V T'
delta=1
#year='2006'
#PATH_DATA=$1   #'/media/workspace/VOCES/Runs/NEMO_ORCA'
## Variables à modifier - FIN

  scriptfile=NEMO_$2.des ##"CMM_${var}.des"
  echo $scriptfile
  rm $scriptfile
##rm -f $scriptfile 

cat >> $scriptfile <<EOF
&FORMAT_RECORD
    D_TYPE  ='  MC',
    D_FORMAT       ='  1A',
/
&BACKGROUND_RECORD
    D_EXPNUM       = '0063',
    D_MODNUM       = '  AA',
    D_TITLE        ='CMM monthly 3D fields',
    D_T0TIME       ='01-JAN-1980 00:00:00',
    D_TIME_UNIT    = 86400,
/
&MESSAGE_RECORD
      D_MESSAGE     = ' ',
      D_ALERT_ON_OPEN      = F,
      D_ALERT_ON_OUTPUT    = F,
/
&EXTRA_RECORD
/
EOF
file_nc=`ls $1/*$2.nc |sort` #*_${var}.nc |sort`
echo $1/*$2.nc $file_nc
for file in $file_nc; do
  ncdump -v T $file |grep ' T = [^U]' > tmp
  date1=`cut -f4 -d' ' tmp |cut -f1 -d','`
  date2=`cut -f5 -d' ' tmp |cut -f1 -d','`
  delta=`echo ${date2}-${date1} | bc -l`
  ncdump -h $file |grep 'time = UNLIMITED' > tmp
  nbind=`cut -f6 -d' ' tmp | cut -f2 -d'('`
  nbindm1=`expr "$nbind" - 1`
  #dateend=`expr "$date1" \+ "$delta" \* "$nbindm1"`
  dateend=`echo ${date1}+$delta*$nbindm1 | bc -l`
cat >> $scriptfile <<EOF
&STEPFILE_RECORD
  S_FILENAME     ="$file",
  S_START = $date1,
  S_END   = $date2,
  S_DELTA = $delta,
  S_NUM_OF_FILES = 1,
/
EOF
done
cat >> $scriptfile <<EOF
&STEPFILE_RECORD
    s_filename    = '**END OF STEPFILES**'
/
EOF
