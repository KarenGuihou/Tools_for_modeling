#!/bin/bash

#sty='1980'
#eny='2006'

#yyyy=$sty
#while [ "$yyyy" -lt "$eny" ]; do
#  echo $yyyy
#  $yyyy=`expr $yyyy + 1 `
#done

for yyyy in {1980..2006}; do
  for m in {1..12}; do
    mm=`printf "%02g\n" $m`
    python mat_2_nc.py $yyyy $mm
  done
done
