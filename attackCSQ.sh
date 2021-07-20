#!/bin/bash

for ((i=1; i<=32; i=i+1))
do
  (
    python patch_attack.py --source_txt ./attack/source2.txt --test_txt ./attack/test2.txt --attack_which $[i*3] --gpu_id 1
  )&

if [ $[$i%2] == 0 ]
then
    wait
fi

done
