#! /usr/bin/python3

import os
import re
import math
import os

NB_COMP=os.environ['nb_comp']
current_line=0
file_comp = open("batch21_7159_P40.0.csv", "r")
file_unified_format = open("Parmec_Test.csv", "a")

for line in file_comp:
    if current_line >=1:
        line=re.sub(',',' ',line)
        line_split=line.split()
        COMP_ID='ID03'
        COMP_NAME='lattice'        
        coorX=float(line_split[-3])
        coorY=float(line_split[-2])
        coorZ=float(line_split[-1])
        print(coorX,coorY,coorZ)
        to_write='component'+str(current_line)+',  03,'+ '"interstitial_brick",'+ '"'+'Lattice_MLA'+str(current_line)+'",'+' ('+str(coorX)+','+str(coorY)+','+str(coorZ)+','+'NaN,NaN,NaN)\n'
        file_unified_format.write(to_write)
    current_line += 1


file_unified_format.close()
