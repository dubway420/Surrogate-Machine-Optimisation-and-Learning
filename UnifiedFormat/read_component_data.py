#! /usr/bin/python3

import os
import re
import math
import os


NB_COMP=os.environ['nb_comp']

for   time_curr in range(int(NB_COMP)):
    line_towrite1='*COMPONENT_DATA: DISPLACEMENT \n'
    line_towrite2='**component_data_length,  3 \n'
    line_towrite3='**component_data_units, (‘mm’, ‘mm’, ‘mm’) \n'
    time_step='***time_step, '+ str(time_curr)+'\n'
    file_to_read="batch21_7159_P40."+str(time_curr)+".csv"
    try:
        file_comp_data = open(file_to_read, "r")
        file_unified_format = open("Parmec_Test.csv", "a")
        file_unified_format.write(line_towrite1)
        file_unified_format.write(line_towrite2)
        file_unified_format.write(line_towrite3)
        file_unified_format.write(time_step)
            
        current_line=0
        for line in file_comp_data:
            if current_line >=1:
                line=re.sub(',',' ',line)
                line_split=line.split()
                DX=float(line_split[0])
                DY=float(line_split[1])
                DZ=float(line_split[2])
                print(DX,DY,DZ)
                to_write=str(DX)+','+str(DY)+','+str(DZ)+','+'\n'
                file_unified_format.write(to_write)
            current_line += 1
    except:
        pass
    
    
    file_unified_format.close()

# ***time_step, 0.000 
# +1.18750e+02,  -1.18750e+02,  +1.03150e+02,
