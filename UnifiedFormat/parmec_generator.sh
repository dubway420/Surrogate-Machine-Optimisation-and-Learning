#! /bin/bash

export nb_comp=$(($(wc -l batch21_7159_P40.0.csv | awk '{print $1}')-1))
export date=$(date +'%Y-%m-%d')
export time=$(date +%F_%H-%M-%S)
export GENERATOR_DIR=`echo $PWD`
export NUMBER_OF_COMP=$nb_comp
export complete_time_vector_array=()
for file_name in $(ls batch21_7159_P40.*.csv); do
  time=$(ls $file_name | awk '{print $2}' FS=.)
  echo $time
  complete_time_vector_array+=( $time )
done

export complete_time_vector_arr=$(echo  ${complete_time_vector_array[@]} | awk '{print }' RS=' ' ORS=',')
echo $complete_time_vector_arr


#Write MetaData
echo "*METADATA 
# date_time_UniForm_file_created: $date
**UniForm_version_number, 1.0 
**coding, ‘utf-8’ 
**scale_factor, 1.
#**experimental_uncertainty, TBC 
# data_source: UoM 
# data_source_type: Parmec
# data_source_contact: huw Jones 
# date_data_created: $date
# time_data_created: $time 
# original_filename: $GENERATOR_DIR
# model_type: Multi Layer Array 
# model_description: Multi Layer Array (MLA8), XX% cracked acetal bricks ">  ./Parmec_Test.csv
  
echo "


"  >>  ./Parmec_Test.csv

echo "*GENERAL_INFORMATION
**NUMBER_OF_COMPonents, $NUMBER_OF_COMP
**coord_system_zero_location, ‘middle_base_of_core’, (0,0,0), ‘m’" >>  ./Parmec_Test.csv
 
echo "


"  >>  ./Parmec_Test.csv


echo "# NON-HOMOGENEOUS TIME DATA 
*TIME_DATA: 
**time_external_input_applied, 0 
**time_vector_type, ‘NON-HOMOGENEOUS’, 5, ‘ms’ 
***complete_time_vector,  ($complete_time_vector_arr)  
" >>  ./Parmec_Test.csv

 
echo "


"  >>  ./Parmec_Test.csv

# Components vector
echo "*COMPONENT_INFORMATION 
**NaN_handling, 'DEFAULT' 
**component_info_vector, 
**component_info_units, (‘m’, ‘m’, ‘m’) "  >>  ./Parmec_Test.csv
python3  read_component.py 
 
echo "


"  >>  ./Parmec_Test.csv





echo "*COMPONENT_DATA: DISPLACEMENT 
**component_data_length,  3 
**component_data_units, (‘m’, ‘m’, ‘m’) " >>  ./Parmec_Test.csv
# Components DATA_DISPLACEMENT
python3  read_component_data.py 
