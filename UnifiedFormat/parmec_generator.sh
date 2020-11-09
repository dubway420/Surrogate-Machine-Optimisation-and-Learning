#! /bin/bash

nb_comp=$(($(wc -l batch21_7159_P40.0.csv | awk '{print $1}')-1))
date=$(date +'%Y-%m-%d')
time=$(date +%F_%H-%M-%S)
GENERATOR_DIR=`echo $PWD`
NUMBER_OF_COMP=$nb_comp
allThreads=(1 2 4 8 16 32 64 128)
allRuntimes=()
for t in ${allThreads[@]}; do
  runtime=$(./pipeline --threads $t)
  allRuntimes+=( $runtime )
done


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
  

echo "*GENERAL_INFORMATION
**NUMBER_OF_COMPonents, $NUMBER_OF_COMP
**coord_system_zero_location, ‘middle_base_of_core’, (0,0,0), ‘m’" >>  ./Parmec_Test.csv


echo "# NON-HOMOGENEOUS TIME DATA 
*TIME_DATA: 
**time_external_input_applied, 0 
**time_vector_type, ‘NON-HOMOGENEOUS’, 5, ‘ms’ 
***complete_time_vector,  (0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080, 0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115, 0.120, 0.125, 0.130, 0.135, 0.140, 0.145, 0.150, 0.155, 0.160, 0.165, 0.170, 0.175, 0.180, 0.185, 0.190, 0.195, 0.200, 0.205, 0.210, 0.215, 0.220, 0.225, 0.230, 0.235, 0.240, 0.245, 0.250, 0.255, 0.260, 0.265, 0.270, 0.275, 0.280, 0.285, 0.290, 0.295, 0.300, 0.305, 0.310, 0.315, 0.320, 0.325, 0.330, 0.335, 0.340, 0.345, 0.350, 0.355, 0.360, 0.365, 0.370, 0.375, 0.380, 0.385, 0.390, 0.395, 0.400, 0.405, 0.410, 0.415, 0.420, 0.425, 0.430, 0.435, 0.440, 0.445, 0.450, 0.455, 0.460, 0.465, 0.470, 0.475, 0.480, 0.485, 0.490, 0.495, 0.500, 0.505, 0.510, 0.515, 0.520, 0.525, 0.530, 0.535, 0.540, 0.545, 0.550, 0.555, 0.560, 0.565, 0.570, 0.575, 0.580, 0.585, 0.590, 0.595, 0.600, 0.605, 0.610, 0.615, 0.620, 0.625, 0.630, 0.635, 0.640, 0.645, 0.650, 0.655, 0.660, 0.665, 0.670, 0.675, 0.680, 0.685, 0.690, 0.695, 0.700, 0.705, 0.710, 0.715, 0.720, 0.725, 0.730, 0.735, 0.740, 0.745, 0.750, 0.755, 0.760, 0.765, 0.770, 0.775, 0.780, 0.785, 0.790, 0.795, 0.800, 0.805, 0.810, 0.815, 0.820, 0.825, 0.830, 0.835, 0.840, 0.845, 0.850, 0.855, 0.860, 0.865, 0.870, 0.875, 0.880, 0.885, 0.890, 0.895, 0.900, 0.905, 0.910, 0.915, 0.920, 0.925, 0.930, 0.935, 0.940, 0.945, 0.950, 0.955, 0.960, 0.965, 0.970, 0.975, 0.980, 0.985, 0.990, 0.995, 1.000, 1.005, 1.010, 1.015, 1.020, 1.025, 1.030, 1.035, 1.040, 1.045, 1.050, 1.055, 1.060, 1.065, 1.070, 1.075, 1.080, 1.085, 1.090, 1.095, 1.100, 1.105, 1.110, 1.115, 1.120, 1.125, 1.130, 1.135, 1.140, 1.145, 1.150, 1.155, 1.160, 1.165, 1.170, 1.175, 1.180, 1.185, 1.190, 1.195, 1.200, 1.205, 1.210, 1.215, 1.220, 1.225, 1.230, 1.235, 1.240, 1.245, 1.250, 1.255, 1.260, 1.265, 1.270, 1.275, 1.280, 1.285, 1.290, 1.295, 1.300, 1.305, 1.310, 1.315, 1.320, 1.325, 1.330, 1.335, 1.340, 1.345, 1.350, 1.355, 1.360, 1.365, 1.370, 1.375, 1.380, 1.385, 1.390, 1.395, 1.400, 1.405, 1.410, 1.415, 1.420, 1.425, 1.430, 1.435, 1.440, 1.445, 1.450, 1.455, 1.460, 1.465, 1.470, 1.475, 1.480, 1.485, 1.490, 1.495, 1.500, 1.505, 1.510, 1.515, 1.520, 1.525, 1.530, 1.535, 1.540, 1.545, 1.550, 1.555, 1.560, 1.565, 1.570, 1.575, 1.580, 1.585, 1.590, 1.595, 1.600, 1.605, 1.610, 1.615, 1.620, 1.625, 1.630, 1.635, 1.640, 1.645, 1.650, 1.655, 1.660, 1.665, 1.670, 1.675, 1.680, 1.685, 1.690, 1.695, 1.700, 1.705, 1.710, 1.715, 1.720, 1.725, 1.730, 1.735, 1.740, 1.745, 1.750, 1.755, 1.760, 1.765, 1.770, 1.775, 1.780, 1.785, 1.790, 1.795, 1.800, 1.805, 1.810, 1.815, 1.820, 1.825, 1.830, 1.835, 1.840, 1.845, 1.850, 1.855, 1.860, 1.865, 1.870, 1.875, 1.880, 1.885, 1.890, 1.895, 1.900, 1.905, 1.910, 1.915, 1.920, 1.925, 1.930, 1.935, 1.940, 1.945, 1.950, 1.955, 1.960, 1.965, 1.970, 1.975, 1.980, 1.985, 1.990, 1.995, 2.000, 2.005, 2.010, 2.015, 2.020, 2.025, 2.030, 2.035, 2.040, 2.045, 2.050, 2.055, 2.060, 2.065, 2.070, 2.075, 2.080, 2.085, 2.090, 2.095, 2.100, 2.105, 2.110, 2.115, 2.120, 2.125, 2.130, 2.135, 2.140, 2.145, 2.150, 2.155, 2.160, 2.165, 2.170, 2.175, 2.180, 2.185, 2.190, 2.195, 2.200, 2.205, 2.210, 2.215, 2.220, 2.225, 2.230, 2.235, 2.240, 2.245, 2.250, 2.255, 2.260, 2.265, 2.270, 2.275, 2.280, 2.285, 2.290, 2.295, 2.300, 2.305, 2.310, 2.315, 2.320, 2.325, 2.330, 2.335, 2.340, 2.345, 2.350, 2.355, 2.360, 2.365, 2.370, 2.375, 2.380, 2.385, 2.390, 2.395, 2.400, 2.405, 2.410, 2.415, 2.420, 2.425, 2.430, 2.435, 2.440, 2.445, 2.450, 2.455, 2.460, 2.465, 2.470, 2.475, 2.480, 2.485, 2.490, 2.495, 2.500, 2.505, 2.510, 2.515, 2.520, 2.525, 2.530, 2.535, 2.540, 2.545, 2.550, 2.555, 2.560, 2.565, 2.570, 2.575, 2.580, 2.585, 2.590, 2.595, 2.600, 2.605, 2.610, 2.615, 2.620, 2.625, 2.630, 2.635, 2.640, 2.645, 2.650, 2.655, 2.660, 2.665, 2.670, 2.675, 2.680, 2.685, 2.690, 2.695, 2.700, 2.705, 2.710, 2.715, 2.720, 2.725, 2.730, 2.735, 2.740, 2.745, 2.750, 2.755, 2.760, 2.765, 2.770, 2.775, 2.780, 2.785, 2.790, 2.795, 2.800, 2.805, 2.810, 2.815, 2.820, 2.825, 2.830, 2.835, 2.840, 2.845, 2.850, 2.855, 2.860, 2.865, 2.870, 2.875, 2.880, 2.885, 2.890, 2.895, 2.900, 2.905, 2.910, 2.915, 2.920, 2.925, 2.930, 2.935, 2.940, 2.945, 2.950, 2.955, 2.960, 2.965, 2.970, 2.975, 2.980, 2.985, 2.990, 2.995, 3.000, 3.005, 3.010, 3.015, 3.020, 3.025, 3.030, 3.035, 3.040, 3.045, 3.050, 3.055, 3.060, 3.065, 3.070, 3.075, 3.080, 3.085, 3.090, 3.095, 3.100, 3.105, 3.110, 3.115, 3.120, 3.125, 3.130, 3.135, 3.140, 3.145, 3.150, 3.155, 3.160, 3.165, 3.170, 3.175, 3.180, 3.185, 3.190, 3.195, 3.200, 3.205, 3.210, 3.215, 3.220, 3.225, 3.230, 3.235, 3.240, 3.245, 3.250, 3.255, 3.260, 3.265, 3.270, 3.275, 3.280, 3.285, 3.290, 3.295, 3.300, 3.305, 3.310, 3.315, 3.320, 3.325, 3.330, 3.335, 3.340, 3.345, 3.350, 3.355, 3.360, 3.365, 3.370, 3.375, 3.380, 3.385, 3.390, 3.395, 3.400, 3.405, 3.410, 3.415, 3.420, 3.425, 3.430, 3.435, 3.440, 3.445, 3.450, 3.455, 3.460, 3.465, 3.470, 3.475, 3.480, 3.485, 3.490, 3.495, 3.500, 3.505, 3.510, 3.515, 3.520, 3.525, 3.530, 3.535, 3.540, 3.545, 3.550, 3.555, 3.560, 3.565, 3.570, 3.575, 3.580, 3.585, 3.590, 3.595, 3.600, 3.605, 3.610, 3.615, 3.620, 3.625, 3.630, 3.635, 3.640, 3.645, 3.650, 3.655, 3.660, 3.665, 3.670, 3.675, 3.680, 3.685, 3.690, 3.695, 3.700, 3.705, 3.710, 3.715, 3.720, 3.725, 3.730, 3.735, 3.740, 3.745, 3.750, 3.755, 3.760, 3.765, 3.770, 3.775, 3.780, 3.785, 3.790, 3.795, 3.800, 3.805, 3.810, 3.815, 3.820, 3.825, 3.830, 3.835, 3.840, 3.845, 3.850, 3.855, 3.860, 3.865, 3.870, 3.875, 3.880, 3.885, 3.890, 3.895, 3.900, 3.905, 3.910, 3.915, 3.920, 3.925, 3.930, 3.935, 3.940, 3.945, 3.950, 3.955, 3.960, 3.965, 3.970, 3.975, 3.980, 3.985, 3.990, 3.995, 4.000, 4.005, 4.010, 4.015, 4.020, 4.025, 4.030, 4.035, 4.040, 4.045, 4.050, 4.055, 4.060, 4.065, 4.070, 4.075, 4.080, 4.085, 4.090, 4.095, 4.100, 4.105, 4.110, 4.115, 4.120, 4.125, 4.130, 4.135, 4.140, 4.145, 4.150, 4.155, 4.160, 4.165, 4.170, 4.175, 4.180, 4.185, 4.190, 4.195, 4.200, 4.205, 4.210, 4.215, 4.220, 4.225, 4.230, 4.235, 4.240, 4.245, 4.250, 4.255, 4.260, 4.265, 4.270, 4.275, 4.280, 4.285, 4.290, 4.295, 4.300, 4.305, 4.310, 4.315, 4.320, 4.325, 4.330, 4.335, 4.340, 4.345, 4.350, 4.355, 4.360, 4.365, 4.370, 4.375, 4.380, 4.385, 4.390, 4.395, 4.400, 4.405, 4.410, 4.415, 4.420, 4.425, 4.430, 4.435, 4.440, 4.445, 4.450, 4.455, 4.460, 4.465, 4.470, 4.475, 4.480, 4.485, 4.490, 4.495, 4.500, 4.505, 4.510, 4.515, 4.520, 4.525, 4.530, 4.535, 4.540, 4.545, 4.550, 4.555, 4.560, 4.565, 4.570, 4.575, 4.580, 4.585, 4.590, 4.595, 4.600, 4.605, 4.610, 4.615, 4.620, 4.625, 4.630, 4.635, 4.640, 4.645, 4.650, 4.655, 4.660, 4.665, 4.670, 4.675, 4.680, 4.685, 4.690, 4.695, 4.700, 4.705, 4.710, 4.715, 4.720, 4.725, 4.730, 4.735, 4.740, 4.745, 4.750, 4.755, 4.760, 4.765, 4.770, 4.775, 4.780, 4.785, 4.790, 4.795, 4.800, 4.805, 4.810, 4.815, 4.820, 4.825, 4.830, 4.835, 4.840, 4.845, 4.850, 4.855, 4.860, 4.865, 4.870, 4.875, 4.880, 4.885, 4.890, 4.895, 4.900, 4.905, 4.910, 4.915, 4.920, 4.925, 4.930, 4.935, 4.940, 4.945, 4.950, 4.955, 4.960, 4.965, 4.970, 4.975, 4.980, 4.985, 4.990, 4.995, 5.000, 5.005, 5.010, 5.015, 5.020, 5.025, 5.030, 5.035, 5.040, 5.045, 5.050, 5.055, 5.060, 5.065, 5.070, 5.075, 5.080, 5.085, 5.090, 5.095, 5.100, 5.105, 5.110, 5.115, 5.120, 5.125, 5.130, 5.135, 5.140, 5.145, 5.150, 5.155, 5.160, 5.165, 5.170, 5.175, 5.180, 5.185, 5.190, 5.195, 5.200, 5.205, 5.210, 5.215, 5.220, 5.225, 5.230, 5.235, 5.240, 5.245, 5.250, 5.255, 5.260, 5.265, 5.270, 5.275, 5.280, 5.285, 5.290, 5.295, 5.300, 5.305, 5.310, 5.315, 5.320, 5.325, 5.330, 5.335, 5.340, 5.345, 5.350, 5.355, 5.360, 5.365, 5.370, 5.375, 5.380, 5.385, 5.390, 5.395, 5.400, 5.405, 5.410, 5.415, 5.420, 5.425, 5.430, 5.435, 5.440, 5.445, 5.450, 5.455, 5.460, 5.465, 5.470, 5.475, 5.480, 5.485, 5.490, 5.495, 5.500, 5.505, 5.510, 5.515, 5.520, 5.525, 5.530, 5.535, 5.540, 5.545, 5.550, 5.555, 5.560, 5.565, 5.570, 5.575, 5.580, 5.585, 5.590, 5.595, 5.600, 5.605, 5.610, 5.615, 5.620, 5.625, 5.630, 5.635, 5.640, 5.645, 5.650, 5.655, 5.660, 5.665, 5.670, 5.675, 5.680, 5.685, 5.690, 5.695, 5.700, 5.705, 5.710, 5.715, 5.720, 5.725, 5.730, 5.735, 5.740, 5.745, 5.750, 5.755, 5.760, 5.765, 5.770, 5.775, 5.780, 5.785, 5.790, 5.795, 5.800, 5.805, 5.810, 5.815, 5.820, 5.825, 5.830, 5.835, 5.840, 5.845, 5.850, 5.855, 5.860, 5.865, 5.870, 5.875, 5.880, 5.885, 5.890, 5.895, 5.900, 5.905, 5.910, 5.915, 5.920, 5.925, 5.930, 5.935, 5.940, 5.945, 5.950, 5.955, 5.960, 5.965, 5.970, 5.975, 5.980, 5.985, 5.990, 5.995, 6.000, 6.005, 6.010, 6.015, 6.020, 6.025, 6.030, 6.035, 6.040, 6.045, 6.050, 6.055, 6.060, 6.065, 6.070, 6.075, 6.080, 6.085, 6.090, 6.095, 6.100, 6.105, 6.110, 6.115, 6.120, 6.125, 6.130, 6.135, 6.140, 6.145, 6.150, 6.155, 6.160, 6.165, 6.170, 6.175, 6.180, 6.185, 6.190, 6.195, 6.200, 6.205, 6.210, 6.215, 6.220, 6.225, 6.230, 6.235, 6.240, 6.245, 6.250, 6.255, 6.260, 6.265, 6.270, 6.275, 6.280, 6.285, 6.290, 6.295, 6.300, 6.305, 6.310, 6.315, 6.320, 6.325, 6.330, 6.335, 6.340, 6.345, 6.350, 6.355, 6.360, 6.365, 6.370, 6.375, 6.380, 6.385, 6.390, 6.395, 6.400, 6.405, 6.410, 6.415, 6.420, 6.425, 6.430, 6.435, 6.440, 6.445, 6.450, 6.455, 6.460, 6.465, 6.470, 6.475, 6.480, 6.485, 6.490, 6.495, 6.500, 6.505, 6.510, 6.515, 6.520, 6.525, 6.530, 6.535, 6.540, 6.545, 6.550, 6.555, 6.560, 6.565, 6.570, 6.575, 6.580, 6.585, 6.590, 6.595, 6.600, 6.605, 6.610, 6.615, 6.620, 6.625, 6.630, 6.635, 6.640, 6.645, 6.650, 6.655, 6.660, 6.665, 6.670, 6.675, 6.680, 6.685, 6.690, 6.695, 6.700, 6.705, 6.710, 6.715, 6.720, 6.725, 6.730, 6.735, 6.740, 6.745, 6.750, 6.755, 6.760, 6.765, 6.770, 6.775, 6.780, 6.785, 6.790, 6.795, 6.800, 6.805, 6.810, 6.815, 6.820, 6.825, 6.830, 6.835, 6.840, 6.845, 6.850, 6.855, 6.860, 6.865, 6.870, 6.875, 6.880, 6.885, 6.890, 6.895, 6.900, 6.905, 6.910, 6.915, 6.920, 6.925, 6.930, 6.935, 6.940, 6.945, 6.950, 6.955, 6.960, 6.965, 6.970, 6.975, 6.980, 6.985, 6.990, 6.995, 7.000, 7.005, 7.010, 7.015, 7.020, 7.025, 7.030, 7.035, 7.040, 7.045, 7.050, 7.055, 7.060, 7.065, 7.070, 7.075, 7.080, 7.085, 7.090, 7.095, 7.100, 7.105, 7.110, 7.115, 7.120, 7.125, 7.130, 7.135, 7.140, 7.145, 7.150, 7.155, 7.160, 7.165, 7.170, 7.175, 7.180, 7.185, 7.190, 7.195, 7.200, 7.205, 7.210, 7.215, 7.220, 7.225, 7.230, 7.235, 7.240, 7.245, 7.250, 7.255, 7.260, 7.265, 7.270, 7.275, 7.280, 7.285, 7.290, 7.295, 7.300, 7.305, 7.310, 7.315, 7.320, 7.325, 7.330, 7.335, 7.340, 7.345, 7.350, 7.355, 7.360, 7.365, 7.370, 7.375, 7.380, 7.385, 7.390, 7.395, 7.400, 7.405, 7.410, 7.415, 7.420, 7.425, 7.430, 7.435, 7.440, 7.445, 7.450, 7.455, 7.460, 7.465, 7.470, 7.475, 7.480, 7.485, 7.490, 7.495, 7.500, 7.505, 7.510, 7.515, 7.520, 7.525, 7.530, 7.535, 7.540, 7.545, 7.550, 7.555, 7.560, 7.565, 7.570, 7.575, 7.580, 7.585, 7.590, 7.595, 7.600, 7.605, 7.610, 7.615, 7.620, 7.625, 7.630, 7.635, 7.640, 7.645, 7.650, 7.655, 7.660, 7.665, 7.670, 7.675, 7.680, 7.685, 7.690, 7.695, 7.700, 7.705, 7.710, 7.715, 7.720, 7.725, 7.730, 7.735, 7.740, 7.745, 7.750, 7.755, 7.760, 7.765, 7.770, 7.775, 7.780, 7.785, 7.790, 7.795, 7.800, 7.805, 7.810, 7.815, 7.820, 7.825, 7.830, 7.835, 7.840, 7.845, 7.850, 7.855, 7.860, 7.865, 7.870, 7.875, 7.880, 7.885, 7.890, 7.895, 7.900, 7.905, 7.910, 7.915, 7.920, 7.925, 7.930, 7.935, 7.940, 7.945, 7.950, 7.955, 7.960, 7.965, 7.970, 7.975, 7.980, 7.985, 7.990, 7.995, 8.000, 8.005, 8.010, 8.015, 8.020, 8.025, 8.030, 8.035, 8.040, 8.045, 8.050, 8.055, 8.060, 8.065, 8.070, 8.075, 8.080, 8.085, 8.090, 8.095, 8.100, 8.105, 8.110, 8.115, 8.120, 8.125, 8.130, 8.135, 8.140, 8.145, 8.150, 8.155, 8.160, 8.165, 8.170, 8.175, 8.180, 8.185, 8.190, 8.195, 8.200, 8.205, 8.210, 8.215, 8.220, 8.225, 8.230, 8.235, 8.240, 8.245, 8.250, 8.255, 8.260, 8.265, 8.270, 8.275, 8.280, 8.285, 8.290, 8.295, 8.300, 8.305, 8.310, 8.315, 8.320, 8.325, 8.330, 8.335, 8.340, 8.345, 8.350, 8.355, 8.360, 8.365, 8.370, 8.375, 8.380, 8.385, 8.390, 8.395, 8.400, 8.405, 8.410, 8.415, 8.420, 8.425, 8.430, 8.435, 8.440, 8.445, 8.450, 8.455, 8.460, 8.465, 8.470, 8.475, 8.480, 8.485, 8.490, 8.495, 8.500, 8.505, 8.510, 8.515, 8.520, 8.525, 8.530, 8.535, 8.540, 8.545, 8.550, 8.555, 8.560, 8.565, 8.570, 8.575, 8.580, 8.585, 8.590, 8.595, 8.600, 8.605, 8.610, 8.615, 8.620, 8.625, 8.630, 8.635, 8.640, 8.645, 8.650, 8.655, 8.660, 8.665, 8.670, 8.675, 8.680, 8.685, 8.690, 8.695, 8.700, 8.705, 8.710, 8.715, 8.720, 8.725, 8.730, 8.735, 8.740, 8.745, 8.750, 8.755, 8.760, 8.765, 8.770, 8.775, 8.780, 8.785, 8.790, 8.795, 8.800, 8.805, 8.810, 8.815, 8.820, 8.825, 8.830, 8.835, 8.840, 8.845, 8.850, 8.855, 8.860, 8.865, 8.870, 8.875, 8.880, 8.885, 8.890, 8.895, 8.900, 8.905, 8.910, 8.915, 8.920, 8.925, 8.930, 8.935, 8.940, 8.945, 8.950, 8.955, 8.960, 8.965, 8.970, 8.975, 8.980, 8.985, 8.990, 8.995, 9.000, 9.005, 9.010, 9.015, 9.020, 9.025, 9.030, 9.035, 9.040, 9.045, 9.050, 9.055, 9.060, 9.065, 9.070, 9.075, 9.080, 9.085, 9.090, 9.095, 9.100, 9.105, 9.110, 9.115, 9.120, 9.125, 9.130, 9.135, 9.140, 9.145, 9.150, 9.155, 9.160, 9.165, 9.170, 9.175, 9.180, 9.185, 9.190, 9.195, 9.200, 9.205, 9.210, 9.215, 9.220, 9.225, 9.230, 9.235, 9.240, 9.245, 9.250, 9.255, 9.260, 9.265, 9.270, 9.275, 9.280, 9.285, 9.290, 9.295, 9.300, 9.305, 9.310, 9.315, 9.320, 9.325, 9.330, 9.335, 9.340, 9.345, 9.350, 9.355, 9.360, 9.365, 9.370, 9.375, 9.380, 9.385, 9.390, 9.395, 9.400, 9.405, 9.410, 9.415, 9.420, 9.425, 9.430, 9.435, 9.440, 9.445, 9.450, 9.455, 9.460, 9.465, 9.470, 9.475, 9.480, 9.485, 9.490, 9.495, 9.500, 9.505, 9.510, 9.515, 9.520, 9.525, 9.530, 9.535, 9.540, 9.545, 9.550, 9.555, 9.560, 9.565, 9.570, 9.575, 9.580, 9.585, 9.590, 9.595, 9.600, 9.605, 9.610, 9.615, 9.620, 9.625, 9.630, 9.635, 9.640, 9.645, 9.650, 9.655, 9.660, 9.665, 9.670, 9.675, 9.680, 9.685, 9.690, 9.695, 9.700, 9.705, 9.710, 9.715, 9.720, 9.725, 9.730, 9.735, 9.740, 9.745, 9.750, 9.755, 9.760, 9.765, 9.770, 9.775, 9.780, 9.785, 9.790, 9.795, 9.800, 9.805, 9.810, 9.815, 9.820, 9.825, 9.830, 9.835, 9.840, 9.845, 9.850, 9.855, 9.860, 9.865, 9.870, 9.875, 9.880, 9.885, 9.890, 9.895, 9.900, 9.905, 9.910, 9.915, 9.920, 9.925, 9.930, 9.935, 9.940, 9.945, 9.950, 9.955, 9.960, 9.965, 9.970, 9.975, 9.980, 9.985, 9.990, 9.995, 10.000, 10.005, 10.010, 10.015, 10.020, 10.025, 10.030, 10.035, 10.040, 10.045, 10.050, 10.055, 10.060, 10.065, 10.070, 10.075, 10.080, 10.085, 10.090, 10.095, 10.100, 10.105, 10.110, 10.115, 10.120, 10.125, 10.130, 10.135, 10.140, 10.145, 10.150, 10.155, 10.160, 10.165, 10.170, 10.175, 10.180, 10.185, 10.190, 10.195, 10.200, 10.205, 10.210, 10.215, 10.220, 10.225, 10.230, 10.235, 10.240, 10.245, 10.250, 10.255, 10.260, 10.265, 10.270, 10.275, 10.280, 10.285, 10.290, 10.295, 10.300, 10.305, 10.310, 10.315, 10.320, 10.325, 10.330, 10.335, 10.340, 10.345, 10.350, 10.355, 10.360, 10.365, 10.370, 10.375, 10.380, 10.385, 10.390, 10.395, 10.400, 10.405, 10.410, 10.415, 10.420, 10.425, 10.430, 10.435, 10.440, 10.445, 10.450, 10.455, 10.460, 10.465, 10.470, 10.475, 10.480, 10.485, 10.490, 10.495, 10.500, 10.505, 10.510, 10.515, 10.520, 10.525, 10.530, 10.535, 10.540, 10.545, 10.550, 10.555, 10.560, 10.565, 10.570, 10.575, 10.580, 10.585, 10.590, 10.595, 10.600, 10.605, 10.610, 10.615, 10.620, 10.625, 10.630, 10.635, 10.640, 10.645, 10.650, 10.655, 10.660, 10.665, 10.670, 10.675, 10.680, 10.685, 10.690, 10.695, 10.700, 10.705, 10.710, 10.715, 10.720, 10.725, 10.730, 10.735, 10.740, 10.745, 10.750, 10.755, 10.760, 10.765, 10.770, 10.775, 10.780, 10.785, 10.790, 10.795, 10.800, 10.805, 10.810, 10.815, 10.820, 10.825, 10.830, 10.835, 10.840, 10.845, 10.850, 10.855, 10.860, 10.865, 10.870, 10.875, 10.880, 10.885, 10.890, 10.895, 10.900, 10.905, 10.910, 10.915, 10.920, 10.925, 10.930, 10.935, 10.940, 10.945, 10.950, 10.955, 10.960, 10.965, 10.970, 10.975, 10.980, 10.985, 10.990, 10.995, 11.000, 11.005, 11.010, 11.015, 11.020, 11.025, 11.030, 11.035, 11.040, 11.045, 11.050, 11.055, 11.060, 11.065, 11.070, 11.075, 11.080, 11.085, 11.090, 11.095, 11.100, 11.105, 11.110, 11.115, 11.120, 11.125, 11.130, 11.135, 11.140, 11.145, 11.150, 11.155, 11.160, 11.165, 11.170, 11.175, 11.180, 11.185, 11.190, 11.195, 11.200, 11.205, 11.210, 11.215, 11.220, 11.225, 11.230, 11.235, 11.240, 11.245, 11.250, 11.255, 11.260, 11.265, 11.270, 11.275, 11.280, 11.285, 11.290, 11.295, 11.300, 11.305, 11.310, 11.315, 11.320, 11.325, 11.330, 11.335, 11.340, 11.345, 11.350, 11.355, 11.360, 11.365, 11.370, 11.375, 11.380, 11.385, 11.390, 11.395, 11.400, 11.405, 11.410, 11.415, 11.420, 11.425, 11.430, 11.435, 11.440, 11.445, 11.450, 11.455, 11.460, 11.465, 11.470, 11.475, 11.480, 11.485, 11.490, 11.495, 11.500, 11.505, 11.510, 11.515, 11.520, 11.525, 11.530, 11.535, 11.540, 11.545, 11.550, 11.555, 11.560, 11.565, 11.570, 11.575, 11.580, 11.585, 11.590, 11.595, 11.600, 11.605, 11.610, 11.615, 11.620, 11.625, 11.630, 11.635, 11.640, 11.645, 11.650, 11.655, 11.660, 11.665, 11.670, 11.675, 11.680, 11.685, 11.690, 11.695, 11.700, 11.705, 11.710, 11.715, 11.720, 11.725, 11.730, 11.735, 11.740, 11.745, 11.750, 11.755, 11.760, 11.765, 11.770, 11.775, 11.780, 11.785, 11.790, 11.795, 11.800, 11.805, 11.810, 11.815, 11.820, 11.825, 11.830, 11.835, 11.840, 11.845, 11.850, 11.855, 11.860, 11.865, 11.870, 11.875, 11.880, 11.885, 11.890, 11.895, 11.900, 11.905, 11.910, 11.915, 11.920, 11.925, 11.930, 11.935, 11.940, 11.945, 11.950, 11.955, 11.960, 11.965, 11.970, 11.975, 11.980, 11.985, 11.990, 11.995, 12.000, 12.005, 12.010, 12.015, 12.020, 12.025, 12.030, 12.035, 12.040, 12.045, 12.050, 12.055, 12.060, 12.065, 12.070, 12.075, 12.080, 12.085, 12.090, 12.095, 12.100, 12.105, 12.110, 12.115, 12.120, 12.125, 12.130, 12.135, 12.140, 12.145, 12.150, 12.155, 12.160, 12.165, 12.170, 12.175, 12.180, 12.185, 12.190, 12.195, 12.200, 12.205, 12.210, 12.215, 12.220, 12.225, 12.230, 12.235, 12.240, 12.245, 12.250, 12.255, 12.260, 12.265, 12.270, 12.275, 12.280, 12.285, 12.290, 12.295, 12.300, 12.305, 12.310, 12.315, 12.320, 12.325, 12.330, 12.335, 12.340, 12.345, 12.350, 12.355, 12.360, 12.365, 12.370, 12.375, 12.380, 12.385, 12.390, 12.395, 12.400, 12.405, 12.410, 12.415, 12.420, 12.425, 12.430, 12.435, 12.440, 12.445, 12.450, 12.455, 12.460, 12.465, 12.470, 12.475, 12.480, 12.485, 12.490, 12.495, 12.500, 12.505, 12.510, 12.515, 12.520, 12.525, 12.530, 12.535, 12.540, 12.545, 12.550, 12.555, 12.560, 12.565, 12.570, 12.575, 12.580, 12.585, 12.590, 12.595, 12.600, 12.605, 12.610, 12.615, 12.620, 12.625, 12.630, 12.635, 12.640, 12.645, 12.650, 12.655, 12.660, 12.665, 12.670, 12.675, 12.680, 12.685, 12.690, 12.695, 12.700, 12.705, 12.710, 12.715, 12.720, 12.725, 12.730, 12.735, 12.740, 12.745, 12.750, 12.755, 12.760, 12.765, 12.770, 12.775, 12.780, 12.785, 12.790, 12.795, 12.800, 12.805, 12.810, 12.815, 12.820, 12.825, 12.830, 12.835, 12.840, 12.845, 12.850, 12.855, 12.860, 12.865, 12.870, 12.875, 12.880, 12.885, 12.890, 12.895, 12.900, 12.905, 12.910, 12.915, 12.920, 12.925, 12.930, 12.935, 12.940, 12.945, 12.950, 12.955, 12.960, 12.965, 12.970, 12.975, 12.980, 12.985, 12.990, 12.995, 13.000, 13.005, 13.010, 13.015, 13.020, 13.025, 13.030, 13.035, 13.040, 13.045, 13.050, 13.055, 13.060, 13.065, 13.070, 13.075, 13.080, 13.085, 13.090, 13.095, 13.100, 13.105, 13.110, 13.115, 13.120, 13.125, 13.130, 13.135, 13.140, 13.145, 13.150, 13.155, 13.160, 13.165, 13.170, 13.175, 13.180, 13.185, 13.190, 13.195, 13.200, 13.205, 13.210, 13.215, 13.220, 13.225, 13.230, 13.235, 13.240, 13.245, 13.250, 13.255, 13.260, 13.265, 13.270, 13.275, 13.280, 13.285, 13.290, 13.295, 13.300, 13.305, 13.310, 13.315, 13.320, 13.325, 13.330, 13.335, 13.340, 13.345, 13.350, 13.355, 13.360, 13.365, 13.370, 13.375, 13.380, 13.385, 13.390, 13.395, 13.400, 13.405, 13.410, 13.415, 13.420, 13.425, 13.430, 13.435, 13.440, 13.445, 13.450, 13.455, 13.460, 13.465, 13.470, 13.475, 13.480, 13.485, 13.490, 13.495, 13.500, 13.505, 13.510, 13.515, 13.520, 13.525, 13.530, 13.535, 13.540, 13.545, 13.550, 13.555, 13.560, 13.565, 13.570, 13.575, 13.580, 13.585, 13.590, 13.595, 13.600, 13.605, 13.610, 13.615, 13.620, 13.625, 13.630, 13.635, 13.640, 13.645, 13.650, 13.655, 13.660, 13.665, 13.670, 13.675, 13.680, 13.685, 13.690, 13.695, 13.700, 13.705, 13.710, 13.715, 13.720, 13.725, 13.730, 13.735, 13.740, 13.745, 13.750, 13.755, 13.760, 13.765, 13.770, 13.775, 13.780, 13.785, 13.790, 13.795, 13.800, 13.805, 13.810, 13.815, 13.820, 13.825, 13.830, 13.835, 13.840, 13.845, 13.850, 13.855, 13.860, 13.865, 13.870, 13.875, 13.880, 13.885, 13.890, 13.895, 13.900, 13.905, 13.910, 13.915, 13.920, 13.925, 13.930, 13.935, 13.940, 13.945, 13.950, 13.955, 13.960, 13.965, 13.970, 13.975, 13.980, 13.985, 13.990, 13.995, 14.000, 14.005, 14.010, 14.015, 14.020, 14.025, 14.030, 14.035, 14.040, 14.045, 14.050, 14.055, 14.060, 14.065, 14.070, 14.075, 14.080, 14.085, 14.090, 14.095, 14.100, 14.105, 14.110, 14.115, 14.120, 14.125, 14.130, 14.135, 14.140, 14.145, 14.150, 14.155, 14.160, 14.165, 14.170, 14.175, 14.180, 14.185, 14.190, 14.195, 14.200, 14.205, 14.210, 14.215, 14.220, 14.225, 14.230, 14.235, 14.240, 14.245, 14.250, 14.255, 14.260, 14.265, 14.270, 14.275, 14.280, 14.285, 14.290, 14.295, 14.300, 14.305, 14.310, 14.315, 14.320, 14.325, 14.330, 14.335, 14.340, 14.345, 14.350, 14.355, 14.360, 14.365, 14.370, 14.375, 14.380, 14.385, 14.390, 14.395, 14.400, 14.405, 14.410, 14.415, 14.420, 14.425, 14.430, 14.435, 14.440, 14.445, 14.450, 14.455, 14.460, 14.465, 14.470, 14.475, 14.480, 14.485, 14.490, 14.495, 14.500, 14.505, 14.510, 14.515, 14.520, 14.525, 14.530, 14.535, 14.540, 14.545, 14.550, 14.555, 14.560, 14.565, 14.570, 14.575, 14.580, 14.585, 14.590, 14.595, 14.600, 14.605, 14.610, 14.615, 14.620, 14.625, 14.630, 14.635, 14.640, 14.645, 14.650, 14.655, 14.660, 14.665, 14.670, 14.675, 14.680, 14.685, 14.690, 14.695, 14.700, 14.705, 14.710, 14.715, 14.720, 14.725, 14.730, 14.735, 14.740, 14.745, 14.750, 14.755, 14.760, 14.765, 14.770, 14.775, 14.780, 14.785, 14.790, 14.795, 14.800, 14.805, 14.810, 14.815, 14.820, 14.825, 14.830, 14.835, 14.840, 14.845, 14.850, 14.855, 14.860, 14.865, 14.870, 14.875, 14.880, 14.885, 14.890, 14.895, 14.900, 14.905, 14.910, 14.915, 14.920, 14.925, 14.930, 14.935, 14.940, 14.945, 14.950, 14.955, 14.960, 14.965, 14.970, 14.975, 14.980, 14.985, 14.990, 14.995, 15.000, 15.005, 15.010, 15.015, 15.020, 15.025, 15.030, 15.035, 15.040, 15.045, 15.050, 15.055, 15.060, 15.065, 15.070, 15.075, 15.080, 15.085, 15.090, 15.095, 15.100, 15.105, 15.110, 15.115, 15.120, 15.125, 15.130, 15.135, 15.140, 15.145, 15.150, 15.155, 15.160, 15.165, 15.170, 15.175, 15.180, 15.185, 15.190, 15.195, 15.200, 15.205, 15.210, 15.215, 15.220, 15.225, 15.230, 15.235, 15.240, 15.245, 15.250, 15.255, 15.260, 15.265, 15.270, 15.275, 15.280, 15.285, 15.290, 15.295, 15.300, 15.305)  
" >>  ./Parmec_Test.csv



echo "*COMPONENT_INFORMATION
**NaN_handling, 'CUSTOM',1.E-4
**component_info_units, (m, m, m, rad, rad, rad)
**component_info_vector," >>  ./columnIB_FILLER
# Generate Column of Interstitial Brick
X_init="-0.0000556"
Y_init="-0.0000193"
Z_init=$((0))
for LayerY in $(seq 1 $NUMBER_OF_Y)
do
    YcompI=`echo "$Y_init+$((LayerY-1))*0.07" | bc`
    for LayerX in $(seq 1 $NUMBER_OF_X)
    do
        XcompI=`echo "$X_init+$((LayerX-1))*0.07" | bc`
        #echo $XcompI && sleep 5
        for compI in $(seq 1 $NUMBER_OF_COMPZ)
        do
           compIB=$((2*(compI-1)+1))
           compIB=$((compIB+2*(LayerX-1)*NUMBER_OF_COMPZ+2*(LayerY-1)*NUMBER_OF_COMPZ))
           compFB=$((2*(compI-1)+2))
           compFB=$((compFB+2*(LayerX-1)*NUMBER_OF_COMPZ+2*(LayerY-1)*NUMBER_OF_COMPZ))
           NcompI=$((compI-1))
           ZcompI=`echo "$Z_init+$((NcompI))*0.29" | bc`
           echo "component$compIB, ’01’, ‘interstitial_brick’, ‘NaN’, ($XcompI, $YcompI, $ZcompI, 0, 0, 0) "  \
               >>  ./columnIB_FILLER
           ZcompI=`echo "$Z_init+0.08+$((NcompI))*0.29" | bc`
           echo "component$compFB, ’02’, ‘interstitial_brick’, ‘NaN’, ($XcompI, $YcompI, $ZcompI, 0, 0, 0) "  \
               >>  ./columnIB_FILLER
        done
    done
done

## Generate Column of Filler Brick
#for compI in $(seq $((NUMBER_OF_COMPZ+1)) $((NUMBER_OF_COMPZ+NUMBER_OF_COMPZ)))
#do
   #NcompI=$((compI-1))
   #ZcompI=`echo "$Z_init+0.155+$((NcompI))*0.31" | bc`
   #echo "component$compI, ’02’, ‘interstitial_brick’, ‘NaN’, (-5.56E-4, -1.93e-4, $ZcompI, 0, 0, 0) "  \
       #>>  ./columnIB_FILLER
#done
echo "
 "  >>  ./columnIB_FILLER

# Generate Time Data for Displacment
echo "*COMPONENT_DATA, DISPLACEMENT
**component_data_length, 6
**component_data_units, (mm, mm, mm, rad, rad, rad)
**NaN_handling, 'DEFAULT'
**component_data_offset, ‘ZERO’" >>  ./columnIB_FILLER

echo "
 "  >>  ./columnIB_FILLER

#Generate Time Data
for timeI in $(seq $((0)) $((NB_TIME_STEP-1)))
do
echo "***time_step, $timeI" >>  ./columnIB_FILLER
    # Generate Column of Filler Brick
    for compI in $(seq $((1)) $((NUMBER_TOTAL_COMP_WHOLE)))
    do
       val1=`shuf -i 1-10 -n 1`
       val2=`shuf -i 1-10 -n 1`
       val3=`shuf -i 1-10 -n 1`
       val4=`shuf -i 1-10 -n 1`
       val5=`shuf -i 1-10 -n 1`
       val6=`shuf -i 1-10 -n 1`
       val7=`shuf -i 50-100 -n 1`
       val8=`shuf -i 150-200 -n 1`
       sign_random=`shuf -i 1-2 -n 1`
       if  [   "$sign_random" == "1" ];
       then
          sign="+"
          othersign="+"
       else
          sign="-"   
          othersign="-"   
       fi
       echo "component$compI, "$sign$val1.$val7$val8"e-04,"$othersign$val2.$val7$val8"e-04,"$sign$val3.$val7$val8"e-04, "$othersign$val4.$val7$val8"e-04,"$sign$val5.$val7$val8"e-04,"$sign$val6.$val7$val8"e-04 "  \
           >>  ./columnIB_FILLER
    done
echo "
 "  >>  ./columnIB_FILLER

done





