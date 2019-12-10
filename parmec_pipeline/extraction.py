import sys
input_file_path = str(sys.argv[1])
input_file_name = str(sys.argv[2])

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

file_to_open = input_file_path + input_file_name + "0rb.xmf"
save_file = input_file_path + '/' + input_file_name + '/' + input_file_name + ".csv"

# create a new 'XDMF Reader'
cracks_1000_10rbxmf = XDMFReader(FileNames=[file_to_open])
cracks_1000_10rbxmf.PointArrayStatus = ['ANGVEL', 'DISPL', 'FORCE', 'LINVEL', 'NUMBER', 'ORIENT', 'ORIENT1', 'ORIENT2', 'ORIENT3', 'TORQUE']

# save data
SaveData(save_file, proxy=cracks_1000_10rbxmf, WriteAllTimeSteps=1)

# get animation scene
# # animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
# # animationScene1.UpdateAnimationUsingDataTimeSteps()


# Properties modified on cracks_1000_10rbxmf
# # cracks_1000_10rbxmf.GridStatus = ['PARMEC rigid bodies', 'PARMEC rigid bodies[1]', 'PARMEC rigid bodies[2]', 'PARMEC rigid bodies[3]', 'PARMEC rigid bodies[4]', 'PARMEC rigid bodies[5]', 'PARMEC rigid bodies[6]', 'PARMEC rigid bodies[7]', 'PARMEC rigid bodies[8]', 'PARMEC rigid bodies[9]', 'PARMEC rigid bodies[10]', 'PARMEC rigid bodies[11]', 'PARMEC rigid bodies[12]', 'PARMEC rigid bodies[13]', 'PARMEC rigid bodies[14]', 'PARMEC rigid bodies[15]', 'PARMEC rigid bodies[16]', 'PARMEC rigid bodies[17]', 'PARMEC rigid bodies[18]', 'PARMEC rigid bodies[19]', 'PARMEC rigid bodies[20]', 'PARMEC rigid bodies[21]', 'PARMEC rigid bodies[22]', 'PARMEC rigid bodies[23]', 'PARMEC rigid bodies[24]', 'PARMEC rigid bodies[25]', 'PARMEC rigid bodies[26]', 'PARMEC rigid bodies[27]', 'PARMEC rigid bodies[28]', 'PARMEC rigid bodies[29]', 'PARMEC rigid bodies[30]', 'PARMEC rigid bodies[31]', 'PARMEC rigid bodies[32]', 'PARMEC rigid bodies[33]', 'PARMEC rigid bodies[34]', 'PARMEC rigid bodies[35]', 'PARMEC rigid bodies[36]', 'PARMEC rigid bodies[37]', 'PARMEC rigid bodies[38]', 'PARMEC rigid bodies[39]', 'PARMEC rigid bodies[40]', 'PARMEC rigid bodies[41]', 'PARMEC rigid bodies[42]', 'PARMEC rigid bodies[43]', 'PARMEC rigid bodies[44]', 'PARMEC rigid bodies[45]', 'PARMEC rigid bodies[46]', 'PARMEC rigid bodies[47]', 'PARMEC rigid bodies[48]', 'PARMEC rigid bodies[49]', 'PARMEC rigid bodies[50]', 'PARMEC rigid bodies[51]', 'PARMEC rigid bodies[52]', 'PARMEC rigid bodies[53]', 'PARMEC rigid bodies[54]', 'PARMEC rigid bodies[55]', 'PARMEC rigid bodies[56]', 'PARMEC rigid bodies[57]', 'PARMEC rigid bodies[58]', 'PARMEC rigid bodies[59]', 'PARMEC rigid bodies[60]', 'PARMEC rigid bodies[61]', 'PARMEC rigid bodies[62]', 'PARMEC rigid bodies[63]', 'PARMEC rigid bodies[64]', 'PARMEC rigid bodies[65]', 'PARMEC rigid bodies[66]', 'PARMEC rigid bodies[67]', 'PARMEC rigid bodies[68]', 'PARMEC rigid bodies[69]', 'PARMEC rigid bodies[70]', 'PARMEC rigid bodies[71]', 'PARMEC rigid bodies[72]', 'PARMEC rigid bodies[73]', 'PARMEC rigid bodies[74]', 'PARMEC rigid bodies[75]', 'PARMEC rigid bodies[76]', 'PARMEC rigid bodies[77]', 'PARMEC rigid bodies[78]', 'PARMEC rigid bodies[79]', 'PARMEC rigid bodies[80]', 'PARMEC rigid bodies[81]', 'PARMEC rigid bodies[82]', 'PARMEC rigid bodies[83]', 'PARMEC rigid bodies[84]', 'PARMEC rigid bodies[85]', 'PARMEC rigid bodies[86]', 'PARMEC rigid bodies[87]', 'PARMEC rigid bodies[88]', 'PARMEC rigid bodies[89]', 'PARMEC rigid bodies[90]', 'PARMEC rigid bodies[91]', 'PARMEC rigid bodies[92]', 'PARMEC rigid bodies[93]', 'PARMEC rigid bodies[94]', 'PARMEC rigid bodies[95]', 'PARMEC rigid bodies[96]', 'PARMEC rigid bodies[97]', 'PARMEC rigid bodies[98]', 'PARMEC rigid bodies[99]', 'PARMEC rigid bodies[100]', 'PARMEC rigid bodies[101]', 'PARMEC rigid bodies[102]', 'PARMEC rigid bodies[103]', 'PARMEC rigid bodies[104]', 'PARMEC rigid bodies[105]', 'PARMEC rigid bodies[106]', 'PARMEC rigid bodies[107]', 'PARMEC rigid bodies[108]', 'PARMEC rigid bodies[109]', 'PARMEC rigid bodies[110]', 'PARMEC rigid bodies[111]', 'PARMEC rigid bodies[112]', 'PARMEC rigid bodies[113]', 'PARMEC rigid bodies[114]', 'PARMEC rigid bodies[115]', 'PARMEC rigid bodies[116]', 'PARMEC rigid bodies[117]', 'PARMEC rigid bodies[118]', 'PARMEC rigid bodies[119]', 'PARMEC rigid bodies[120]', 'PARMEC rigid bodies[121]', 'PARMEC rigid bodies[122]', 'PARMEC rigid bodies[123]', 'PARMEC rigid bodies[124]', 'PARMEC rigid bodies[125]', 'PARMEC rigid bodies[126]', 'PARMEC rigid bodies[127]', 'PARMEC rigid bodies[128]', 'PARMEC rigid bodies[129]', 'PARMEC rigid bodies[130]', 'PARMEC rigid bodies[131]', 'PARMEC rigid bodies[132]', 'PARMEC rigid bodies[133]', 'PARMEC rigid bodies[134]', 'PARMEC rigid bodies[135]', 'PARMEC rigid bodies[136]', 'PARMEC rigid bodies[137]', 'PARMEC rigid bodies[138]', 'PARMEC rigid bodies[139]', 'PARMEC rigid bodies[140]', 'PARMEC rigid bodies[141]', 'PARMEC rigid bodies[142]', 'PARMEC rigid bodies[143]', 'PARMEC rigid bodies[144]', 'PARMEC rigid bodies[145]', 'PARMEC rigid bodies[146]', 'PARMEC rigid bodies[147]', 'PARMEC rigid bodies[148]', 'PARMEC rigid bodies[149]', 'PARMEC rigid bodies[150]', 'PARMEC rigid bodies[151]', 'PARMEC rigid bodies[152]', 'PARMEC rigid bodies[153]', 'PARMEC rigid bodies[154]', 'PARMEC rigid bodies[155]', 'PARMEC rigid bodies[156]', 'PARMEC rigid bodies[157]', 'PARMEC rigid bodies[158]', 'PARMEC rigid bodies[159]', 'PARMEC rigid bodies[160]', 'PARMEC rigid bodies[161]', 'PARMEC rigid bodies[162]', 'PARMEC rigid bodies[163]', 'PARMEC rigid bodies[164]', 'PARMEC rigid bodies[165]', 'PARMEC rigid bodies[166]', 'PARMEC rigid bodies[167]', 'PARMEC rigid bodies[168]', 'PARMEC rigid bodies[169]', 'PARMEC rigid bodies[170]', 'PARMEC rigid bodies[171]', 'PARMEC rigid bodies[172]', 'PARMEC rigid bodies[173]', 'PARMEC rigid bodies[174]', 'PARMEC rigid bodies[175]', 'PARMEC rigid bodies[176]', 'PARMEC rigid bodies[177]', 'PARMEC rigid bodies[178]', 'PARMEC rigid bodies[179]', 'PARMEC rigid bodies[180]', 'PARMEC rigid bodies[181]', 'PARMEC rigid bodies[182]', 'PARMEC rigid bodies[183]', 'PARMEC rigid bodies[184]', 'PARMEC rigid bodies[185]', 'PARMEC rigid bodies[186]', 'PARMEC rigid bodies[187]', 'PARMEC rigid bodies[188]', 'PARMEC rigid bodies[189]', 'PARMEC rigid bodies[190]', 'PARMEC rigid bodies[191]', 'PARMEC rigid bodies[192]', 'PARMEC rigid bodies[193]', 'PARMEC rigid bodies[194]', 'PARMEC rigid bodies[195]', 'PARMEC rigid bodies[196]', 'PARMEC rigid bodies[197]', 'PARMEC rigid bodies[198]', 'PARMEC rigid bodies[199]', 'PARMEC rigid bodies[200]', 'PARMEC rigid bodies[201]', 'PARMEC rigid bodies[202]', 'PARMEC rigid bodies[203]', 'PARMEC rigid bodies[204]', 'PARMEC rigid bodies[205]', 'PARMEC rigid bodies[206]', 'PARMEC rigid bodies[207]', 'PARMEC rigid bodies[208]', 'PARMEC rigid bodies[209]', 'PARMEC rigid bodies[210]', 'PARMEC rigid bodies[211]', 'PARMEC rigid bodies[212]', 'PARMEC rigid bodies[213]', 'PARMEC rigid bodies[214]', 'PARMEC rigid bodies[215]', 'PARMEC rigid bodies[216]', 'PARMEC rigid bodies[217]', 'PARMEC rigid bodies[218]', 'PARMEC rigid bodies[219]', 'PARMEC rigid bodies[220]', 'PARMEC rigid bodies[221]', 'PARMEC rigid bodies[222]', 'PARMEC rigid bodies[223]', 'PARMEC rigid bodies[224]', 'PARMEC rigid bodies[225]', 'PARMEC rigid bodies[226]', 'PARMEC rigid bodies[227]', 'PARMEC rigid bodies[228]', 'PARMEC rigid bodies[229]', 'PARMEC rigid bodies[230]', 'PARMEC rigid bodies[231]', 'PARMEC rigid bodies[232]', 'PARMEC rigid bodies[233]', 'PARMEC rigid bodies[234]', 'PARMEC rigid bodies[235]', 'PARMEC rigid bodies[236]', 'PARMEC rigid bodies[237]', 'PARMEC rigid bodies[238]', 'PARMEC rigid bodies[239]', 'PARMEC rigid bodies[240]', 'PARMEC rigid bodies[241]', 'PARMEC rigid bodies[242]', 'PARMEC rigid bodies[243]', 'PARMEC rigid bodies[244]', 'PARMEC rigid bodies[245]', 'PARMEC rigid bodies[246]', 'PARMEC rigid bodies[247]', 'PARMEC rigid bodies[248]', 'PARMEC rigid bodies[249]', 'PARMEC rigid bodies[250]', 'PARMEC rigid bodies[251]', 'PARMEC rigid bodies[252]', 'PARMEC rigid bodies[253]', 'PARMEC rigid bodies[254]', 'PARMEC rigid bodies[255]', 'PARMEC rigid bodies[256]', 'PARMEC rigid bodies[257]', 'PARMEC rigid bodies[258]', 'PARMEC rigid bodies[259]', 'PARMEC rigid bodies[260]', 'PARMEC rigid bodies[261]', 'PARMEC rigid bodies[262]', 'PARMEC rigid bodies[263]', 'PARMEC rigid bodies[264]', 'PARMEC rigid bodies[265]', 'PARMEC rigid bodies[266]', 'PARMEC rigid bodies[267]', 'PARMEC rigid bodies[268]', 'PARMEC rigid bodies[269]', 'PARMEC rigid bodies[270]', 'PARMEC rigid bodies[271]']
