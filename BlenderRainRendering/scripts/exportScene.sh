pyFile=$(pwd)/exportScene.py
baseRoot=/home/zhoukaibin/disk1/Code/BlenderRainRendering
dataRoot=${baseRoot}/data
blenderRoot=${dataRoot}/blenderFiles  #Blender文件位置
resultRoot=${dataRoot}/source         #结果文件位置

############################## lane场景 ##############################
# scene=lane  #场景名
# sequences=(front low mid side)  #所有序列
# for sequence in ${sequences[@]} #遍历所有序列
# do
#     if [ ! -d ${resultRoot}/${scene}/${sequence} ]; then    #判断目录是否存在，不存在则创建
#         mkdir -p ${resultRoot}/${scene}/${sequence}
#     fi
#     cd ${resultRoot}/${scene}/${sequence} && blender -b ${blenderRoot}/${scene}/${scene}_${sequence}.blend  -P ${pyFile}    #导出该序列所有帧的相机参数
# done
#############################################################################

############################### citystreet场景 ###############################
scene=citystreet  #场景名
sequences=(far front back sideinner sideleft sideright)  #所有序列
for sequence in ${sequences[@]} #遍历所有序列
do
    if [ ! -d ${resultRoot}/${scene}/${sequence} ]; then    #判断目录是否存在，不存在则创建
        mkdir -p ${resultRoot}/${scene}/${sequence}
    fi
    cd ${resultRoot}/${scene}/${sequence} && blender -b ${blenderRoot}/${scene}/${scene}_${sequence}.blend  -P ${pyFile}    #导出该序列所有帧的相机参数
done
#############################################################################

############################### japanesestreet场景 ###############################
scene=japanesestreet  #场景名
sequences=(camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 camera10)  #所有序列
for sequence in ${sequences[@]} #遍历所有序列
do
    if [ ! -d ${resultRoot}/${scene}/${sequence} ]; then    #判断目录是否存在，不存在则创建
        mkdir -p ${resultRoot}/${scene}/${sequence}
    fi
    cd ${resultRoot}/${scene}/${sequence} && blender -b ${blenderRoot}/${scene}/${scene}_${sequence}.blend  -P ${pyFile}    #导出该序列所有帧的相机参数
done
#############################################################################