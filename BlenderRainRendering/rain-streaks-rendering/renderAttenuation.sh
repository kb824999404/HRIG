baseRoot=/home/zhoukaibin/Code/BlenderRainRendering
dataRoot=${baseRoot}/data
sourceRoot=${dataRoot}/source
outputRoot=${dataRoot}/attenuation


############################### citystreet场景 ###############################
# scene=citystreet  #场景名
# # sequences=(far front back sideinner sideleft sideright)  #所有序列
# sequences=(far)  #所有序列
# intensities=(10)  #雨强度
# for sequence in ${sequences[@]} #遍历所有序列
# do
#     for intensity in ${intensities[@]} #遍历所有强度
#     do
#         python renderAttenuation.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
#         -i ${intensity} --output ${outputRoot}
#         # 后台运行，Log输出到文件
#         # logPath=${outputRoot}/${scene}/${sequence}/${intensity}mm
#         # mkdir -p ${logPath}
#         # time=$(date +"%Y_%m_%d_%S")
#         # logFile=${logPath}/composite_${time}.log
#         # nohup python -u composite.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
#         #     -i ${intensity} --output ${outputRoot}\
#         #     > ${logFile} 2>&1 &
#         # echo Run composite.py, log at ${logFile}
#     done
# done
#############################################################################

############################### japanesestreet场景 ###############################
scene=japanesestreet  #场景名
# sequences=(far front back sideinner sideleft sideright)  #所有序列
sequences=(camera6)  #所有序列
intensities=(100)  #雨强度
for sequence in ${sequences[@]} #遍历所有序列
do
    for intensity in ${intensities[@]} #遍历所有强度
    do
        python renderAttenuation.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
        -i ${intensity} --output ${outputRoot}
        # 后台运行，Log输出到文件
        # logPath=${outputRoot}/${scene}/${sequence}/${intensity}mm
        # mkdir -p ${logPath}
        # time=$(date +"%Y_%m_%d_%S")
        # logFile=${logPath}/composite_${time}.log
        # nohup python -u composite.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
        #     -i ${intensity} --output ${outputRoot}\
        #     > ${logFile} 2>&1 &
        # echo Run composite.py, log at ${logFile}
    done
done
#############################################################################