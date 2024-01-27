baseRoot=/home/zhoukaibin/disk1/Code/BlenderRainRendering
dataRoot=${baseRoot}/data
sourceRoot=${dataRoot}/source
outputRoot=${dataRoot}/output

############################## lane场景 ##############################
scene=lane  #场景名
sequences=(front low mid side)  #所有序列
intensities=(10 25 50 100)
for sequence in ${sequences[@]} #遍历所有序列
do
    for intensity in ${intensities[@]}
    do
        python composite.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
        -i ${intensity} -fe 18 --output ${outputRoot}
        python composite.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
        -i ${intensity} -fs 18 -fe 36 --output ${outputRoot}
        python composite.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
        -i ${intensity} -fs 36 --output ${outputRoot}
        # 后台运行，Log输出到文件
        # logPath=${outputRoot}/${scene}/${sequence}/${intensity}mm
        # mkdir -p ${logPath}
        # time=$(date +"%Y_%m_%d_%S")
        # logFile=${logPath}/composite_${time}.log
        # nohup python composite.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
        #     -i ${intensity} --output ${outputRoot}\
        #     > ${logFile} 2>&1 &
        # echo Run composite.py, log at ${logFile}
    done
done
#############################################################################

############################### citystreet场景 ###############################
# scene=citystreet  #场景名
# sequences=(far front back sideinner sideleft sideright)  #所有序列
# intensity=100
# for sequence in ${sequences[@]} #遍历所有序列
# do
#     python composite.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
#     -i ${intensity} --output ${outputRoot}
#     # 后台运行，Log输出到文件
#     # logPath=${outputRoot}/${scene}/${sequence}/${intensity}mm
#     # mkdir -p ${logPath}
#     # time=$(date +"%Y_%m_%d_%S")
#     # logFile=${logPath}/composite_${time}.log
#     # nohup python -u composite.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
#     #     -i ${intensity} --output ${outputRoot}\
#     #     > ${logFile} 2>&1 &
#     # echo Run composite.py, log at ${logFile}
# done
#############################################################################

############################### japanesestreet场景 ###############################
# scene=japanesestreet  #场景名
# sequences=(camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 camera10)  #所有序列
# intensities=(10 25 50 100)  #雨强度
# for sequence in ${sequences[@]} #遍历所有序列
# do
#     for intensity in ${intensities[@]}
#     do
#         python composite.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
#         -i ${intensity} --output ${outputRoot} -fe 12
#         python composite.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
#         -i ${intensity} --output ${outputRoot} -fs 12
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