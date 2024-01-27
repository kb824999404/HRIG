baseRoot=/home/zhoukaibin/Code/BlenderRainRendering
dataRoot=${baseRoot}/data
sourceRoot=${dataRoot}/source
attenuationRoot=${dataRoot}/attenuation
outputRoot=${dataRoot}/output

############################## lane场景 ##############################
# scene=lane  #场景名
# sequences=(front low mid side)  #所有序列
# intensity=10
# for sequence in ${sequences[@]} #遍历所有序列
# do
#     python compositeAttenuation.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
#     -i ${intensity} -fe 10 --output ${outputRoot}
#     # 后台运行，Log输出到文件
#     # logPath=${outputRoot}/${scene}/${sequence}/${intensity}mm
#     # mkdir -p ${logPath}
#     # time=$(date +"%Y_%m_%d_%S")
#     # logFile=${logPath}/composite_${time}.log
#     # nohup python compositeAttenuation.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
#     #     -i ${intensity} --output ${outputRoot}\
#     #     > ${logFile} 2>&1 &
#     # echo Run compositeAttenuation.py, log at ${logFile}
# done
#############################################################################

############################### citystreet场景 ###############################
# scene=citystreet  #场景名
# sequences=(far front back sideinner sideleft sideright)  #所有序列
# intensity=100
# for sequence in ${sequences[@]} #遍历所有序列
# do
#     python compositeAttenuation.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
#     -i ${intensity} --output ${outputRoot}
#     # 后台运行，Log输出到文件
#     # logPath=${outputRoot}/${scene}/${sequence}/${intensity}mm
#     # mkdir -p ${logPath}
#     # time=$(date +"%Y_%m_%d_%S")
#     # logFile=${logPath}/composite_${time}.log
#     # nohup python -u compositeAttenuation.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
#     #     -i ${intensity} --output ${outputRoot}\
#     #     > ${logFile} 2>&1 &
#     # echo Run compositeAttenuation.py, log at ${logFile}
# done
#############################################################################

############################### japanesestreet场景 ###############################
scene=japanesestreet  #场景名
# sequences=(camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 camera10)  #所有序列
sequences=(camera1 camera2 camera3 camera4 camera6 camera9)  #所有序列
intensities=(10 25 50 100)  #雨强度
for sequence in ${sequences[@]} #遍历所有序列
do
    for intensity in ${intensities[@]}
    do
        python compositeAttenuation.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
        -i ${intensity} -a ${attenuationRoot} --output ${outputRoot} -fe 12
        python compositeAttenuation.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
        -i ${intensity} -a ${attenuationRoot} --output ${outputRoot} -fs 12
        # 后台运行，Log输出到文件
        # logPath=${outputRoot}/${scene}/${sequence}/${intensity}mm
        # mkdir -p ${logPath}
        # time=$(date +"%Y_%m_%d_%S")
        # logFile=${logPath}/composite_${time}.log
        # nohup python -u compositeAttenuation.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
        #     -i ${intensity} --output ${outputRoot}\
        #     > ${logFile} 2>&1 &
        # echo Run compositeAttenuation.py, log at ${logFile}
    done
done
#############################################################################