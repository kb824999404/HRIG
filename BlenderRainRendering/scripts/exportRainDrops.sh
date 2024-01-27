pyFile=$(pwd)/exportRainDrops.py
baseRoot=/home/zhoukaibin/disk1/Code/BlenderRainRendering
dataRoot=${baseRoot}/data
sourceRoot=${dataRoot}/source
outputRoot=${dataRoot}/particles

############################## lane场景 ##############################
scene=lane  #场景名
sequences=(front low mid side)  #所有序列
intensities=(10 25 50 100)
for sequence in ${sequences[@]} #遍历所有序列
do
        for intensity in ${intensities[@]}
        do
                sourceFile=${sourceRoot}/${scene}/${sequence}/scene_info.json
                python exportRainDrops.py -S ${sourceFile} -O ${outputRoot} \
                        --scene ${scene} --sequence ${sequence} \
                        -c ${baseRoot}/configs/${scene}.yaml -I ${intensity}
                # 后台运行，Log输出到文件
                # logPath=${outputRoot}/${scene}/${sequence}/${intensity}mm
                # mkdir -p ${logPath}
                # time=$(date +"%Y_%m_%d_%S")
                # logFile=${logPath}/exportRainDrops_${time}.log
                # nohup python exportRainDrops.py -S ${sourceFile} -O ${outputRoot} \
                #         --scene ${scene} --sequence ${sequence} \
                #         -c ${baseRoot}/configs/${scene}.yaml -I ${intensity}\
                #         > ${logFile} 2>&1 &
                # echo Run exportRainDrops.py, log at ${logFile}
        done
done
#############################################################################

############################### citystreet场景 ###############################
# scene=citystreet  #场景名
# sequences=(far front back sideinner sideleft sideright)  #所有序列
# intensity=10
# for sequence in ${sequences[@]} #遍历所有序列
# do
#         sourceFile=${sourceRoot}/${scene}/${sequence}/scene_info.json
#         python exportRainDrops.py -S ${sourceFile} -O ${outputRoot} \
#                 --scene ${scene} --sequence ${sequence} \
#                 -c ${baseRoot}/configs/${scene}.yaml -I ${intensity}
#         # 后台运行，Log输出到文件
#         # logPath=${outputRoot}/${scene}/${sequence}/${intensity}mm
#         # mkdir -p ${logPath}
#         # time=$(date +"%Y_%m_%d_%S")
#         # logFile=${logPath}/exportRainDrops_${time}.log
#         # nohup python -u exportRainDrops.py -S ${sourceFile} -O ${outputRoot} \
#         #         --scene ${scene} --sequence ${sequence} \
#         #         -c ${baseRoot}/configs/${scene}.yaml -I ${intensity} \
#         #         > ${logFile} 2>&1 &
#         # echo Run exportRainDrops.py, log at ${logFile}
# done
#############################################################################

############################### japanesestreet场景 ###############################
# scene=japanesestreet
# sequences=(camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 camera10)  #所有序列
# intensities=(10 25 50 100)
# for sequence in ${sequences[@]} #遍历所有序列
# do
#         for intensity in ${intensities[@]}
#         do
#                 sourceFile=${sourceRoot}/${scene}/${sequence}/scene_info.json
#                 python exportRainDrops.py -S ${sourceFile} -O ${outputRoot} \
#                         --scene ${scene} --sequence ${sequence} \
#                         -c ${baseRoot}/configs/${scene}_${sequence}.yaml -I ${intensity}
#                 # 后台运行，Log输出到文件
#                 # logPath=${outputRoot}/${scene}/${sequence}/${intensity}mm
#                 # mkdir -p ${logPath}
#                 # time=$(date +"%Y_%m_%d_%S")
#                 # logFile=${logPath}/exportRainDrops_${time}.log
#                 # nohup python -u exportRainDrops.py -S ${sourceFile} -O ${outputRoot} \
#                 #         --scene ${scene} --sequence ${sequence} \
#                 #         -c ${baseRoot}/configs/${scene}.yaml -I ${intensity} \
#                 #         > ${logFile} 2>&1 &
#                 # echo Run exportRainDrops.py, log at ${logFile}
#         done
# done
#############################################################################