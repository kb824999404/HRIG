baseRoot=/home/zhoukaibin/disk1/Code/BlenderRainRendering
dataRoot=${baseRoot}/data
blenderRoot=${dataRoot}/blenderFiles  #Blender文件位置
outputRoot=${dataRoot}/output

############################## lane场景 ##############################
scene=lane  #场景名
sequences=(front low mid side)  #所有序列
intensities=(10 25 50 100)
for sequence in ${sequences[@]} #遍历所有序列
do
  for intensity in ${intensities[@]}
  do
    blender -b ${blenderRoot}/${scene}/${scene}_${sequence}.blend -P lightingStreaks.py -- \
      -O ${outputRoot} --scene ${scene} --sequence ${sequence} -i ${intensity} -fe 18
    blender -b ${blenderRoot}/${scene}/${scene}_${sequence}.blend -P lightingStreaks.py -- \
      -O ${outputRoot} --scene ${scene} --sequence ${sequence} -i ${intensity} -fs 18 -fe 36
    blender -b ${blenderRoot}/${scene}/${scene}_${sequence}.blend -P lightingStreaks.py -- \
      -O ${outputRoot} --scene ${scene} --sequence ${sequence} -i ${intensity} -fs 36
    # 后台运行，Log输出到文件
    # logPath=${outputRoot}/${scene}/${sequence}/${intensity}mm
    # mkdir -p ${logPath}
    # time=$(date +"%Y_%m_%d_%S")
    # logFile=${logPath}/lightingStreaks_${time}.log
    # nohup blender -b ${blenderRoot}/${scene}/${scene}_${sequence}.blend -P lightingStreaks.py -- \
    #     -O ${outputRoot} --scene ${scene} --sequence ${sequence} -i ${intensity}\
    #     > ${logFile} 2>&1 &
    # echo Run lightingStreaks.py, log at ${logFile}
  done
done
#############################################################################

############################### citystreet场景 ###############################
# scene=citystreet  #场景名
# sequences=(far front back sideinner sideleft sideright)  #所有序列
# intensity=100
# for sequence in ${sequences[@]} #遍历所有序列
# do
#     blender -b ${blenderRoot}/${scene}/${scene}_${sequence}.blend -P lightingStreaks.py -- \
#       -O ${outputRoot} --scene ${scene} --sequence ${sequence} -i ${intensity}
#     # 后台运行，Log输出到文件
#     # logPath=${outputRoot}/${scene}/${sequence}/${intensity}mm
#     # mkdir -p ${logPath}
#     # time=$(date +"%Y_%m_%d_%S")
#     # logFile=${logPath}/lightingStreaks_${time}.log
#     # nohup blender -b ${blenderRoot}/${scene}/${scene}_${sequence}.blend -P lightingStreaks.py -- \
#     #     -O ${outputRoot} --scene ${scene} --sequence ${sequence} -i ${intensity}\
#     #     > ${logFile} 2>&1 &
#     # echo Run lightingStreaks.py, log at ${logFile}
# done
#############################################################################

############################### japanesestreet场景 ###############################
# scene=japanesestreet  #场景名
# sequences=(camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9 camera10)  #所有序列
# intensities=(10 25 50 100)  #雨强度
# for sequence in ${sequences[@]} #遍历所有序列
# do
#   for intensity in ${intensities[@]}
#   do
#     blender -b ${blenderRoot}/${scene}/${scene}_${sequence}.blend -P lightingStreaks.py -- \
#       -O ${outputRoot} --scene ${scene} --sequence ${sequence} -i ${intensity} -fe 12
#     blender -b ${blenderRoot}/${scene}/${scene}_${sequence}.blend -P lightingStreaks.py -- \
#       -O ${outputRoot} --scene ${scene} --sequence ${sequence} -i ${intensity} -fs 12
#     # 后台运行，Log输出到文件
#     # logPath=${outputRoot}/${scene}/${sequence}/${intensity}mm
#     # mkdir -p ${logPath}
#     # time=$(date +"%Y_%m_%d_%S")
#     # logFile=${logPath}/lightingStreaks_${time}.log
#     # nohup blender -b ${blenderRoot}/${scene}/${scene}_${sequence}.blend -P lightingStreaks.py -- \
#     #     -O ${outputRoot} --scene ${scene} --sequence ${sequence} -i ${intensity}\
#     #     > ${logFile} 2>&1 &
#     # echo Run lightingStreaks.py, log at ${logFile}
#   done
# done
#############################################################################