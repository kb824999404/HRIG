baseRoot=/home/zhoukaibin/disk1/Code/BlenderRainRendering
dataRoot=${baseRoot}/data
sourceRoot=${dataRoot}/source
particlesRoot=${dataRoot}/particles
streaksRoot=${dataRoot}/rainstreakdb
outputRoot=${dataRoot}/output

############################## lane场景 ##############################
scene=lane  #场景名
intensities=(10 25 50 100)  #雨强度
sequences=(front low mid side)  #所有序列
for sequence in ${sequences[@]} #遍历所有序列
do
  for intensity in ${intensities[@]}
  do
    python renderStreaks.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
      -r ${particlesRoot} -sd ${streaksRoot} -i ${intensity} --output ${outputRoot} -fe 18
    python renderStreaks.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
      -r ${particlesRoot} -sd ${streaksRoot} -i ${intensity} --output ${outputRoot} -fs 18 -fe 36
    python renderStreaks.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
      -r ${particlesRoot} -sd ${streaksRoot} -i ${intensity} --output ${outputRoot} -fs 36
    # 后台运行，Log输出到文件
    # logPath=${outputRoot}/${scene}/${sequence}/${intensity}mm
    # mkdir -p ${logPath}
    # time=$(date +"%Y_%m_%d_%S")
    # logFile=${logPath}/renderStreaks_${time}.log
    # nohup python renderStreaks.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
    #   -r ${particlesRoot} -sd ${streaksRoot} -i ${intensity} --output ${outputRoot}\
    #     > ${logFile} 2>&1 &
    # echo Run renderStreaks.py, log at ${logFile}
  done
done
#############################################################################

############################### citystreet场景 ###############################
# scene=citystreet  #场景名
# intensity=100  #雨强度
# sequences=(far front back sideinner sideleft sideright)  #所有序列
# for sequence in ${sequences[@]} #遍历所有序列
# do
#     python renderStreaks.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
#       -r ${particlesRoot} -sd ${streaksRoot} -i ${intensity} --output ${outputRoot}
#     # 后台运行，Log输出到文件
#     # logPath=${outputRoot}/${scene}/${sequence}/${intensity}mm
#     # mkdir -p ${logPath}
#     # time=$(date +"%Y_%m_%d_%S")
#     # logFile=${logPath}/renderStreaks_${time}.log
#     # nohup python -u renderStreaks.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
#     #   -r ${particlesRoot} -sd ${streaksRoot} -i ${intensity} -ff 16 --output ${outputRoot}\
#     #     > ${logFile} 2>&1 &
#     # echo Run renderStreaks.py, log at ${logFile}
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
#     python renderStreaks.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
#       -r ${particlesRoot} -sd ${streaksRoot} -i ${intensity} --output ${outputRoot} -fe 12
#     python renderStreaks.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
#       -r ${particlesRoot} -sd ${streaksRoot} -i ${intensity} --output ${outputRoot} -fs 12
#     # 后台运行，Log输出到文件
#     # logPath=${outputRoot}/${scene}/${sequence}/${intensity}mm
#     # mkdir -p ${logPath}
#     # time=$(date +"%Y_%m_%d_%S")
#     # logFile=${logPath}/renderStreaks_${time}.log
#     # nohup python -u renderStreaks.py --dataset ${scene} -k ${sourceRoot} -s ${sequence} \
#     #   -r ${particlesRoot} -sd ${streaksRoot} -i ${intensity} -ff 16 --output ${outputRoot}\
#     #     > ${logFile} 2>&1 &
#     # echo Run renderStreaks.py, log at ${logFile}
#   done
# done
#############################################################################