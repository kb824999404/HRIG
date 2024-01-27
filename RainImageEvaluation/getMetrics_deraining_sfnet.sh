testsetRoot=data_deraining
predRoot=/home/zhoukaibin/data/Code/SIRR/SFNet/Deraining/results

# SPAData
testsetName=SPAData
testsetPath=${testsetRoot}/${testsetName}

# trainsetName=RainTrainL
# predPath=${predRoot}/${trainsetName}_${testsetName}
# predName=${trainsetName}_${testsetName}
# resultRoot=results_deraining
# python getMetrics_deraining.py \
#     -test ${testsetPath} \
#     --predRoot ${predPath} \
#     --predName ${predName} \
#     --result ${resultRoot}

# trainsetName=RainTrainL_ratio1
# predPath=${predRoot}/${trainsetName}_${testsetName}
# predName=${trainsetName}_${testsetName}
# resultRoot=results_deraining
# python getMetrics_deraining.py \
#     -test ${testsetPath} \
#     --predRoot ${predPath} \
#     --predName ${predName} \
#     --result ${resultRoot}

# trainsetName=RainTrainL_ratio1_nocolor
# predPath=${predRoot}/${trainsetName}_${testsetName}
# predName=${trainsetName}_${testsetName}
# resultRoot=results_deraining
# python getMetrics_deraining.py \
#     -test ${testsetPath} \
#     --predRoot ${predPath} \
#     --predName ${predName} \
#     --result ${resultRoot}

trainsetName=Rain1400
predPath=${predRoot}/${trainsetName}_${testsetName}
predName=${trainsetName}_${testsetName}
resultRoot=results_deraining
python getMetrics_deraining.py \
    -test ${testsetPath} \
    --predRoot ${predPath} \
    --predName ${predName} \
    --result ${resultRoot}

trainsetName=Rain1400_ratio1_nocolor
predPath=${predRoot}/${trainsetName}_${testsetName}
predName=${trainsetName}_${testsetName}
resultRoot=results_deraining
python getMetrics_deraining.py \
    -test ${testsetPath} \
    --predRoot ${predPath} \
    --predName ${predName} \
    --result ${resultRoot}