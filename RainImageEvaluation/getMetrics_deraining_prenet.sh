testsetRoot=data_deraining
predRoot=/home/zhoukaibin/data/Code/SIRR/PReNet/results/HRIG
resultRoot=results_deraining/prenet

# SPAData Testset
testsetName=SPAData
testsetPath=${testsetRoot}/${testsetName}

# TrainL SPAData
trainsetName=RainTrainL
predPath=${predRoot}/${trainsetName}/${testsetName}
predName=${trainsetName}_${testsetName}
python getMetrics_deraining.py \
    -test ${testsetPath} \
    --predRoot ${predPath} \
    --predName ${predName} \
    --result ${resultRoot}

trainsetName=RainTrainL_ratio1
predPath=${predRoot}/${trainsetName}/${testsetName}
predName=${trainsetName}_${testsetName}
python getMetrics_deraining.py \
    -test ${testsetPath} \
    --predRoot ${predPath} \
    --predName ${predName} \
    --result ${resultRoot}


# Rain1400 SPAData
trainsetName=Rain1400
predPath=${predRoot}/${trainsetName}/${testsetName}
predName=${trainsetName}_${testsetName}
python getMetrics_deraining.py \
    -test ${testsetPath} \
    --predRoot ${predPath} \
    --predName ${predName} \
    --result ${resultRoot}

trainsetName=Rain1400_ratio1
predPath=${predRoot}/${trainsetName}/${testsetName}
predName=${trainsetName}_${testsetName}
python getMetrics_deraining.py \
    -test ${testsetPath} \
    --predRoot ${predPath} \
    --predName ${predName} \
    --result ${resultRoot}