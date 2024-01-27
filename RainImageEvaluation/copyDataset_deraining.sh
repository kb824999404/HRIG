testsetRoot=/home/zhoukaibin/data/dataset/SyntheticRainTest
testsetName=Rain100L
testsetPath=${testsetRoot}/${testsetName}
output=data_deraining/${testsetName}

# python copyDataset_deraining.py -d ${testsetPath} -n ${testsetName} -o ${output}

testsetName=Rain100H
testsetPath=${testsetRoot}/${testsetName}
output=data_deraining/${testsetName}
# python copyDataset_deraining.py -d ${testsetPath} -n ${testsetName} -o ${output}

testsetName=Rain1400
testsetPath=${testsetRoot}/${testsetName}
output=data_deraining/${testsetName}
python copyDataset_deraining.py -d ${testsetPath} -n ${testsetName} -o ${output}