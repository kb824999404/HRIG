bgRoot=/home/zhoukaibin/data/dataset/R100
rainRoot=/home/zhoukaibin/data/dataset/rain-mask/customdb_512_naive_db
resultRoot=/home/zhoukaibin/data/dataset/RealBGMask/R100_rain512_randomBG

python createDatasetReal.py -b ${bgRoot} -r ${rainRoot} -o ${resultRoot}