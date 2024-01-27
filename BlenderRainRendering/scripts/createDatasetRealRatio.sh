rainRoot=/home/ubuntu/data/rain-mask-wind-single/rain_512

# RainTrainL
# bgRoot=/home/ubuntu/data/SyntheticRain/RainTrainL/label
# ratio=1
# resultRoot=/home/ubuntu/data/RealBGMaskNew/RainTrainL_rain512_randomRain_ratio${ratio}
# python createDatasetRealRatio.py -b ${bgRoot} -r ${rainRoot} -o ${resultRoot} -n ${ratio}
# ratio=2
# resultRoot=/home/ubuntu/data/RealBGMaskNew/RainTrainL_rain512_randomRain_ratio${ratio}
# python createDatasetRealRatio.py -b ${bgRoot} -r ${rainRoot} -o ${resultRoot} -n ${ratio}

# RainTrainH
# bgRoot=/home/ubuntu/data/SyntheticRain/RainTrainH/label
# ratio=1
# resultRoot=/home/ubuntu/data/RealBGMaskNew/RainTrainH_rain512_randomRain_ratio${ratio}
# python createDatasetRealRatio.py -b ${bgRoot} -r ${rainRoot} -o ${resultRoot} -n ${ratio}
# ratio=2
# resultRoot=/home/ubuntu/data/RealBGMaskNew/RainTrainH_rain512_randomRain_ratio${ratio}
# python createDatasetRealRatio.py -b ${bgRoot} -r ${rainRoot} -o ${resultRoot} -n ${ratio}


# Rain12600
bgRoot=/home/ubuntu/data/SyntheticRain/Rain12600/ground_truth
ratio=1
resultRoot=/home/ubuntu/data/RealBGMaskNew/Rain12600_rain512_randomRain_ratio${ratio}
python createDatasetRealRatio.py -b ${bgRoot} -r ${rainRoot} -o ${resultRoot} -n ${ratio}
ratio=2
resultRoot=/home/ubuntu/data/RealBGMaskNew/Rain12600_rain512_randomRain_ratio${ratio}
python createDatasetRealRatio.py -b ${bgRoot} -r ${rainRoot} -o ${resultRoot} -n ${ratio}