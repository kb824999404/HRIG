scene=citystreet  #场景名
sequences=(far front back sideinner sideleft sideright)  #所有序列
intensities=(10 25 50 100)  #雨强度
for sequence in ${sequences[@]} #遍历所有序列
do
    for intensity in ${intensities[@]} #遍历所有强度
    do
        rm -r data/output/${scene}/${sequence}/${intensity}mm/wind_0_0_0_0/buffer_color
        rm -r data/output/${scene}/${sequence}/${intensity}mm/wind_0_0_0_0/rainy_image
    done
done