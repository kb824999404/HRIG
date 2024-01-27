rootPath=$(pwd)
# Scene=japanesestreet
Scene=citystreet

# 1. render background and depth images
# cd ${rootPath}/data/blenderFiles/${Scene} && bash render.sh
# cd ${rootPath}/data/blenderFiles/${Scene} && bash renderDepth.sh

# 2. export scene info from blender
# cd ${rootPath}/scripts && bash exportScene.sh

# 3. simulate raindrops and export them to files
cd ${rootPath}/scripts && bash exportRainDrops.sh

# 4. render rain streaks layer
cd ${rootPath}/rain-streaks-rendering && bash renderStreaks.sh

# 5. light rain layer
cd ${rootPath}/rain-streaks-rendering && bash lightingStreaks.sh

# 6. render attenuation
# cd ${rootPath}/rain-streaks-rendering && bash renderAttenuation.sh

# 7. composite rain layer and background image
cd ${rootPath}/rain-streaks-rendering && bash composite.sh