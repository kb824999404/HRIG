# Blender Rain Rendering

## File Structure

* `data/`
  * `rainstreakdb/`：The rain streaks database
    * `env_light_database/`
    * `point_light_database/`
  * `blenderFiles/`：The blender files
    * `${Scene}/`
      * `*.blender`
  * `source/`：The rendered background and depth images
    * `${Scene}/`
      * `${Sequence}/`
        * `background/`
        * `depth/`
        * `scene_info.json`
  * `particles/`：The raindrop particle files
    * `${Scene}/`
      * `${Sequence}/`
        * `${Intensity}mm/`
          * `${Name}/`
            * `raindrops_*.pkl`
  * `output/`：The rendered rain layers and rainy images
    * `${Scene}`
      * `${Sequence}`
        * `${Intensity}mm/`
          * `${Name}/`
            * `buffers/`
            * `buffer_color/`
            * `depth/`
            * `rain_lighting/`
            * `rain_layer/`
            * `rainy_image/`
* `scripts/`：Scripts to simulate rain drops and export scene info from blender
  * `exportScene.py`
  * `exportScene.sh`
  * `rainDropSimulator.py`
  * `simulatorBlender.py`
  * `simulatorTaichi.py`
  * `exportRainDrops.py`
  * `exportRainDrops.sh`
  * `createDataset.py`
  * `createDataset.sh`
  * `createDatasetReal.py`
  * `createDatasetReal.sh`
  * `createDatasetRealRatio.py`
  * `createDatasetRealRatio.sh`
* `rain-streaks-rendering/`：Scripts to render rain layers and rainy images
  * `common/`
    * `my_utils.py`
    * `bad_weather.py`
    * `generator.py`
    * `scene.py`
  * `renderStreaks.py`
  * `renderStreaks.sh`
  * `lightingStreaks.py`
  * `lightingStreaks.sh`
  * `renderAttenuation.py`
  * `renderAttenuation.sh`
  * `composite.py`
  * `composite.sh`
  * `compositeAttenuation.py`
  * `compositeAttenuation.sh`
* `configs`：scene config files for rainDropSimulator
* `run.sh`：The script to execute the total rendering pipeline

## How To Run

1. Download the blender files and the rain streak database, and put them in the `data` directory
   * Rain streak database: [rain streak database](https://pan.baidu.com/s/14G4fE8_7lswvod6OtIbOew?pwd=v9b2)(Extraction Code: v9b2)
   * Blender files: [Google Drive](https://drive.google.com/drive/folders/1MSS-iNaLxI05K_10pHMWYibrDJtMJngP?usp=sharing), [BaiduCloud](https://pan.baidu.com/s/14G4fE8_7lswvod6OtIbOew?pwd=v9b2)(Extraction Code: v9b2)
2. Modify the `${dataRoot}` in the scripts `data/blenderFiles/${Scene}/render.sh` and  `data/blenderFiles/${Scene}/renderDepth.sh` , and run them to render background and depth images
   * `cd data/blenderFiles/${Scene} && bash render.sh`
   * `cd data/blenderFiles/${Scene} && bash renderDepth.sh`
3. Modify the `${dataRoot}` in the scripts `scripts/exportScene.sh` , and run it to export scene info from blender
   * `cd scripts && bash exportScene.sh`
4. Modify the `${dataRoot}` in the scripts `scripts/exportRainDrops.sh` , and run it to simulate raindrops and export them to files
   * `cd scripts && bash exportRainDrops.sh`
5. Modify the `${dataRoot}` in the scripts `rain-streaks-rendering/renderStreaks.sh` , and run it to render rain layer
   * `cd rain-streaks-rendering && bash renderStreaks.sh`
6. Modify the `${dataRoot}` in the scripts `rain-streaks-rendering/lightingStreaks.sh` , and run it to light rain layer
   * `cd rain-streaks-rendering && bash lightingStreaks.sh`
7. Modify the `${dataRoot}` in the scripts `rain-streaks-rendering/composite.sh` , and run it to composite rain layer and background image
   * `cd rain-streaks-rendering && bash composite.sh`
8. Modify the `${dataRoot}` and `${resultRoot}` in the scripts `scripts/createDataset.sh` , and run it to create the dataset
   * `cd scripts && bash createDataset.sh`