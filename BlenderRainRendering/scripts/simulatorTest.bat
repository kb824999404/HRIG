@echo off
SET baseRoot=%cd%\..
SET dataRoot=%baseRoot%\data
SET blenderRoot=%dataRoot%\blenderFiles
SET scene=japanesestreet
SET sequence=camera1
SET blenderFile=%blenderRoot%\%scene%\%scene%_%sequence%.blend

blender %blenderFile% -P simulatorBlender.py -- -c %baseRoot%\configs\%scene%_%sequence%.yaml
@REM python simulatorTaichi.py -c %baseRoot%\configs\%scene%_%sequence%.yaml