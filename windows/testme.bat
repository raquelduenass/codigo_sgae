REM Anaconda prompt
REM Edit: substitute 'cda' by your windows user name
call C:\Users\cda\Anaconda3\Scripts\activate.bat C:\Users\cda\Anaconda3
REM Activate environment
call activate tensorflow
REM Execute Python script
cd ..
python test.py --experiment_rootdir=./models ^
--weights_fname=./models/weights_064.h5 ^
--test_dir=./input/thermal_hand_gesture_recognition_dataset_80x60/testing/ ^
--img_mode=rgb
REM Leave command windows opened
cmd /k