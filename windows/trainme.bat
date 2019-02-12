REM Anaconda prompt
REM Edit: substitute 'cda' by your windows user name
call C:\Users\cda\Anaconda3\Scripts\activate.bat C:\Users\cda\Anaconda3
REM Activate environment
call activate tensorflow
REM Execute Python script
cd ..
python train.py --experiment_rootdir=./models/test_1 ^
--train_dir=./input/thermal_hand_gesture_recognition_dataset_80x60/training/ ^
--val_dir=./input/thermal_hand_gesture_recognition_dataset_80x60/validation/ ^
--img_mode=rgb 
REM Leave command windows opened
cmd /k