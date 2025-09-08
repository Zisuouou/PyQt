@echo off
echo [BAT] ENABLE_XLA=%ENABLE_XLA%
title Faster RCNN Training
set PYTHONPATH=%PYTHONPATH%;D:/AI_SVT_Training_mk;D:/AI_SVT_Training_mk/slim
python D:/AI_SVT_Training_mk/autoMakeTrain.py
start /b board.bat
python D:/AI_SVT_Training_mk/object_detection/create_tf_record.py
python D:/AI_SVT_Training_mk/object_detection/model_main_tf2_FRCNN_res50.py
python D:/AI_SVT_Training_mk/model_main_tf2.py ^
  --pipeline_config_path=D:/AI_SVT_Training_mk/configs/faster_rcnn_resnet50_v1_800x1333_batch1.config ^
  --model_dir=D:/AI_SVT_Training_mk/train_result ^
  --alsologtostderr
pause

