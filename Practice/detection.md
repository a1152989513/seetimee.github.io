跑通后的准确率
![image](https://user-images.githubusercontent.com/50852027/154178095-6021d593-bb5c-45a4-9cec-37f971f6a4f1.png)

```python

!git clone https://gitee.com/paddlepaddle/PaddleOCR

import os
# 修改代码运行的默认目录为 /home/aistudio/PaddleOCR
os.chdir("/home/aistudio/PaddleOCR")
# 安装PaddleOCR第三方依赖
!pip install --upgrade pip
!pip install -r requirements.txt

# --image_dir 指向要预测的图像路径  --rec false表示不使用识别识别，只执行文本检测
! paddleocr --image_dir ./PaddleOCR/doc/imgs/12.jpg --rec false


%cd /home/aistudio/work
!wget https://paddleocr.bj.bcebos.com/dataset/det_data_lesson_demo.tar


%cd /home/aistudio/work
!tar xvf det_data_lesson_demo.tar -C /home/aistudio/data/

!wget -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_5_pretrained.pdparams
```

```
#det_mv3_db.yml参考，跑80轮hmean大概0.6，替换/home/aistudio/PaddleOCR/configs/det/det_mv3_db.yml文件内容
Global:
  use_gpu: true
  epoch_num: 200
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: ./output/db_mv3/
  save_epoch_step: 500
  # evaluation is run every 2000 iterations
  eval_batch_step: [0, 500]
  cal_metric_during_train: False
  pretrained_model: /home/aistudio/work/pretrain_models/MobileNetV3_large_x0_5_pretrained
  checkpoints:
  save_inference_dir:
  use_visualdl: False
  infer_img: doc/imgs_en/img_10.jpg
  save_res_path: ./output/det_db/predicts_db.txt

Architecture:
  model_type: det
  algorithm: DB
  Transform:
  Backbone:
    name: MobileNetV3
    scale: 0.5
    model_name: large
  Neck:
    name: DBFPN
    out_channels: 256
  Head:
    name: DBHead
    k: 50

Loss:
  name: DBLoss
  balance_loss: true
  main_loss_type: DiceLoss
  alpha: 5
  beta: 10
  ohem_ratio: 3

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Step
    learning_rate: 0.0005
    step_size: 80
    gamma: 0.1
  regularizer:
    name: 'L2'
    factor: 0

PostProcess:
  name: DBPostProcess
  thresh: 0.3
  box_thresh: 0.6
  max_candidates: 1000
  unclip_ratio: 1.5

Metric:
  name: DetMetric
  main_indicator: hmean

Train:
  dataset:
    name: SimpleDataSet
    data_dir: /home/aistudio/data/det_data_lesson_demo
    label_file_list:
      - /home/aistudio/data/det_data_lesson_demo/train.txt
    ratio_list: [1.0]
    transforms:
      - DecodeImage: # load image
          img_mode: BGR
          channel_first: False
      - DetLabelEncode: # Class handling label
      - IaaAugment:
          augmenter_args:
            - { 'type': Fliplr, 'args': { 'p': 0.5 } }
            - { 'type': Affine, 'args': { 'rotate': [-10, 10] } }
            - { 'type': Resize, 'args': { 'size': [0.5, 3] } }
      - EastRandomCropData:
          size: [640, 640]
          max_tries: 50
          keep_ratio: true
      - MakeBorderMap:
          shrink_ratio: 0.5
          thresh_min: 0.4
          thresh_max: 0.8
      - MakeShrinkMap:
          shrink_ratio: 0.4
          min_text_size: 8
      - NormalizeImage:
          scale: 1./255.
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
          order: 'hwc'
      - ToCHWImage:
      - KeepKeys:
          keep_keys: ['image', 'threshold_map', 'threshold_mask', 'shrink_map', 'shrink_mask'] # the order of the dataloader list
  loader:
    shuffle: True
    drop_last: False
    batch_size_per_card: 16
    num_workers: 1
    use_shared_memory: False

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: /home/aistudio/data/det_data_lesson_demo/
    label_file_list:
      - /home/aistudio/data/det_data_lesson_demo/eval.txt
    transforms:
      - DecodeImage: # load image
          img_mode: BGR
          channel_first: False
      - DetLabelEncode: # Class handling label
      - DetResizeForTest:
          image_shape: [736, 1280]
      - NormalizeImage:
          scale: 1./255.
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
          order: 'hwc'
      - ToCHWImage:
      - KeepKeys:
          keep_keys: ['image', 'shape', 'polys', 'ignore_tags']
  loader:
    shuffle: False
    drop_last: False
    batch_size_per_card: 1 # must be 1
    num_workers: 1
    use_shared_memory: False
```

```python
#%cd /home/aistudio/work
#!wget -P ./train_data/  https://paddleocr.bj.bcebos.com/dataset/train_icdar2015_label.txt
#!wget -P ./train_data/  https://paddleocr.bj.bcebos.com/dataset/test_icdar2015_label.txt
print('hello')


断点训练加上 -o 参数
%cd /home/aistudio/PaddleOCR 
!python tools/train.py -c configs/det/det_mv3_db.yml -o Global.pretrained_model=./pretrain_models//MobileNetV3_large_x0_5_pretrained.pdparams


#评估脚本
%cd /home/aistudio/PaddleOCR
!python tools/eval.py -c configs/det/det_mv3_db.yml -o Global.checkpoints=./output/db_mv3/best_accuracy

#模型位置
%cd /home/aistudio/PaddleOCR
!ls output/db_mv3/best_accuracy.*

#训练日志
%cd /home/aistudio/PaddleOCR
!tailf output/db_mv3/train.log
```

