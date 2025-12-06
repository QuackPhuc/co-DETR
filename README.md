Cấu trúc thư mục

```
Co-DETR-pytorch/
|   .gitignore
|   PROJECT_STATUS.md
|   README.md
|   requirements.txt
|   setup.py
|
+---.project
|   \---plan
|           co-detr-pytorch-reimplementation.md
|
+---codetr
|   |   __init__.py
|   |   
|   +---configs
|   |   |   config.py
|   |   |   __init__.py
|   |   |
|   |   \---defaults
|   |           co_deformable_detr_r50.yaml
|   |
|   +---data
|   |   |   dataloader.py
|   |   |   __init__.py
|   |   |
|   |   +---datasets
|   |   |       yolo_dataset.py
|   |   |       __init__.py
|   |   |
|   |   \---transforms
|   |           transforms.py
|   |           __init__.py
|   |
|   +---engine
|   |       evaluator.py
|   |       hooks.py
|   |       lr_scheduler.py
|   |       trainer.py
|   |       __init__.py
|   |
|   +---models
|   |   |   detector.py
|   |   |   __init__.py
|   |   |
|   |   +---backbone
|   |   |       resnet.py
|   |   |       __init__.py
|   |   |
|   |   +---heads
|   |   |       atss_head.py
|   |   |       detr_head.py
|   |   |       roi_head.py
|   |   |       rpn_head.py
|   |   |       __init__.py
|   |   |
|   |   +---losses
|   |   |       focal_loss.py
|   |   |       giou_loss.py
|   |   |       l1_loss.py
|   |   |       __init__.py
|   |   |
|   |   +---matchers
|   |   |       hungarian_matcher.py
|   |   |       __init__.py
|   |   |
|   |   +---neck
|   |   |       channel_mapper.py
|   |   |       __init__.py
|   |   |
|   |   +---transformer
|   |   |       attention.py
|   |   |       decoder.py
|   |   |       encoder.py
|   |   |       transformer.py
|   |   |       __init__.py
|   |   |
|   |   \---utils
|   |           box_ops.py
|   |           misc.py
|   |           position_encoding.py
|   |           query_denoising.py
|   |           __init__.py
|   |
|   \---utils
|           distributed.py
|           __init__.py
|
+---codetr.egg-info
|       dependency_links.txt
|       PKG-INFO
|       requires.txt
|       SOURCES.txt
|       top_level.txt
|
+---configs
|       co_deformable_detr_r50_yolo.yaml
|       overfit_test.yaml
|
+---docs
+---mmdet-version
|   |   __init__.py
|   |
|   +---configs
|   |   +---co_deformable_detr
|   |   |       co_deformable_detr_mask_r50_1x_coco.py
|   |   |       co_deformable_detr_r50_1x_coco.py
|   |   |       co_deformable_detr_swin_base_1x_coco.py
|   |   |       co_deformable_detr_swin_base_3x_coco.py
|   |   |       co_deformable_detr_swin_large_1x_coco.py
|   |   |       co_deformable_detr_swin_large_900q_3x_coco.py
|   |   |       co_deformable_detr_swin_small_1x_coco.py
|   |   |       co_deformable_detr_swin_small_3x_coco.py
|   |   |       co_deformable_detr_swin_tiny_1x_coco.py
|   |   |       co_deformable_detr_swin_tiny_3x_coco.py
|   |   |
|   |   +---co_dino
|   |   |       co_dino_5scale_9encoder_lsj_r50_1x_coco.py
|   |   |       co_dino_5scale_9encoder_lsj_r50_3x_coco.py
|   |   |       co_dino_5scale_lsj_r50_1x_coco.py
|   |   |       co_dino_5scale_lsj_r50_1x_lvis.py
|   |   |       co_dino_5scale_lsj_r50_3x_coco.py
|   |   |       co_dino_5scale_lsj_swin_large_16e_o365tolvis.py
|   |   |       co_dino_5scale_lsj_swin_large_1x_coco.py
|   |   |       co_dino_5scale_lsj_swin_large_2x_coco.py
|   |   |       co_dino_5scale_lsj_swin_large_3x_coco.py
|   |   |       co_dino_5scale_lsj_swin_large_3x_lvis.py
|   |   |       co_dino_5scale_r50_1x_coco.py
|   |   |       co_dino_5scale_r50_1x_lvis.py
|   |   |       co_dino_5scale_swin_large_16e_o365tococo.py
|   |   |       co_dino_5scale_swin_large_1x_coco.py
|   |   |       co_dino_5scale_swin_large_2x_coco.py
|   |   |       co_dino_5scale_swin_large_3x_coco.py
|   |   |
|   |   +---co_dino_vit
|   |   |       co_dino_5scale_lsj_vit_large_lvis.py
|   |   |       co_dino_5scale_lsj_vit_large_lvis_instance.py
|   |   |       co_dino_5scale_vit_large_coco.py
|   |   |       co_dino_5scale_vit_large_coco_instance.py
|   |   |
|   |   \---_base_
|   |       |   default_runtime.py
|   |       |
|   |       +---datasets
|   |       |       cityscapes_detection.py
|   |       |       cityscapes_instance.py
|   |       |       coco_detection.py
|   |       |       coco_instance.py
|   |       |       coco_instance_semantic.py
|   |       |       coco_panoptic.py
|   |       |       deepfashion.py
|   |       |       lvis_v0.5_instance.py
|   |       |       lvis_v1_instance.py
|   |       |       openimages_detection.py
|   |       |       voc0712.py
|   |       |       wider_face.py
|   |       |
|   |       +---models
|   |       |       cascade_mask_rcnn_r50_fpn.py
|   |       |       cascade_rcnn_r50_fpn.py
|   |       |       faster_rcnn_r50_caffe_c4.py
|   |       |       faster_rcnn_r50_caffe_dc5.py
|   |       |       faster_rcnn_r50_fpn.py
|   |       |       fast_rcnn_r50_fpn.py
|   |       |       mask_rcnn_r50_caffe_c4.py
|   |       |       mask_rcnn_r50_fpn.py
|   |       |       retinanet_r50_fpn.py
|   |       |       rpn_r50_caffe_c4.py
|   |       |       rpn_r50_fpn.py
|   |       |       ssd300.py
|   |       |
|   |       \---schedules
|   |               schedule_1x.py
|   |               schedule_20e.py
|   |               schedule_2x.py
|   |
|   \---models
|           co_atss_head.py
|           co_deformable_detr_head.py
|           co_detr.py
|           co_dino_head.py
|           co_roi_head.py
|           norm.py
|           query_denoising.py
|           swin_transformer.py
|           transformer.py
|           __init__.py
|
+---tests
|   |   conftest.py
|   |   test_backbone.py
|   |   test_config.py
|   |   test_detector.py
|   |   test_heads.py
|   |   test_integration.py
|   |   test_losses.py
|   |   test_matchers.py
|   |   test_neck.py
|   |   test_transformer.py
|   |   __init__.py
|   |
|   +---test_data
|   |       test_dataloader.py
|   |       test_dataset.py
|   |       test_transforms.py
|   |       __init__.py
|   |
|   +---test_engine
|   |       test_evaluator.py
|   |       test_trainer.py
|   |       __init__.py
|   |
|   \---test_utils
|           test_box_ops.py
|           test_misc.py
|           test_position_encoding.py
|           __init__.py
|
\---tools
        convert_weights.py
        create_sample_dataset.py
        inference.py
        train.py
        visualize.py
```
