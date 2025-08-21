python demo.py --config-file ../configs/cvs/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048_fold1.yaml \
--input /media/SSD1/scanar/endovis/LEMIS-challenge/region_proposals/data/sages/frames/video_001/* \
--output outputs \
--opts MODEL.WEIGHTS /home/jclyons/endovis/challenge_2025/MaskDINO/outputs/cvs-challenge_pretrained_endoscapes_fold1_2/model_best.pth