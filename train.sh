# python train_net.py --num-gpus 3 \
# --config-file configs/endoscapes/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml \
# OUTPUT_DIR outputs/endoscapes_7 \
# MODEL.WEIGHTS maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth

python train_net.py --num-gpus 3 \
--config-file configs/cvs/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048_fold1.yaml \
OUTPUT_DIR outputs/cvs-challenge_pretrained_endoscapes_fold1_3 \
MODEL.WEIGHTS outputs/endoscapes_6/model_best.pth

python train_net.py --num-gpus 3 \
--config-file configs/cvs/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048_fold2.yaml \
OUTPUT_DIR outputs/cvs-challenge_pretrained_endoscapes_fold2_3 \
MODEL.WEIGHTS outputs/endoscapes_6/model_best.pth