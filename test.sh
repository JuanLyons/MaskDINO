python train_net.py --eval-only --num-gpus 1 \
--config-file configs/cvs/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048_fold1.yaml \
OUTPUT_DIR outputs/cvs-challenge_pretrained_endoscapes_fold1_2_test \
MODEL.WEIGHTS outputs/cvs-challenge_pretrained_endoscapes_fold1_2/model_best.pth

python train_net.py --eval-only --num-gpus 1 \
--config-file configs/cvs/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048_fold2.yaml \
OUTPUT_DIR outputs/cvs-challenge_pretrained_endoscapes_fold2_2_test \
MODEL.WEIGHTS outputs/cvs-challenge_pretrained_endoscapes_fold2_2/model_best.pth