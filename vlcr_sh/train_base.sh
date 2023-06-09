CUDA_VISIBLE_DEVICES=1,2,3,4,5 python3 -m torch.distributed.launch --nproc_per_node=5 --nnodes=1 --master_port=29500 train_final_dist.py \
	--batch_size=40 \
	--train_data {YOUR_TRAIN_DATA_PATH} \
	--select_data=MJ-ST \
	--batch_ratio 0.5-0.5\
	--valid_data {YOUR_EVAL_DATA_PATH} \
	--valInterval 5000 \
	--num_iter 1000000 \
	--scheduler \
    --T_0=1000000 \
	--lr 0.6 \
	--workers=10 \
	--isrand_aug \
	--rgb \
	--imgH=32 \
	--imgW=128 \
	--CRM \
	--backbone_type=Base \
    --SVTRPatchEmbed \
    --definition_string=VLVLVLV \
	--supervised_blocks=0,1,2,3,4,5,6 \
	--MultiCELoss=7 \
	--SE \
	--saved_path=vlcr_train \
	--exp_name=base \
	--amp \