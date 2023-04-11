CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --master_port=29504 train_final_dist.py \
	--batch_size=100 \
	--train_data {YOUR_TRAIN_DATA_PATH} \
	--select_data=MJ-ST \
	--batch_ratio 0.5-0.5\
	--valid_data {YOUR_EVAL_DATA_PATH} \
	--valInterval 5000 \
	--num_iter 2000000 \
	--scheduler \
    --T_0=2000000 \
	--lr 0.3 \
	--workers=10 \
	--isrand_aug \
	--rgb \
	--imgH=32 \
	--imgW=128 \
	--CRM \
	--backbone_type=Tiny \
    --SVTRPatchEmbed \
    --definition_string=VLV \
	--supervised_blocks=0,1,2 \
	--MultiCELoss=3 \
	--SE \
	--saved_path=vlcr_train \
	--exp_name=tiny \
	--amp \