CUDA_VISIBLE_DEVICES=4 python3 test_final.py --eval_data {YOUR_EVAL_DATA_PATH} --benchmark_all_eval \
    --data_filtering_off --rgb --fast_acc \
    --model_dir=vlcr_model_weight/tiny.pth \
    --imgH=32 --imgW=128 \
	--CRM \
	--SVTRPatchEmbed \
	--backbone_type=Tiny \
	--SE \
    --definition_string=VLV \
	--supervised_blocks=0,1,2 \
	--MultiCELoss=3 \
	--amp