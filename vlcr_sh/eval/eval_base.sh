CUDA_VISIBLE_DEVICES=0 python3 test_final.py --eval_data {YOUR_EVAL_DATA_PATH} --benchmark_all_eval \
    --data_filtering_off --rgb --fast_acc \
    --model_dir=vlcr_model_weight/base.pth \
    --imgH=32 --imgW=128 \
    --CRM \
    --SVTRPatchEmbed \
	--backbone_type=Base \
    --SE \
    --definition_string=VLVLVLV \
	--supervised_blocks=0,1,2,3,4,5,6 \
	--MultiCELoss=7 \
    --amp

