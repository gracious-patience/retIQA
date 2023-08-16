botnet_pretrain="/home/sharfikeg/my_files/VIPNet/pretrained_model/botnet_model_best.pth.tar"
logging_path="/home/sharfikeg/my_files/retIQA/dc_ret/DistorsionFeatureExtractor/results.csv"

tid2013_botnet_finetune="/home/sharfikeg/my_files/retIQA/dc_ret/finetune_botnet50_tid2013_checkpoints/checkpoint_model_best_heads16.pth"
tid2013_data_path="/home/s-kastryulin/data/tid2013/distorted_images/"
tid2013_ref_path="/home/s-kastryulin/data/tid2013/reference_images/"
tid2013_csv_path="/home/sharfikeg/my_files/retIQA/tid2013/tid2013_info.csv"

kadid10k_botnet_finetune="/home/sharfikeg/my_files/retIQA/dc_ret/finetune_botnet50_kadid10k_checkpoints/checkpoint_model_best_heads16.pth"
kadid10k_data_path="/home/sharfikeg/my_files/retIQA/kadid10k/distorted_images/"
kadid10k_ref_path="/home/sharfikeg/my_files/retIQA/kadid10k/reference_images/"
kadid10k_csv_path="/home/sharfikeg/my_files/retIQA/kadid10k/kadid10k_info.csv"

csiq_botnet_finetune="/home/sharfikeg/my_files/retIQA/dc_ret/finetune_botnet50_csiq_checkpoints/checkpoint_model_best_heads16.pth"
csiq_data_path="/home/sharfikeg/my_files/retIQA/csiq/distorted_images/"
csiq_ref_path="/home/sharfikeg/my_files/retIQA/csiq/reference_images/"
csiq_csv_path="/home/sharfikeg/my_files/retIQA/csiq/csiq_info.csv"

# experiments
exp1="--model finetune_botnet50 --finetune 0 --img_width 288 --img_height 384 --num_classes 125 --num_heads 16 --dataset tid2013 --data_path $tid2013_data_path --ref_path $tid2013_ref_path --batch_size 128 --batch_size2 16 --num_workers 24 --lr 0.005 --seed 5 --csv_path $tid2013_csv_path --botnet_pretrain $tid2013_botnet_finetune --device_num 0 --logging_path $logging_path --k 9 --aggregation averaging --epochs 30"
exp2="--model finetune_botnet50 --finetune 0 --img_width 288 --img_height 384 --num_classes 125 --num_heads 16 --dataset kadid10k --data_path $kadid10k_data_path --ref_path $kadid10k_ref_path --batch_size 128 --batch_size2 16 --num_workers 24 --lr 0.005 --seed 5 --csv_path $kadid10k_csv_path --botnet_pretrain $kadid10k_botnet_finetune --device_num 0 --logging_path $logging_path --k 9 --aggregation averaging --epochs 30"

python3 main.py $exp1
python3 main $exp2