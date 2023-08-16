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

# tid2013 exps
for i in {1..15}
do
    exp1="--model finetune_botnet50 --finetune 1 --retrieve 1 --num_iters 1 --img_width 288 --img_height 384 --num_classes 125 --num_heads 16 --dataset tid2013 --data_path $tid2013_data_path --ref_path $tid2013_ref_path --batch_size 128 --batch_size2 32 --num_workers 12 --lr 0.005 --seed $i --csv_path $tid2013_csv_path --botnet_pretrain $botnet_pretrain --device_num 0 --logging_path $logging_path --k 9 --aggregation averaging --epochs 30"
    python3 main.py $exp1
done

# kadid10k exps
for i in {1..15}
do
    exp2="--model finetune_botnet50 --finetune 0 --retrieve 1 --num_iters 1 --img_width 288 --img_height 384 --num_classes 125 --num_heads 16 --dataset kadid10k --data_path $kadid10k_data_path --ref_path $kadid10k_ref_path --batch_size 128 --batch_size2 32 --num_workers 12 --lr 0.005 --seed $i --csv_path $kadid10k_csv_path --botnet_pretrain $botnet_pretrain --device_num 0 --logging_path $logging_path --k 9 --aggregation averaging --epochs 30"
    python3 main.py $exp2
done

# csiq exps
for i in {1..15}
do
    exp3="--model finetune_botnet50 --finetune 0 --retrieve 1 --num_iters 1 --img_width 288 --img_height 384 --num_classes 125 --num_heads 16 --dataset csiq --data_path $csiq_data_path --ref_path $csiq_ref_path --batch_size 128 --batch_size2 32 --num_workers 12 --lr 0.005 --seed $i --csv_path $csiq_csv_path --botnet_pretrain $botnet_pretrain --device_num 0 --logging_path $logging_path --k 9 --aggregation averaging --epochs 30"
    python3 main.py $exp2
done
