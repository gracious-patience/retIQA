backbone_device="cuda:6"
logging_path="/home/sharfikeg/my_files/retIQA/dc_ret/DistorsionFeatureExtractor/results.csv"
tres_save_path="/extra_disk_1/sharfikeg/Save_TReS/"
tres_launch_training="/home/sharfikeg/my_files/retIQA/dc_ret/DistorsionFeatureExtractor/TReS/run.py"
tres_batchsize=2
tres_patches=50


tid2013_botnet_finetune="/home/sharfikeg/my_files/retIQA/dc_ret/finetune_botnet50_tid2013_checkpoints/checkpoint_model_best_heads16.pth"
tid2013_data_path="/home/s-kastryulin/data/tid2013/distorted_images/"
tid2013_ref_path="/home/s-kastryulin/data/tid2013/reference_images/"
tid2013_csv_path="/home/sharfikeg/my_files/retIQA/tid2013/tid2013_info.csv"
tid2013_uni_csv_path="/home/sharfikeg/my_files/retIQA/tid2013/tid2013_uni_info.csv"
tid2013_num_classes=120
k_tid=9

kadid10k_botnet_finetune="/home/sharfikeg/my_files/retIQA/dc_ret/finetune_botnet50_kadid10k_checkpoints/checkpoint_model_best_heads16.pth"
kadid10k_data_path="/home/sharfikeg/my_files/retIQA/kadid10k/distorted_images/"
kadid10k_ref_path="/home/sharfikeg/my_files/retIQA/kadid10k/reference_images/"
kadid10k_csv_path="/home/sharfikeg/my_files/retIQA/kadid10k/kadid10k_info.csv"
kadid10k_uni_csv_path="/home/sharfikeg/my_files/retIQA/kadid10k/kadid10k_uni_info.csv"
kadid10k_num_classes=125
k_kadid=9

csiq_botnet_finetune="/home/sharfikeg/my_files/retIQA/dc_ret/finetune_botnet50_csiq_checkpoints/checkpoint_model_best_heads16.pth"
csiq_data_path="/home/sharfikeg/my_files/retIQA/csiq/distorted_images/"
csiq_ref_path="/home/sharfikeg/my_files/retIQA/csiq/src_imgs/"
csiq_csv_path="/home/sharfikeg/my_files/retIQA/csiq/csiq_info.csv"
csiq_uni_csv_path="/home/sharfikeg/my_files/retIQA/csiq/csiq_uni_info.csv"
csiq_num_classes=30
k_csiq=9

koniq10k_data_path="/home/s-kastryulin/data/koniq10k/512x384/"
koniq10k_csv_path="/home/s-kastryulin/data/koniq10k/koniq10k_info.csv"
koniq10k_big_data_path="/home/s-kastryulin/data/koniq10k/1024x768/"
koniq10k_big_csv_path="/home/s-kastryulin/data/koniq10k/koniq10k_big_info.csv"
koniq10k_uni_csv_path="/home/s-kastryulin/data/koniq10k/koniq10k_uni_info.csv"
k_koniq=30

spaq_data_path="/extra_disk_1/sharfikeg/spaq/TestImage/"
spaq_csv_path="/extra_disk_1/sharfikeg/spaq/spaq_info.csv"
spaq_uni_csv_path="/home/sharfikeg/my_files/extra_disk_1/spaq/spaq_uni_info.csv"
k_spaq=50

biq_data_path="/home/sharfikeg/my_files/extra_disk_1/BIQ2021/Images/"
biq_csv_path="/home/sharfikeg/my_files/extra_disk_1/BIQ2021/biq_info.csv"
biq_uni_csv_path="/home/sharfikeg/my_files/extra_disk_1/BIQ2021/biq_uni_info.csv"
k_biq=30

pipal_data_path="/home/sharfikeg/my_files/extra_disk_1/pipal/train/Train_Dist/"
pipal_ref_path="/home/sharfikeg/my_files/extra_disk_1/pipal/train/Train_Ref/"
pipal_csv_path="/home/sharfikeg/my_files/extra_disk_1/pipal/train/pipal_info.csv"
pipal_uni_csv_path="/home/sharfikeg/my_files/extra_disk_1/pipal/train/pipal_uni_info.csv"
pipal_num_seeds=5
k_pipal=20

# pipal exps
exp1=" --dataset pipal --data_path $pipal_data_path --ref_path $pipal_ref_path --csv_path $pipal_csv_path --batch_size2 $tres_batchsize --batch_size 12 --uni 0 --backbone_device $backbone_device --num_seeds $pipal_num_seeds  "
python3 pure_tres.py $exp1
