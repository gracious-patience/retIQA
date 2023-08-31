cuda=3
backbone_cuda=4
botnet_pretrain="/home/sharfikeg/my_files/VIPNet/pretrained_model/botnet_model_best.pth.tar"
botnet_pretrain_classes=150
my_botnet_pretrain="/home/sharfikeg/my_files/retIQA/dc_ret/my_botnet_pretrain/checkpoint_model_best_heads16.pth"
my_botnet_pretrain_classes=125
logging_path="/home/sharfikeg/my_files/retIQA/dc_ret/DistorsionFeatureExtractor/results.csv"
tres_save_path="/extra_disk_1/sharfikeg/Save_TReS/"
tres_launch_training="/home/sharfikeg/my_files/retIQA/dc_ret/DistorsionFeatureExtractor/TReS/run.py"
tres_batchsize=5
tres_patches=50

tid2013_botnet_finetune="/home/sharfikeg/my_files/retIQA/dc_ret/finetune_botnet50_tid2013_checkpoints/checkpoint_model_best_heads16.pth"
tid2013_data_path="/home/s-kastryulin/data/tid2013/distorted_images/"
tid2013_ref_path="/home/s-kastryulin/data/tid2013/reference_images/"
tid2013_csv_path="/home/sharfikeg/my_files/retIQA/tid2013/tid2013_info.csv"
# tid2013_tres_path="/home/sharfikeg/my_files/retIQA/dc_ret/DistorsionFeatureExtractor/TReS/pretrained_models/tid2013_1_2021/sv/bestmodel_1_2021"
tid2013_num_classes=120

kadid10k_botnet_finetune="/home/sharfikeg/my_files/retIQA/dc_ret/finetune_botnet50_kadid10k_checkpoints/checkpoint_model_best_heads16.pth"
kadid10k_data_path="/home/sharfikeg/my_files/retIQA/kadid10k/distorted_images/"
kadid10k_ref_path="/home/sharfikeg/my_files/retIQA/kadid10k/reference_images/"
kadid10k_csv_path="/home/sharfikeg/my_files/retIQA/kadid10k/kadid10k_info.csv"
# kadid10k_tres_path="/home/sharfikeg/my_files/retIQA/dc_ret/DistorsionFeatureExtractor/TReS/pretrained_models/kadid10k_1_2021/bestmodel_1_2021"
kadid10k_num_classes=125

csiq_botnet_finetune="/home/sharfikeg/my_files/retIQA/dc_ret/finetune_botnet50_csiq_checkpoints/checkpoint_model_best_heads16.pth"
csiq_data_path="/home/sharfikeg/my_files/retIQA/csiq/distorted_images/"
csiq_ref_path="/home/sharfikeg/my_files/retIQA/csiq/src_imgs/"
csiq_csv_path="/home/sharfikeg/my_files/retIQA/csiq/csiq_info.csv"
# csiq_tres_path="/home/sharfikeg/my_files/retIQA/dc_ret/DistorsionFeatureExtractor/TReS/pretrained_models/csiq_1_2021/sv/bestmodel_1_2021"
csiq_num_classes=30

# koniq10k_data_path="/home/s-kastryulin/data/koniq10k/512x384/"
# koniq10k_csv_path="/home/sharfikeg/my_files/retIQA/koniq10k_extra_info.csv"
koniq10k_data_path="/home/s-kastryulin/data/koniq10k/1024x768/"
koniq10k_csv_path="/home/sharfikeg/my_files/retIQA/koniq10k_info.csv"
# koniq10k_tres_pretrain="/home/sharfikeg/my_files/retIQA/dc_ret/DistorsionFeatureExtractor/TReS/pretrained_models/koniq/bestmodel_1_2021"

spaq_data_path="/extra_disk_1/sharfikeg/spaq/TestImage/"
spaq_csv_path="/extra_disk_1/sharfikeg/spaq/spaq_info.csv"


# kadid exps
for i in {5..5}
do
    OMP_NUM_THREADS=54 python $tres_launch_training --num_encoder_layerst 2 --dim_feedforwardt 64 --nheadt 16 --network 'resnet50' --batch_size 128  --svpath $tres_save_path --droplr 1 --epochs 5 --gpunum $backbone_cuda --datapath '/home/sharfikeg/my_files/retIQA/kadid10k' --dataset 'kadid10k' --seed $i --vesion 1
done