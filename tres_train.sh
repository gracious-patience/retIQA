cuda=6
backbone_device="cuda:6"
retrieval_device="cuda:6"
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
tid2013_num_classes=120

kadid10k_botnet_finetune="/home/sharfikeg/my_files/retIQA/dc_ret/finetune_botnet50_kadid10k_checkpoints/checkpoint_model_best_heads16.pth"
kadid10k_data_path="/home/sharfikeg/my_files/retIQA/kadid10k/distorted_images/"
kadid10k_ref_path="/home/sharfikeg/my_files/retIQA/kadid10k/reference_images/"
kadid10k_csv_path="/home/sharfikeg/my_files/retIQA/kadid10k/kadid10k_info.csv"
kadid10k_num_classes=125

csiq_botnet_finetune="/home/sharfikeg/my_files/retIQA/dc_ret/finetune_botnet50_csiq_checkpoints/checkpoint_model_best_heads16.pth"
csiq_data_path="/home/sharfikeg/my_files/retIQA/csiq/distorted_images/"
csiq_ref_path="/home/sharfikeg/my_files/retIQA/csiq/src_imgs/"
csiq_csv_path="/home/sharfikeg/my_files/retIQA/csiq/csiq_info.csv"
csiq_num_classes=30

koniq10k_data_path="/home/s-kastryulin/data/koniq10k/512x384/"
koniq10k_csv_path="/home/s-kastryulin/data/koniq10k/koniq10k_info.csv"
# koniq10k_data_path="/home/s-kastryulin/data/koniq10k/1024x768/"

spaq_data_path="/extra_disk_1/sharfikeg/spaq/TestImage/"
spaq_csv_path="/extra_disk_1/sharfikeg/spaq/spaq_info.csv"

biq_data_path="/home/sharfikeg/my_files/extra_disk_1/BIQ2021/Images/"
biq_csv_path="/home/sharfikeg/my_files/extra_disk_1/BIQ2021/biq_info.csv"

pipal_data_path="/home/sharfikeg/my_files/extra_disk_1/pipal/train/Train_Dist/"
pipal_ref_path="/home/sharfikeg/my_files/extra_disk_1/pipal/train/Train_Ref/"
pipal_csv_path="/home/sharfikeg/my_files/extra_disk_1/pipal/train/pipal_info.csv"


# # biq exps
# for i in {1..5}
# do
#     OMP_NUM_THREADS=12 python $tres_launch_training --num_encoder_layerst 2 --dim_feedforwardt 64 --nheadt 16 --network 'resnet50' --batch_size 53  --svpath $tres_save_path --droplr 1 --epochs 5 --gpunum $cuda --datapath '/home/sharfikeg/my_files/extra_disk_1/BIQ2021' --dataset 'biq' --seed $i --vesion 1
# done

# # clive exps
# for i in {1..10}
# do
#     OMP_NUM_THREADS=2 python $tres_launch_training --num_encoder_layerst 2 --dim_feedforwardt 64 --nheadt 16 --network 'resnet50' --batch_size 53  --svpath $tres_save_path --droplr 1 --epochs 5 --gpunum $cuda --datapath '/home/s-kastryulin/data/LIVE-itW' --dataset 'clive' --seed $i --vesion 1
# done


# csiq exps
for i in {4..10}
do
    OMP_NUM_THREADS=2 python $tres_launch_training --num_encoder_layerst 2 --dim_feedforwardt 64 --nheadt 16 --network 'resnet50' --batch_size 53  --svpath /home/sharfikeg/my_files/extra_disk_1/Save_TReS/ --droplr 1 --epochs 5 --gpunum $cuda --datapath '/home/sharfikeg/my_files/retIQA/csiq' --dataset 'csiq' --seed $i --vesion 1
done

# # tid2013 exps
# for i in {1..5}
# do
#     OMP_NUM_THREADS=54 python $tres_launch_training --num_encoder_layerst 2 --dim_feedforwardt 64 --nheadt 16 --network 'resnet50' --batch_size 128  --svpath $tres_save_path --droplr 1 --epochs 5 --gpunum $cuda --datapath '/home/s-kastryulin/data/tid2013' --dataset 'tid2013' --seed $i --vesion 1
# done

# # kadid10k exps
# for i in {1..5}
# do
#     OMP_NUM_THREADS=54 python $tres_launch_training --num_encoder_layerst 2 --dim_feedforwardt 64 --nheadt 16 --network 'resnet50' --batch_size 128  --svpath $tres_save_path --droplr 1 --epochs 5 --gpunum $cuda --datapath '/home/sharfikeg/my_files/retIQA/kadid10k' --dataset 'kadid10k' --seed $i --vesion 1
# done

# # koniq10k exps
# for i in {1..5}
# do
#     OMP_NUM_THREADS=54 python $tres_launch_training --num_encoder_layerst 2 --dim_feedforwardt 64 --nheadt 16 --network 'resnet50' --batch_size 128  --svpath $tres_save_path --droplr 1 --epochs 5 --gpunum $backbone_cuda --datapath '/home/sharfikeg/my_files/retIQA/koniq10k' --dataset 'koniq' --seed $i --vesion 1
# done

# # spaq exps
# for i in {1..5}
# do
#     OMP_NUM_THREADS=54 python $tres_launch_training --num_encoder_layerst 2 --dim_feedforwardt 64 --nheadt 16 --network 'resnet50' --batch_size 128  --svpath $tres_save_path --droplr 1 --epochs 5 --gpunum $backbone_cuda --datapath '/extra_disk_1/sharfikeg/spaq' --dataset 'spaq' --seed $i --vesion 1
# done