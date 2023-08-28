cuda=0
backbone_cuda=2
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

koniq10k_data_path="/home/s-kastryulin/data/koniq10k/512x384/"
koniq10k_csv_path="/home/sharfikeg/my_files/retIQA/koniq10k_extra_info.csv"
# koniq10k_data_path="/home/s-kastryulin/data/koniq10k/1024x768/"
# koniq10k_csv_path="/home/sharfikeg/my_files/retIQA/koniq10k_info.csv"
# koniq10k_tres_pretrain="/home/sharfikeg/my_files/retIQA/dc_ret/DistorsionFeatureExtractor/TReS/pretrained_models/koniq/bestmodel_1_2021"

spaq_data_path="/extra_disk_1/sharfikeg/spaq/TestImage/"
spaq_csv_path="/extra_disk_1/sharfikeg/spaq/spaq_info.csv"

liveitw_test_data_path="/home/s-kastryulin/data/LIVE-itW/Images/"
liveitw_train_data_path="/home/s-kastryulin/data/koniq10k/1024x768/"
liveitw_train_csv_path="/home/sharfikeg/my_files/retIQA/liveitw_info.csv"
liveitw_test_csv_path="/home/sharfikeg/my_files/retIQA/liveitw_test_info.csv"
# liveitw_tres_path="/home/sharfikeg/my_files/retIQA/dc_ret/DistorsionFeatureExtractor/TReS/pretrained_models/liveitw/bestmodel_1_2021"

# experiments




# my pretrain, no finetune
# no reference setup
# retrieval

# koniq10k exps
# for i in {1..1}
# do
#     exp1="--model finetune_botnet50 --finetune 0 --retrieve 1 --ret_tr resize --num_iters 1 --img_width 288 --img_height 384 --num_classes $my_botnet_pretrain_classes --pretrain_classes $my_botnet_pretrain_classes --num_heads 16 --dataset koniq10k --data_path $koniq10k_data_path --batch_size 96 --batch_size2 8 --num_workers 12 --lr 0.005 --seed $i --csv_path $koniq10k_csv_path --botnet_pretrain $my_botnet_pretrain --baseline_pretrain $koniq10k_tres_pretrain --device_num $cuda --backbone_device_num $backbone_cuda --logging_path $logging_path --k 9 --aggregation averaging --epochs 30 --baseline no --setup no_reference --patches 1"
#     python3 main.py $exp1
# done

# # live-itw exps
# for i in {1..1}
# do
#     exp1="--model finetune_botnet50 --finetune 0 --retrieve 1 --ret_tr resize --num_iters 1 --img_width 288 --img_height 384 --num_classes $my_botnet_pretrain_classes --pretrain_classes $my_botnet_pretrain_classes --num_heads 16 --dataset liveitw --data_path $liveitw_train_data_path --batch_size 96 --batch_size2 8 --num_workers 12 --lr 0.005 --seed $i --csv_path $liveitw_train_csv_path --botnet_pretrain $my_botnet_pretrain --baseline_pretrain $liveitw_tres_path --device_num $cuda --backbone_device_num $backbone_cuda --logging_path $logging_path --k 9 --aggregation averaging --epochs 30 --baseline no --setup no_reference --patches 1 --test_data_path $liveitw_test_data_path --test_csv_path $liveitw_test_csv_path"
#     python3 main.py $exp1
# done

# # tid2013 exps
# for i in {1..20}
# do
#     exp1="--model finetune_botnet50 --finetune 0 --retrieve 1 --ret_tr resize --num_iters 1 --img_width 288 --img_height 384 --num_classes $tid2013_num_classes --pretrain_classes $my_botnet_pretrain_classes --num_heads 16 --dataset tid2013 --data_path $tid2013_data_path --ref_path $tid2013_ref_path --batch_size 128 --batch_size2 96 --num_workers 12 --lr 0.005 --seed $i --csv_path $tid2013_csv_path --botnet_pretrain $my_botnet_pretrain --baseline_pretrain $tid2013_tres_path --device_num $cuda --backbone_device_num $backbone_cuda --logging_path $logging_path --k 9 --aggregation averaging --epochs 30 --baseline no --setup no_reference --patches 1"
#     python3 main.py $exp1
# done

# # csiq exps
# for i in {1..20}
# do
#     exp3="--model finetune_botnet50 --finetune 0 --retrieve 1 --ret_tr resize --num_iters 1 --img_width 288 --img_height 384 --num_classes $csiq_num_classes --pretrain_classes $my_botnet_pretrain_classes --num_heads 16 --dataset csiq --data_path $csiq_data_path --ref_path $csiq_ref_path --batch_size 128 --batch_size2 96 --num_workers 12 --lr 0.005 --seed $i --csv_path $csiq_csv_path --botnet_pretrain $my_botnet_pretrain --baseline_pretrain $csiq_tres_path --device_num $cuda --backbone_device_num $backbone_cuda --logging_path $logging_path --k 9 --aggregation averaging --epochs 30 --baseline no --setup no_reference --patches 1"
#     python3 main.py $exp3
# done

# # kadid10k exps
# for i in {1..20}
# do
#     exp2="--model finetune_botnet50 --finetune 0 --retrieve 1 --ret_tr resize --num_iters 1 --img_width 288 --img_height 384 --num_classes $kadid10k_num_classes --pretrain_classes $my_botnet_pretrain_classes --num_heads 16 --dataset kadid10k --data_path $kadid10k_data_path --ref_path $kadid10k_ref_path --batch_size 128 --batch_size2 96 --num_workers 12 --lr 0.005 --seed $i --csv_path $kadid10k_csv_path --botnet_pretrain $my_botnet_pretrain --baseline_pretrain $kadid10k_tres_path --device_num $cuda --backbone_device_num $backbone_cuda --logging_path $logging_path --k 9 --aggregation averaging --epochs 30 --baseline no --setup no_reference --patches 1"
#     python3 main.py $exp2
# done




# my pretrain, no finetune
# no reference setup
# train TReS from scratch
# TReS + retrieval

# csiq exps
# for i in {1..10}
# do
#     exp3="--model finetune_botnet50 --finetune 0 --retrieve 1 --ret_tr resize --num_iters 1 --img_width 288 --img_height 384 --num_classes $csiq_num_classes --pretrain_classes $my_botnet_pretrain_classes --num_heads 16 --dataset csiq --data_path $csiq_data_path --ref_path $csiq_ref_path --batch_size 128 --batch_size2 $tres_batchsize --num_workers 12 --lr 0.005 --seed $i --csv_path $csiq_csv_path --botnet_pretrain $my_botnet_pretrain --baseline_pretrain ${tres_save_path}csiq_1_${i}/sv/bestmodel_1_${i} --device_num $cuda --backbone_device_num $backbone_cuda --logging_path $logging_path --k 9 --aggregation averaging --epochs 30 --baseline tres --setup no_reference --patches $tres_patches"
#     OMP_NUM_THREADS=54 python $tres_launch_training --num_encoder_layerst 2 --dim_feedforwardt 64 --nheadt 16 --network 'resnet50' --batch_size 128  --svpath $tres_save_path --droplr 1 --epochs 5 --gpunum $backbone_cuda --datapath '/home/sharfikeg/my_files/retIQA/csiq' --dataset 'csiq' --seed $i --vesion 1
#     python3 main.py $exp3
# done

# tid2013
# for i in {1..2}
# do
#     exp1="--model finetune_botnet50 --finetune 0 --retrieve 1 --ret_tr resize --num_iters 1 --img_width 288 --img_height 384 --num_classes $tid2013_num_classes --pretrain_classes $my_botnet_pretrain_classes --num_heads 16 --dataset tid2013 --data_path $tid2013_data_path --ref_path $tid2013_ref_path --batch_size 128 --batch_size2 $tres_batchsize --num_workers 12 --lr 0.005 --seed $i --csv_path $tid2013_csv_path --botnet_pretrain $my_botnet_pretrain --baseline_pretrain ${tres_save_path}tid2013_1_${i}/sv/bestmodel_1_${i} --device_num $cuda --backbone_device_num $backbone_cuda --logging_path $logging_path --k 9 --aggregation averaging --epochs 30 --baseline tres --setup no_reference --patches $tres_patches"
#     python3 main.py $exp1
# done
# for i in {1..10}
# do
#     exp1="--model finetune_botnet50 --finetune 0 --retrieve 1 --ret_tr resize --num_iters 1 --img_width 288 --img_height 384 --num_classes $tid2013_num_classes --pretrain_classes $my_botnet_pretrain_classes --num_heads 16 --dataset tid2013 --data_path $tid2013_data_path --ref_path $tid2013_ref_path --batch_size 128 --batch_size2 $tres_batchsize --num_workers 12 --lr 0.005 --seed $i --csv_path $tid2013_csv_path --botnet_pretrain $my_botnet_pretrain --baseline_pretrain ${tres_save_path}tid2013_1_${i}/sv/bestmodel_1_${i} --device_num $cuda --backbone_device_num $backbone_cuda --logging_path $logging_path --k 9 --aggregation averaging --epochs 30 --baseline tres --setup no_reference --patches $tres_patches"
#     OMP_NUM_THREADS=54 python $tres_launch_training --num_encoder_layerst 2 --dim_feedforwardt 64 --nheadt 16 --network 'resnet50' --batch_size 128  --svpath $tres_save_path --droplr 1 --epochs 5 --gpunum $backbone_cuda --datapath '/home/s-kastryulin/data/tid2013' --dataset 'tid2013' --seed $i --vesion 1
#     # python3 main.py $exp1
# done



# # live-itw exps
# for i in {1..7}
# do
#     exp1="--model finetune_botnet50 --finetune 0 --retrieve 1 --ret_tr resize --num_iters 1 --img_width 288 --img_height 384 --num_classes $my_botnet_pretrain_classes --pretrain_classes $my_botnet_pretrain_classes --num_heads 16 --dataset liveitw --data_path $liveitw_train_data_path --batch_size 96 --batch_size2 $tres_batchsize --num_workers 12 --lr 0.005 --seed $i --csv_path $liveitw_train_csv_path --botnet_pretrain $my_botnet_pretrain --baseline_pretrain $liveitw_tres_path --device_num $cuda --backbone_device_num $backbone_cuda --logging_path $logging_path --k 9 --aggregation averaging --epochs 30 --baseline tres --setup no_reference --patches $tres_patches --test_data_path $liveitw_test_data_path --test_csv_path $liveitw_test_csv_path"
#     python3 main.py $exp1
# done

# kadid10k exps
for i in {3..10}
do
    exp2="--model finetune_botnet50 --finetune 0 --retrieve 1 --ret_tr resize --num_iters 1 --img_width 288 --img_height 384 --num_classes $kadid10k_num_classes --pretrain_classes $my_botnet_pretrain_classes --num_heads 16 --dataset kadid10k --data_path $kadid10k_data_path --ref_path $kadid10k_ref_path --batch_size 128 --batch_size2 $tres_batchsize --num_workers 12 --lr 0.005 --seed $i --csv_path $kadid10k_csv_path --botnet_pretrain $my_botnet_pretrain --baseline_pretrain ${tres_save_path}kadid10k_1_${i}/sv/bestmodel_1_${i} --device_num $cuda --backbone_device_num $backbone_cuda --logging_path $logging_path --k 9 --aggregation averaging --epochs 30 --baseline tres --setup no_reference --patches $tres_patches"
    OMP_NUM_THREADS=54 python $tres_launch_training --num_encoder_layerst 2 --dim_feedforwardt 64 --nheadt 16 --network 'resnet50' --batch_size 128  --svpath $tres_save_path --droplr 1 --epochs 5 --gpunum $backbone_cuda --datapath '/home/sharfikeg/my_files/retIQA/kadid10k' --dataset 'kadid10k' --seed $i --vesion 1
    # python3 main.py $exp2
done



# my pretrain, finetune
# no reference
# pure retrieval
# tid2013 exps
# for i in {1..15}
# do
#     exp1="--model finetune_botnet50 --finetune 1 --retrieve 1 --ret_tr resize --num_iters 1 --img_width 288 --img_height 384 --num_classes $tid2013_num_classes --pretrain_classes $my_botnet_pretrain_classes --num_heads 16 --dataset tid2013 --data_path $tid2013_data_path --ref_path $tid2013_ref_path --batch_size 128 --batch_size2 96 --num_workers 12 --lr 0.005 --seed $i --csv_path $tid2013_csv_path --botnet_pretrain $my_botnet_pretrain --baseline_pretrain $tid2013_tres_path --device_num $cuda --backbone_device_num $backbone_cuda --logging_path $logging_path --k 9 --aggregation averaging --epochs 30 --baseline no --setup no_reference --patches 1"
#     python3 main.py $exp1
# done

# # csiq exps
# for i in {1..15}
# do
#     exp3="--model finetune_botnet50 --finetune 1 --retrieve 1 --ret_tr resize --num_iters 1 --img_width 288 --img_height 384 --num_classes $csiq_num_classes --pretrain_classes $my_botnet_pretrain_classes --num_heads 16 --dataset csiq --data_path $csiq_data_path --ref_path $csiq_ref_path --batch_size 128 --batch_size2 96 --num_workers 12 --lr 0.005 --seed $i --csv_path $csiq_csv_path --botnet_pretrain $my_botnet_pretrain --baseline_pretrain $csiq_tres_path --device_num $cuda --backbone_device_num $backbone_cuda --logging_path $logging_path --k 9 --aggregation averaging --epochs 30 --baseline no --setup no_reference --patches 1"
#     python3 main.py $exp3
# done

# # kadid10k exps
# for i in {1..15}
# do
#     exp2="--model finetune_botnet50 --finetune 1 --retrieve 1 --ret_tr resize --num_iters 1 --img_width 288 --img_height 384 --num_classes $kadid10k_num_classes --pretrain_classes $my_botnet_pretrain_classes --num_heads 16 --dataset kadid10k --data_path $kadid10k_data_path --ref_path $kadid10k_ref_path --batch_size 128 --batch_size2 96 --num_workers 12 --lr 0.005 --seed $i --csv_path $kadid10k_csv_path --botnet_pretrain $my_botnet_pretrain --baseline_pretrain $kadid10k_tres_path --device_num $cuda --backbone_device_num $backbone_cuda --logging_path $logging_path --k 9 --aggregation averaging --epochs 30 --baseline no --setup no_reference --patches 1"
#     python3 main.py $exp2
# done




# my pretrain, finetune
# no reference setup
# TReS + retrieval
# tid2013 exps
# for i in {10..15}
# do
#     exp1="--model finetune_botnet50 --finetune 1 --retrieve 1 --ret_tr resize --num_iters 1 --img_width 288 --img_height 384 --num_classes $tid2013_num_classes --pretrain_classes $my_botnet_pretrain_classes --num_heads 16 --dataset tid2013 --data_path $tid2013_data_path --ref_path $tid2013_ref_path --batch_size 128 --batch_size2 $tres_batchsize --num_workers 12 --lr 0.005 --seed $i --csv_path $tid2013_csv_path --botnet_pretrain $my_botnet_pretrain --baseline_pretrain $tid2013_tres_path --device_num $cuda --backbone_device_num $backbone_cuda --logging_path $logging_path --k 9 --aggregation averaging --epochs 30 --baseline tres --setup no_reference --patches $tres_patches"
#     python3 main.py $exp1
# done

# csiq exps
# for i in {13..15}
# do
#     exp3="--model finetune_botnet50 --finetune 1 --retrieve 1 --ret_tr resize --num_iters 1 --img_width 288 --img_height 384 --num_classes $csiq_num_classes --pretrain_classes $my_botnet_pretrain_classes --num_heads 16 --dataset csiq --data_path $csiq_data_path --ref_path $csiq_ref_path --batch_size 128 --batch_size2 $tres_batchsize --num_workers 12 --lr 0.005 --seed $i --csv_path $csiq_csv_path --botnet_pretrain $my_botnet_pretrain --baseline_pretrain $csiq_tres_path --device_num $cuda --backbone_device_num $backbone_cuda --logging_path $logging_path --k 9 --aggregation averaging --epochs 30 --baseline tres --setup no_reference --patches $tres_patches"
#     python3 main.py $exp3
# done


# # kadid10k exps
# for i in {1..15}
# do
#     exp2="--model finetune_botnet50 --finetune 1 --retrieve 1 --ret_tr resize --num_iters 1 --img_width 288 --img_height 384 --num_classes $kadid10k_num_classes --pretrain_classes $my_botnet_pretrain_classes --num_heads 16 --dataset kadid10k --data_path $kadid10k_data_path --ref_path $kadid10k_ref_path --batch_size 128 --batch_size2 $tres_batchsize --num_workers 12 --lr 0.005 --seed $i --csv_path $kadid10k_csv_path --botnet_pretrain $my_botnet_pretrain --baseline_pretrain $kadid10k_tres_path --device_num $cuda --backbone_device_num $backbone_cuda --logging_path $logging_path --k 9 --aggregation averaging --epochs 30 --baseline tres --setup no_reference --patches $tres_patches"
#     python3 main.py $exp2
# done