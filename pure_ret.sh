cuda=3
backbone_device="cuda:5"
ret_device="cuda:5"
botnet_pretrain="/home/sharfikeg/my_files/VIPNet/pretrained_model/botnet_model_best.pth.tar"
botnet_pretrain_classes=150
my_botnet_pretrain="/home/sharfikeg/my_files/retIQA/dc_ret/my_botnet_pretrain/checkpoint_model_best_heads16.pth"
my_botnet_pretrain_classes=125
logging_path="/home/sharfikeg/my_files/retIQA/dc_ret/DistorsionFeatureExtractor/results.csv"
tres_save_path="/extra_disk_1/sharfikeg/Save_TReS/"
tres_launch_training="/home/sharfikeg/my_files/retIQA/dc_ret/DistorsionFeatureExtractor/TReS/run.py"
tres_batchsize=5
tres_patches=50
# k=50

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

liveitw_test_data_path="/home/s-kastryulin/data/LIVE-itW/Images/"
liveitw_train_data_path="/home/s-kastryulin/data/koniq10k/1024x768/"
liveitw_train_csv_path="/home/sharfikeg/my_files/retIQA/liveitw_info.csv"
liveitw_test_csv_path="/home/sharfikeg/my_files/retIQA/liveitw_test_info.csv"

# my pretrain, no finetune
# no reference setup
# ret

for k in {100..300..10}
do
    # # kadid10k exps
    # for i in {1..3}
    # do
    #     exp2="--model finetune_botnet50 --finetune 0 --retrieve 1 --ret_tr resize --num_iters 1 --img_width 288 --img_height 384 --num_classes $kadid10k_num_classes --pretrain_classes $my_botnet_pretrain_classes --num_heads 16 --dataset kadid10k --data_path $kadid10k_data_path --ref_path $kadid10k_ref_path --batch_size 64 --batch_size2 4 --num_workers 12 --lr 0.005 --seed $i --csv_path $kadid10k_csv_path --botnet_pretrain $my_botnet_pretrain --baseline_pretrain ${tres_save_path}kadid10k_1_${i}/sv/bestmodel_1_${i} --device_num $cuda --backbone_device $backbone_device --ret_device $ret_device --logging_path $logging_path --k $k --aggregation averaging --epochs 30 --baseline no --setup no_reference --patches 1"
    #     python3 main.py $exp2
    # done

    # # csiq exps
    # for i in {1..3}
    # do
    #     exp3="--model finetune_botnet50 --finetune 0 --retrieve 1 --ret_tr resize --num_iters 1 --img_width 288 --img_height 384 --num_classes $csiq_num_classes --pretrain_classes $my_botnet_pretrain_classes --num_heads 16 --dataset csiq --data_path $csiq_data_path --ref_path $csiq_ref_path --batch_size 128 --batch_size2 8 --num_workers 12 --lr 0.005 --seed $i --csv_path $csiq_csv_path --botnet_pretrain $my_botnet_pretrain --baseline_pretrain ${tres_save_path}csiq_1_${i}/sv/bestmodel_1_${i} --device_num $cuda --backbone_device $backbone_device --ret_device $ret_device --logging_path $logging_path --k $k --aggregation averaging --epochs 30 --baseline no --setup no_reference --patches 1"
    #     python3 main.py $exp3
    # done

    # # tid2013 exps
    # for i in {1..3}
    # do
    #     exp1="--model finetune_botnet50 --finetune 0 --retrieve 1 --ret_tr resize --num_iters 1 --img_width 288 --img_height 384 --num_classes $tid2013_num_classes --pretrain_classes $my_botnet_pretrain_classes --num_heads 16 --dataset tid2013 --data_path $tid2013_data_path --ref_path $tid2013_ref_path --batch_size 128 --batch_size2 8 --num_workers 12 --lr 0.005 --seed $i --csv_path $tid2013_csv_path --botnet_pretrain $my_botnet_pretrain --baseline_pretrain ${tres_save_path}tid2013_1_${i}/sv/bestmodel_1_${i} --device_num $cuda --backbone_device $backbone_device --ret_device $ret_device --logging_path $logging_path --k $k --aggregation averaging --epochs 30 --baseline no --setup no_reference --patches 1"
    #     python3 main.py $exp1
    # done



    # koniq10k exps
    for i in {1..5}
    do
        exp1="--model finetune_botnet50 --finetune 0 --retrieve 1 --ret_tr resize --num_iters 1 --img_width 288 --img_height 384 --num_classes $my_botnet_pretrain_classes --pretrain_classes $my_botnet_pretrain_classes --num_heads 16 --dataset koniq --data_path $koniq10k_data_path --batch_size 64 --batch_size2 32 --num_workers 12 --lr 0.005 --seed $i --csv_path $koniq10k_csv_path --botnet_pretrain $my_botnet_pretrain --baseline_pretrain ${tres_save_path}koniq_1_${i}/sv/bestmodel_1_${i} --device_num $cuda --backbone_device $backbone_device --ret_device $ret_device --logging_path $logging_path --k $k --aggregation averaging --epochs 30 --baseline no --setup no_reference --patches 1"
        python3 main.py $exp1
    done

    # spaq exps
    for i in {1..5}
    do
        exp1="--model finetune_botnet50 --finetune 0 --retrieve 1 --ret_tr resize --num_iters 1 --img_width 288 --img_height 384 --num_classes $my_botnet_pretrain_classes --pretrain_classes $my_botnet_pretrain_classes --num_heads 16 --dataset spaq --data_path $spaq_data_path --batch_size 64 --batch_size2 32 --num_workers 12 --lr 0.005 --seed $i --csv_path $spaq_csv_path --botnet_pretrain $my_botnet_pretrain --baseline_pretrain ${tres_save_path}spaq_1_${i}/sv/bestmodel_1_${i} --device_num $cuda --backbone_device $backbone_device --ret_device $ret_device --logging_path $logging_path --k $k --aggregation averaging --epochs 30 --baseline no --setup no_reference --patches 1"
        python3 main.py $exp1
    done
done
# # live-itw exps
# for i in {1..1}
# do
#     exp1="--model finetune_botnet50 --finetune 0 --retrieve 1 --ret_tr resize --num_iters 1 --img_width 288 --img_height 384 --num_classes $my_botnet_pretrain_classes --pretrain_classes $my_botnet_pretrain_classes --num_heads 16 --dataset liveitw --data_path $liveitw_train_data_path --batch_size 96 --batch_size2 8 --num_workers 12 --lr 0.005 --seed $i --csv_path $liveitw_train_csv_path --botnet_pretrain $my_botnet_pretrain --baseline_pretrain $liveitw_tres_path --device_num $cuda --backbone_device $backbone_device --ret_device $ret_device --logging_path $logging_path --k 9 --aggregation averaging --epochs 30 --baseline no --setup no_reference --patches 1 --test_data_path $liveitw_test_data_path --test_csv_path $liveitw_test_csv_path"
#     python3 main.py $exp1
# done




