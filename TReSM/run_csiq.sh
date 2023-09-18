
# this is for resnet50
# OMP_NUM_THREADS=1 python /home/tres/TReS/run.py  --num_encoder_layerst 2 --dim_feedforwardt 64 --nheadt 16 --network 'resnet50' --batch_size 53  --svpath  '/home/sharfikeg/my_files/retIQA/dc_ret/Save_TReS/'   --droplr 1 --epochs 5 --gpunum '0'  --datapath  '/home/tres/qadata/fblive'    --dataset 'fblive'   --seed 2021 --vesion 1
# OMP_NUM_THREADS=1 python /home/tres/TReS/run.py  --num_encoder_layerst 2 --dim_feedforwardt 64 --nheadt 16 --network 'resnet50' --batch_size 53  --svpath  '/home/sharfikeg/my_files/retIQA/dc_ret/Save_TReS/'   --droplr 1 --epochs 3 --gpunum '0'  --datapath  '/home/tres/qadata/csiq'      --dataset 'csiq'     --seed 2021 --vesion 1


# OMP_NUM_THREADS=24 python /home/sharfikeg/my_files/retIQA/dc_ret/DistorsionFeatureExtractor/TReS/run.py  --num_encoder_layerst 2 --dim_feedforwardt 64 --nheadt 16 --network 'resnet50' --batch_size 53  --svpath  '/extra_disk_1/sharfikeg/Save_TReS/'   --droplr 1 --epochs 3 --gpunum '3'  --datapath  '/extra_disk_1/sharfikeg/spaq'     --dataset 'spaq'    --seed 1 --vesion 1
# OMP_NUM_THREADS=24 python /home/sharfikeg/my_files/retIQA/dc_ret/DistorsionFeatureExtractor/TReS/run.py  --num_encoder_layerst 2 --dim_feedforwardt 64 --nheadt 16 --network 'resnet50' --batch_size 53  --svpath  '/extra_disk_1/sharfikeg/Save_TReS/'   --droplr 1 --epochs 3 --gpunum '2'  --datapath  '/home/sharfikeg/my_files/retIQA/koniq10k'     --dataset 'koniq'    --seed 1 --vesion 1
# OMP_NUM_THREADS=24 python /home/sharfikeg/my_files/retIQA/dc_ret/DistorsionFeatureExtractor/TReS/run.py --num_encoder_layerst 2 --dim_feedforwardt 64 --nheadt 16 --network 'resnet50' --batch_size 53  --svpath  '/extra_disk_1/sharfikeg/Save_TReS/'   --droplr 1 --epochs 3 --gpunum '2'  --datapath  '/home/sharfikeg/my_files/retIQA/kadid10k'  --dataset 'kadid10k' --seed 1 --vesion 1
# OMP_NUM_THREADS=24 python /home/sharfikeg/my_files/retIQA/dc_ret/DistorsionFeatureExtractor/TReS/run.py  --num_encoder_layerst 2 --dim_feedforwardt 64 --nheadt 16 --network 'resnet50' --batch_size 53  --svpath  '/extra_disk_1/sharfikeg/Save_TReS/'   --droplr 1 --epochs 3 --gpunum '2'  --datapath  '/home/s-kastryulin/data/tid2013'   --dataset 'tid2013'  --seed 1 --vesion 1
for i in {2..10}
do
    OMP_NUM_THREADS=2 python /home/sharfikeg/my_files/retIQA/ret/TReSM/run.py  --num_encoder_layerst 2 --dim_feedforwardt 64 --nheadt 16 --network 'resnet50' --batch_size 53  --svpath  '/extra_disk_1/sharfikeg/Save_TReSS/'   --droplr 1 --epochs 100 --gpunum '6'  --datapath  '/home/sharfikeg/my_files/retIQA/csiq'  --dataset 'csiq'  --seed $i --vesion 1 --k 3 --multi_return 0 --finetune 0 --single_channel 1 --lr 2e-4 --lrratio 1 --multi_ranking 0 --resume 0
done


