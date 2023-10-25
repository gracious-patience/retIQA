import os
import argparse
import random
import json
import numpy as np
import torch
from args import Configs
import logging
import data_loader
from sklearn.model_selection import train_test_split

from models import TReS, Net


print('torch version: {}'.format(torch.__version__))


def main(config,device): 
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
    
    folder_path = {
        'live':         config.datapath,
        'csiq':         config.datapath,
        'sly_csiq':     config.datapath,
        'tid2013':      config.datapath,
        'sly_tid2013':  config.datapath,
        'kadid10k':     config.datapath,
        'sly_kadid10k': config.datapath,
        'clive':        config.datapath,
        'koniq':        config.datapath,
        'cross_koniq':  config.datapath,
        'partial_koniq':config.datapath,
        'fblive':       config.datapath,
        'spaq':         config.datapath,
        'cross_spaq':   config.datapath,
        'biq':          config.datapath,
        'pipal':        config.datapath,
        'sly_pipal':    config.datapath
        }

    img_num = {
        'live':         list(range(0, 29)),
        'csiq':         list(range(0, 30)),
        'sly_csiq':     list(range(0, 30)),
        'kadid10k':     list(range(0, 80)),
        'sly_kadid10k': list(range(0, 80)),
        'tid2013':      list(range(0, 25)),
        'sly_tid2013':  list(range(0, 25)),
        'clive':        list(range(0, 1169)),
        'koniq':        list(range(0, 10073)),
        'cross_koniq':  list(range(0, 10073)),
        'partial_koniq':list(range(0, 10073)),
        'fblive':       list(range(0, 39810)),
        'spaq':         list(range(0, 11125)),
        'cross_spaq':   list(range(0, 11125)),
        'biq':          list(range(0, 11989)),
        'pipal':        list(range(0, 200)),
        'sly_pipal':    list(range(0, 200))
        }
    

    print('Testing on {} dataset...'.format(config.dataset))
    
    SavePath = config.svpath
    if config.finetune:
        svPath = SavePath+ config.dataset + '_' + str(config.vesion)+'_'+str(config.seed)+'/k_'+str(config.k)+ f'/lr_{config.lr}_lrratio{config.lrratio}' + '/'+'finetune'
        if config.full_finetune:
            svPath += '/full_finetune'
    else:
        svPath = SavePath+ config.dataset + '_' + str(config.vesion)+'_'+str(config.seed)+ '/k_'+str(config.k)+  f'/lr_{config.lr}_lrratio{config.lrratio}' +'/'+'no_finetune'
    if config.resume:
        svPath += '/resume'
    if config.multi_return:
        svPath += '/multi_return'
        if config.multi_ranking:
            svPath += '/multi_ranking'
        else:
            svPath += '/single_ranking'
    else:
        svPath += '/single_return'
    os.makedirs(svPath, exist_ok=True)
        
    
    
     # fix the seed if needed for reproducibility
    if config.seed == 0:
        pass
    else:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)


    
    pretrained_path = config.svpath + config.dataset + '_' + str(config.vesion)+'_'+str(config.seed)+'/'+'sv/'
    print('path: {}'.format(svPath))
    path = svPath + '/test_index_'+str(config.vesion)+'_'+str(config.seed)+'.json'
    path2 = svPath + '/train_index_'+str(config.vesion)+'_'+str(config.seed)+'.json'

    with open(path) as json_file:
	    test_index = json.load(json_file)
    with open(path2) as json_file:
	    train_index =json.load(json_file)
    
    total_num_images = img_num[config.dataset]
    train_index, test_index = train_test_split(total_num_images, test_size=0.2, random_state=config.seed)
    train_index, _ = train_test_split(train_index, test_size=config.retrieve_size, random_state=config.seed)
    val_index, test_index = train_test_split(test_index, test_size=0.5, random_state=config.seed)

   
    test_loader = data_loader.DataLoader(config.dataset, folder_path[config.dataset],
                                             test_index, config.patch_size,
                                             config.test_patch_num, batch_size=config.batch_size, k=config.k, seed=config.seed, istrain=False,
											 cross_root=config.cross_datapath, cross_dataset=config.cross_dataset,
											 delimeter=config.delimeter, retrieve_size=config.retrieve_size)
    test_data = test_loader.get_data()


    solver = TReS(config,device, svPath, folder_path[config.dataset], train_index, test_index,Net)
    version_test_save = 1000
    srcc_computed, plcc_computed = solver.test(test_data,version_test_save,svPath,config.seed,pretrained=1)
    print('srcc_computed {}, plcc_computed {}'.format(srcc_computed, plcc_computed))

    # logging the performance
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    handler = logging.FileHandler(svPath + '/LogPerformance.log')

    formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    Dataset = config.dataset
    logger.info(Dataset)
    
    PrintToLogg = 'Test PLCC: {}, SROCC: {}'.format(plcc_computed,srcc_computed)
    logger.info(PrintToLogg)
    logger.info('---------------------------')



if __name__ == '__main__':
    
    config = Configs()
    print(config)

    if torch.cuda.is_available():
            if len(config.gpunum)==1:
                device = torch.device("cuda", index=int(config.gpunum))
            else:
                device = torch.device("cpu")
        
    main(config,device)
    