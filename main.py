import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
from config import load_config
from preprocess import load_data, load_data2
from model import ResNet50
from BoTNet import botnet
from RetIQANet import RetIQANet
import torchvision.models as models
from scipy import stats
import time
import pandas as pd

torch.set_num_threads(36)
torch.multiprocessing.set_sharing_strategy('file_system')


def save_checkpoint(best_acc, model, optimizer, args, epoch):
    print('Best Model Saving...')
    # if args.device_num > 1:
    #     model_state_dict = model.module.state_dict()
    # else:
    model_state_dict = model.state_dict()

    torch.save({
        'state_dict': model_state_dict,
        'global_epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }, os.path.join(f'../{args.model}_{args.dataset}_checkpoints', f'checkpoint_model_best_heads{args.num_heads}.pth'))


def _train(epoch, train_loader, model, optimizer, criterion, args):
    model.train()

    losses = 0.
    acc = 0.
    total = 0.
    for idx, (data, _ , target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.to(f"cuda:{args.device_num}"), target['label'].to(f"cuda:{args.device_num}")

        output = model(data)
        _, pred = F.softmax(output, dim=-1).max(1)
        acc += pred.eq(target).sum().item()
        total += target.size(0)

        optimizer.zero_grad()
        loss = criterion(output, target)
        losses += loss
        loss.backward()
        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()

        if idx % args.print_intervals == 0 and idx != 0:
            print('[Epoch: {0:4d}], Loss: {1:.3f}, Acc: {2:.3f}, Correct {3} / Total {4}'.format(epoch,
                                                                                                 losses / (idx + 1),
                                                                                                 acc / total * 100.,
                                                                                                 acc, total))


def _eval(epoch, test_loader, model, args):
    model.eval()

    acc = 0.
    with torch.no_grad():
        for data, _ , target in test_loader:
            if args.cuda:
                data, target = data.to(f"cuda:{args.device_num}"), target['label'].to(f"cuda:{args.device_num}")
            output = model(data)
            _, pred = F.softmax(output, dim=-1).max(1)

            acc += pred.eq(target).sum().item()
        print('[Epoch: {0:4d}], Acc: {1:.3f}'.format(epoch, acc / len(test_loader.dataset) * 100.))

    return acc / len(test_loader.dataset) * 100.


def main(args):

    print(f"Seed: {args.seed}")
    # 1
    if args.finetune:
        train_loader, test_loader = load_data(args)
        print("Data is loaded!")

        if args.model == "finetune_botnet50":
            model = botnet(
                args.botnet_pretrain,
                resolution=(288, 384), heads=args.num_heads, num_classes=150
            )
            model.fc[1] = nn.Linear(in_features=8192, out_features=args.num_classes, bias=True)

        elif args.model == "botnet50":
            model = ResNet50(num_classes=args.num_classes, resolution=(args.img_height, args.img_width), heads=args.num_heads)

        elif args.model == "vgg16_bn":
            model = models.vgg16_bn(pretrained=True)
            model.classifier[-1] = nn.Linear(in_features=4096, out_features=args.num_classes, bias=True)

        elif args.model == "resnet50":
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(in_features=2048, out_features=args.num_classes, bias=True)

        elif args.model == "vit_h_14":
            model = models.vit_h_14(weights="DEFAULT")
            model.heads.head = nn.Linear(in_features=1280, out_features=args.num_classes, bias=True)

        else:
            print("Model type is not implemented!")
            os._exit(0)

        print("Model is loaded!")
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

        if not os.path.isdir(f'../{args.model}_{args.dataset}_checkpoints'):
            os.mkdir(f'../{args.model}_{args.dataset}_checkpoints')

        if args.checkpoints is not None:
            checkpoints = torch.load(os.path.join(f'../{args.model}_checkpoints', args.checkpoints))
            model.load_state_dict(checkpoints['model_state_dict'])
            optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
            start_epoch = checkpoints['global_epoch']
        else:
            start_epoch = 1

        if args.cuda:
            model = model.to(f"cuda:{args.device_num}")

        if not args.evaluation:
            criterion = nn.CrossEntropyLoss()

            if args.model == "finetune_botnet50":
                lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
            else:
                lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 12, T_mult=2, eta_min=0.0001)

            global_acc = 0.
            print("Training started!")
            for epoch in range(start_epoch, args.epochs + 1):
                _train(epoch, train_loader, model, optimizer, criterion, args)
                best_acc = _eval(epoch, test_loader, model, args)
                if global_acc < best_acc:
                    global_acc = best_acc
                    save_checkpoint(best_acc, model, optimizer, args, epoch)

                lr_scheduler.step()
                print('Current Learning Rate: {}'.format(lr_scheduler.get_last_lr()))
        else:
            _eval(start_epoch, test_loader, model, args)

        print("Best acc: ", global_acc)
        finetuned_model_path = os.path.join(f'../{args.model}_{args.dataset}_checkpoints', f'checkpoint_model_best_heads{args.num_heads}.pth')
    else:
        finetuned_model_path = args.botnet_pretrain
        global_acc = torch.load(finetuned_model_path)['best_acc']
        print("Best acc: ", global_acc)

    # 2
    if args.retrieve:
        train_loader, test_loader = load_data2(args)

        model = RetIQANet(
            dpm_checkpoints=finetuned_model_path,
            train_dataset=train_loader,
            cuda=args.device_num,
            K=args.k
        )

        r_s = []
        gr_trs = []
        if args.cuda:
            device = f"cuda:{args.device_num}"
        else:
            device = "cpu"

        for ycbcr, rgb, y in test_loader:
            res = model(ycbcr.to(device), rgb.to(device))
            gr_trs.append(y['metric'])
            r_s.append(res)
        t_r_s = []
        for r in r_s:
            t_r_s.append(torch.tensor(r))
        srocc = stats.spearmanr(torch.concat(t_r_s), torch.concat(gr_trs))[0]
        plcc = stats.pearsonr(torch.concat(t_r_s), torch.concat(gr_trs))[0]

        df = pd.read_csv(args.logging_path)
        results = [
            time.time(),
            args.dataset,
            global_acc,
            args.aggregation,
            args.k,
            args.batch_size2,
            args.seed,
            srocc,
            plcc
        ]
        df.loc[len(df)] = results
        df.to_csv(args.logging_path, index=False)



if __name__ == '__main__':
    args = load_config()
    main(args)