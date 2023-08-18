import argparse


def load_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default="botnet50")
    parser.add_argument('--img_width', type=int, default=32)
    parser.add_argument('--img_height', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_heads', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--ref_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--batch_size2', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--finetune', type=int, default=1)
    parser.add_argument('--retrieve', type=int, default=1)
    parser.add_argument('--num_iters', type=int, default=1)
    parser.add_argument('--ret_tr', type=str, default="crop")
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--print_intervals', type=int, default=100)
    parser.add_argument('--evaluation', type=bool, default=False)
    parser.add_argument('--checkpoints', type=str, default=None, help='model checkpoints path')
    parser.add_argument('--csv_path', type=str, default=None, help='dataset csv info')
    parser.add_argument('--botnet_pretrain', type=str, default=None, help='botnet pretrain path')
    parser.add_argument('--device_num', type=int, default=1)
    parser.add_argument('--gradient_clip', type=float, default=2.)
    parser.add_argument('--logging_path', type=str, default=None, help='path to where store the results')
    parser.add_argument('--k', type=int, default=9, help= "Parameter for nearest friends")
    parser.add_argument('--aggregation', type=str, default=None, help='type of aggregationn of retrieved values')

    return parser.parse_args()
