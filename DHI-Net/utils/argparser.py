import level_inter as li

def get_argparser():
    parser = li.arg.ArgumentParser()

    # Dataset Options
    parser.add_argument("--data_root", type=str, default='./../../building_extraction/data',
                        help="path to Dataset")
    # parser.add_argument("--dataset", type=str, default='voc',
    #                     choices=['whu_w', 'mass','inria'], help='Name of dataset')

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--total_itrs", type=int, default=150,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--weight_decay", type=float, default=1e-3,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=2,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=512)

    parser.add_argument("--model_path", default='./result/wh/ml18.mdl', type=str,
                        help="restore from checkpoint")
    return parser