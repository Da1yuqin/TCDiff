import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Trajectory Model Options',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ## Data
    parser.add_argument("--data_path", type=str, 
                        default="./data/AIOZ_Dataset/", 
                        help="Path to raw data directory containing train/test folders.") 
    parser.add_argument("--processed_data_dir",type=str, default="./data/dataset_backups/", help="Path to save/load preprocessed dataset backups.") 
    parser.add_argument( 
        "--force_reload", default = False, action="store_true", help="Force reprocessing of the dataset instead of loading cached versions."
    )
    parser.add_argument("--no_cache", action="store_true", default=False, help="Disable dataset caching and always load from scratch.")

    ## dancer_num ###
    parser.add_argument( 
        "--required_dancer_num", type = int, default=4, help="Number of dancers required in each sample."
    )

    ## Checkpoint paths
    parser.add_argument("--checkpoint", type=str,  
    # default = "./log/exp_debug/ckpt/epoch-79000.pth",\ 
    default = None,
    ) # Resume path
    parser.add_argument("--ckpt_dir", type=str, default="./log/exp_debug/ckpt/", help="Directory to save model checkpoints.") # Saving path


    ## model
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nfeats", type=int, default=2, help="Dimensionality of input features (e.g. x, y).")
    parser.add_argument("--trans_layer", type=int, default=6, help="Number of transformer layers.")
    parser.add_argument("--window_size", type=int, default=100, help="Sliding window size for training and prediction.") 
    parser.add_argument("--step", type=int, default=25, help="Sliding window step size.")

    ## Out Result
    parser.add_argument(
        "--render_dir", type=str, 
        default="./log/exp_debug/render/", 
        help="Directory to save rendered output videos or images."
    )

    parser.add_argument(
        "--fkout_dir", type=str, 
        # default="./log/exp_debug/fk_out/", 
        default = None,
         help="Optional path for saving forward kinematics output (FK data)."
    )
    
    ## optimization
    parser.add_argument('--total-iter', default=800000, type=int, help='number of total iterations to run')
    parser.add_argument('--print-iter', default=5000, type=int, help='Print logs every N iterations.')
    parser.add_argument("--batch_size", type=int, default=128, help="batch size") 
    parser.add_argument('--lr', default=0.002, type=float, help='max learning rate')
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    parser.add_argument('--lr-scheduler', default=[60000], nargs="+", type=int, help="learning rate schedule (iterations)")
    
    parser.add_argument('--weight-decay', default=1e-6, type=float, help='weight decay') 
    parser.add_argument('--decay-option',default='all', type=str, choices=['all', 'noVQ'], help='disable weight decay on codebook')
    parser.add_argument('--optimizer',default='adamw', type=str, choices=['adam', 'adamw'], help='Optimizer type.')
    parser.add_argument('--seed', default=42, type=int, help='seed for initializing training. ')
    
    ## output 
    parser.add_argument('--out-dir', type=str, default='log/', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out-dir')

    
    return parser.parse_args()

