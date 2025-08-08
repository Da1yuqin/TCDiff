import argparse


def parse_train_opt():
    parser = argparse.ArgumentParser()

    ### Project ###
    parser.add_argument("--project", default="./runs/train")
    parser.add_argument("--exp_name", default="exp")
    
    ### dataset ###
    parser.add_argument("--data_path", type=str, 
            default="./data/AIOZ_Dataset/", 
            help="Path to raw dataset folder (must contain train/test subfolders). Do not modify unless necessary.")
    parser.add_argument("--processed_data_dir",type=str, 
                        default="./data/dataset_backups/", help="Dataset backup path") 
    parser.add_argument("--batch_size", type=int, default=37) 
    parser.add_argument("--window_size", type=int, default=150, help="window size")
    parser.add_argument( 
        "--force_reload", default = False, action="store_true", help="Force reprocessing of the dataset, ignoring cached data."
    )

    parser.add_argument("--no_cache", action="store_true", default = False, help="Disable dataset caching; always reload from disk.")
        
    parser.add_argument(
        "--required_dancer_num", type = int, default=4, help="Number of dancers required in each sample."
    )


    ### Out Result ###
    parser.add_argument(
        "--vis_fk_out", type=str, default="./fk_out4Vis", help="Path to save FK outputs for visualization."
    )
    parser.add_argument(
        "--render_dir", type=str, default="./renders/", help="Path to save rendered sample videos."
    )
    parser.add_argument(
        "--wandb_pj_name", type=str, default="TCDiff", help="Project name for Weights & Biases tracking."
    )

    ### Training ###
    parser.add_argument( "--learning-rate", type=float, default = 0.00005, help="learning rate")
    parser.add_argument("--epochs", type=int, default=10000)

    parser.add_argument(
        "--save_interval",
        type=int,
        default=50,
        help='Log model after every "save_period" epoch',
    )
    parser.add_argument("--ema_interval", type=int, default=1, help="ema every x steps")
    parser.add_argument(
        "--checkpoint", type=str, 
        default = "", 
        help="trained checkpoint path (optional)"
    )

    ## Validation ##
    parser.add_argument(
        "--traj_checkpoint", type=str, 
        default = None,
        # default = "./TrajDecoder/log/exp_debug/ckpt/epoch-79000.pth",
        help="trained trajectory path (optional, only used when mode is 'test')"
    )
    parser.add_argument("--mode", default = "train", choices=["train", "val_without_TrajModel", "test"])
    
    opt = parser.parse_args()
    return opt