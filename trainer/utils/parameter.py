import argparse
import deepspeed
from easydict import EasyDict
from utils import read_from_yaml

def parse_args():
    parser = argparse.ArgumentParser(description="DeepLearning framework based on PyTorch")

    # ----------------------distributed parameter-----------------------
    parser.add_argument("--backend", type=str, default="nccl", help="use for current backend for distributed")
    parser.add_argument("--init_method", type=str, default="tcp://127.0.0.1:56947", help="init method for distributed")
    parser.add_argument("--rank", type=int, default=0, help="current rank for distributed")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank, range 0-7")
    parser.add_argument("--world_size", type=int, default=1, help="current process number for distributed")

    # -----------------------common parameter-----------------------
    parser.add_argument("--yml_path", type=str, default="", help="cfg name")
    parser.add_argument("--pretrained_path", type=str, help="pretrained model path")
    parser.add_argument("--load_optimizer", type=int, default=0, help="0: don't load; 1:load optimizer")
    parser.add_argument(
        "--caption_shuffle_percent",
        type=float,
        default=0,
        help="shuffle the caption with a certain probability, from 0 to 1, don't shuffle if set to 0",
    )
    parser.add_argument("--train_data_index", type=int, default=0, help="train_data_index")
    parser.add_argument(
        "--local_shuffle_type",
        type=int,
        help="0: not use local shuffle "
        "1: use local shuffle by node "
        "2: use local shuffle by card "
        "4: use local shuffle by card in zip format, recommend",
    )
    parser.add_argument("--zip_max_split", type=int, default=1024, help="used when local_shuffle_type=4")
    parser.add_argument(
        "--visual_memory_format", type=str, default="contiguous_format", help="channels_last or " "contiguous_format"
    )
    parser.add_argument("--show_model_arch", type=int, default=0, help="show model arch and params on log")
    parser.add_argument("--output_path", type=str, default="./exp", help="output path for saving log and pth")
    parser.add_argument("--log_interval", type=int, default=100, help="steps to show log info")
    parser.add_argument("--save_per_epochs", type=float, default=0.5, help="epochs to save pth, can be less than 1")
    parser.add_argument("--max_epochs", type=int, default=3, help="training epochs")
    parser.add_argument("--warmup_epochs", type=float, default=0.5, help="warmup epochs, can be less than 1")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="lr scheduler")
    parser.add_argument("--DATALOADER", type=str, default="CLIP_zip_dataloader")

    parser.add_argument("--mode", type=str, default="trian", help="training sign, do not change ")
    parser.add_argument("--resume", type=int, default=0, help="resume model")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="prefetch in dataloader, for faster training")
    parser.add_argument("--lr", type=float, default=0.0008, help="learning rate")
    # parser.add_argument('--optimizer', type=str, default='fused_adamw', help='optimizer')
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--label_smooth", type=float, default=0.1, help="label smooth")
    parser.add_argument("--image_size", type=int, default=224, help="reshape the size of image")
    parser.add_argument("--num_workers", type=int, default=6, help="process number of data loader")
    parser.add_argument("--tokenizer_type", type=str, default="bert_chinese", help="tokenizer type in data loader")

    # ----------------------debug related----------------------
    parser.add_argument("--fix_inputs", type=int, default=0, help="ignore dataloader when training, for profiling")
    parser.add_argument("--profile", type=int, default=0, help="pytorch profile")
    parser.add_argument("--profile_skip_first", type=int, default=5, help="pytorch profile")
    parser.add_argument("--profile_wait", type=int, default=5, help="pytorch profile")
    parser.add_argument("--profile_warmup", type=int, default=2, help="pytorch profile")
    parser.add_argument("--profile_active", type=int, default=3, help="pytorch profile")
    parser.add_argument("--profile_repeat", type=int, default=5, help="pytorch profile")
    parser.add_argument("--profile_step", type=int, default=150, help="npu profile")

    # ----------------------optmize----------------------
    parser.add_argument("--CRITERION", type=str, default="clip_loss_gather_parallel", help="loss function")
    parser.add_argument("--beta1", type=float, default=0.9, help="adam beta1")
    parser.add_argument("--beta2", type=float, default=0.96, help="adam beta2")

    # ----------------------model----------------------
    parser.add_argument("--embed_dim", type=int, default=512, help="dimension of output")
    parser.add_argument("--context_length", type=int, default=80, help="length of token sent to model")

    # ----------------------eval----------------------
    parser.add_argument("--eval_first", type=int, default=0, help="whether eval at 1st step")
    parser.add_argument("--eval_yml_path", type=str, default="", help="cfg name for eval")
    # hwzhquery eval
    parser.add_argument("--do_multilabeling", type=int, default=0, help="validation related @xkx")
    parser.add_argument(
        "--min_recall", type=float, default=0.8, help="hwzhquery eval for thr reliable, eg. recall=80%, test acc"
    )
    parser.add_argument("--exclude", type=list, default=[], help="hwzhquery exclude some part when eval")
    parser.add_argument("--test_plan", type=list, default=[], help="settings for hwzhquery")
    parser.add_argument("--thres", type=int, default=0, help="have correlation to thr reliable")

    # ---------------------ema----------------------------------
    parser.add_argument("--ema", type=int, default=0, help="whether use ema")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="ema_decay")
    parser.add_argument(
        "--ema_multi_tensor_apply_chunk_size", type=int, default=10000, help="ema_multi_tensor_apply_chunk_size"
    )

    parser.add_argument("--user_id", type=str, default="", help="user account")
    parser = deepspeed.add_config_arguments(parser)
    args, unknown = parser.parse_known_args()

    return EasyDict(vars(args))

def parse_args_from_yaml(yml_path):
    config = read_from_yaml(yml_path)
    config_obj = EasyDict(config)
    return config_obj
