import argparse
import os
import shutil

from util import global_util
from model.cnn_model import Net2
from model.mlp_model import NetMLP

def parse_args():
    parser = argparse.ArgumentParser("Parse configuration")
    parser.add_argument("--output", type=str, default="output_jobshop", help="root path of output dir")
    parser.add_argument("--seed", type=int, default=42, help="seed of the random")
    parser.add_argument("--test", default=False, action="store_true", help="whether in test mode")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu id to use. If less than 0, use cpu instead.")
    parser.add_argument("--model_dir", type=str, default="model", help="folder path to save/load neural network models")
    parser.add_argument("--epi", type=int, default=None, help="")
    parser.add_argument("--net", type=str, default="net2", help="network name")
    parser.add_argument(
        "--dueling", default=False, action="store_true", help="whether to use dueling ddqn (use ddqn if false)"
    )
    # for render start
    parser.add_argument("--render", default=False, action="store_true", help="whether in the gui mode")
    parser.add_argument("--mode", default="img", type=str, help="render mode (img, video)")
    parser.add_argument("--ffmpeg", default=None, type=str, help="path of ffmpeg, needed for video mode")
    # for render end

    return parser.parse_args()


def copy_data_folder_to_output(args, override=False):
    """
    拷贝data目录到输出目录下
    :param args:
    :param override: 默认是否覆盖
    :return:
    """
    make_new_dir = True
    if args.test:
        make_new_dir = False
    elif os.path.exists(args.output):
        if args.test:
            make_new_dir = False
        else:
            if override:
                key = "y"
            else:
                key = input("输出目录已经存在，是否覆盖? (y/n)")
            if key == "y":
                make_new_dir = True
                shutil.rmtree(args.output)
            else:
                make_new_dir = False
    if make_new_dir:
        shutil.copytree(os.path.join(global_util.get_project_root(), "data"), os.path.join(args.output, "data"))

    args.data_dir = os.path.join(args.output, "data")


def build_network(args, action_size):
    if args.net == "net2":
        q_local_net = Net2(action_size, args.dueling)
    else:
        q_local_net = NetMLP(input_dim, action_size)
    return q_local_net
