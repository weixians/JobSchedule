import os
import shutil

from model.cnn_model import Net2
from util.global_util import get_project_root


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
        shutil.copytree(os.path.join(get_project_root(), "data"), os.path.join(args.output, "data"))

    args.data_dir = os.path.join(args.output, "data")


def build_network(args, action_size):
    return Net2(action_size, args.dueling)
