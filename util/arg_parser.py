import argparse


def parse_args():
    parser = argparse.ArgumentParser("Parse configuration")
    parser.add_argument("--output", type=str, default="output_jobshop", help="root path of output dir")
    parser.add_argument("--seed", type=int, default=42, help="seed of the random")
    parser.add_argument("--test", default=False, action="store_true", help="whether in test mode")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu id to use. If less than 0, use cpu instead.")
    parser.add_argument("--model_dir", type=str, default="model", help="folder path to save/load neural network models")
    parser.add_argument("--epi", type=int, default=None, help="")
    parser.add_argument(
        "--dueling", default=False, action="store_true", help="whether to use dueling ddqn (use ddqn if false)"
    )
    # for render start
    parser.add_argument("--render", default=False, action="store_true", help="whether in the gui mode")
    parser.add_argument("--mode", default="img", type=str, help="render mode (img, video)")
    parser.add_argument("--ffmpeg", default=None, type=str, help="path of ffmpeg, needed for video mode")
    # for render end

    parser.add_argument("--n_j", type=int, default=10, help="Number of jobs of instance")
    parser.add_argument("--n_m", type=int, default=10, help="Number of machines instance")
    parser.add_argument("--low", type=int, default=1, help="LB of duration")
    parser.add_argument("--high", type=int, default=99, help="UB of duration")
    parser.add_argument("--np_seed_validation", type=int, default=200, help="Seed for numpy for validation")

    return parser.parse_args()
