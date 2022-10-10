import os
import numpy as np

from util.global_util import get_project_root


def gen_instance_uniformly(n_j, n_m, low, high):
    # 每个task的处理时长
    durations = np.random.randint(low=low, high=high, size=(n_j, n_m))
    # 机器编号，从0开始
    machines = np.expand_dims(np.arange(0, n_m), axis=0).repeat(repeats=n_j, axis=0)
    machines = _permute_rows(machines)
    return durations, machines


def _permute_rows(x: np.ndarray):
    """
    打乱每个job的machine处理顺序
    Args:
        x (np.ndarray): shape (n_j,n_m)
    """
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def gen_and_save(j, m, l, h, size, seed):
    np.random.seed(seed)
    data = np.array([gen_instance_uniformly(n_j=j, n_m=m, low=l, high=h) for _ in range(size)])
    print(data.shape)
    folder = os.path.join(get_project_root(), "data")
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "generatedData{}_{}_Seed{}.npy".format(j, m, seed)), data)


if __name__ == "__main__":
    j = 10
    m = 10
    l = 1
    h = 99
    batch_size = 100
    seed = 200

    gen_and_save(j, m, l, h, batch_size, seed)

    # durations_list, machines_list = [], []
    # for _ in range(10):
    #     durations, machines = gen_instance_uniformly(15, 15, 1, 99)
    #     durations_list.append(durations)
    #     machines_list.append(machines)
    # folder = os.path.join(global_util.get_project_root(), "data")
    # os.makedirs(folder, exist_ok=True)
    # with open(os.path.join(folder, "instances.pkl"), "wb") as f:
    #     pickle.dump((durations_list, machines_list), f)
