import os.path

import numpy as np

from util import global_util


class Instance:
    # 案例名称
    name: str
    # job数量
    job_size: int
    # 机器数量
    machine_size: int
    # 每个operation(task)对应的的处理时间
    processing_time: np.ndarray
    # 每个operation(task)对应的的机器编号
    machine_nos: np.ndarray

    def __init__(self, name):
        self.name = name


def load_instances(filename):
    instance_dict = {}
    with open(filename) as f:
        lines = f.readlines()

        i = 0
        while i < len(lines):
            if lines[i].strip() == "+++++++++++++++++++++++++++++":
                i += 2
                instance = Instance(lines[i].strip().replace("instance ", ""))
                for _ in range(4):
                    i += 1
                arr = list(map(int, lines[i].strip().split()))
                instance.job_size = arr[0]
                instance.machine_size = arr[1]
                data = []
                for _ in range(instance.job_size):
                    i += 1
                    data.append(list(map(int, lines[i].strip().split())))
                data = np.array(data)
                instance.machine_nos = data[:, 0::2]
                instance.processing_time = data[:, 1::2]
                instance_dict[instance.name] = instance
                # print(instance.__dict__)
            i += 1
    return instance_dict


if __name__ == "__main__":
    instance_dict = load_instances(os.path.join(global_util.get_project_root(), "data/jobshop1.txt"))
    print(instance_dict["ft06"].machine_nos)
    print(instance_dict["ft06"].processing_time)
