import copy

import numpy as np


def build_action_choices(initial_process_time_table: np.ndarray):
    rule = ScheduleRule(initial_process_time_table)
    return [
        rule.spt,
        rule.lpt,
        rule.spt_plus_sso,
        rule.lpt_plus_lso,
        rule.spt_multiply_twk,
        rule.lpt_multiply_twk,
        rule.spt_divide_twk,
        rule.lpt_divide_twk,
        rule.spt_multiply_twkr,
        rule.lpt_multiply_twkr,
        rule.spt_divide_twkr,
        rule.lpt_divide_twkr,
        rule.srm,
        rule.lrm,
        rule.srpt,
        rule.lrpt,
        rule.sso,
        rule.lso,
    ]


class ScheduleRule:
    """
    遵循规则：表格中为0的元素代表已经执行完或者不需要执行
    """

    def __init__(self, initial_process_time_table: np.ndarray):
        self.job_num, self.operation_num = initial_process_time_table.shape[0], initial_process_time_table.shape[1]
        self.initial_process_time_table = copy.deepcopy(initial_process_time_table)
        self.total_working_times = np.sum(self.initial_process_time_table, axis=1)

    # def spt(self, process_time_table: np.ndarray, static: bool):
    #     """
    #     :param process_time_table:
    #     :param static: 是否为静态排序，是-使用最开始的process_time_table，否则使用更新后的
    #     :return:
    #     """
    #     # time_arr = np.ones(self.job_num) * np.inf
    #     # for i in range(self.job_num):
    #     #     for j in range(self.operation_num):
    #     #         if process_time_table[i, j] != 0:
    #     #             time_arr[i] = process_time_table[i, j]
    #     #             break
    #     table = self.initial_process_time_table if static else process_time_table
    #     sums = np.sum(table, axis=1)
    #     return np.argmin(sums)
    #
    # def lpt(self, process_time_table: np.ndarray, static: bool):
    #     """
    #     选取最长processing time根据最长
    #     :param process_time_table:
    #     :param static: 是否为静态排序，是-使用最开始的process_time_table，否则使用更新后的
    #     :return:
    #     """
    #     table = self.initial_process_time_table if static else process_time_table
    #     sums = np.sum(table, axis=1)
    #     return np.argmax(sums)

    def spt(self, process_time_table: np.ndarray, **kwargs):
        """
        Select the job with the shortest processing time.
        :param process_time_table:
        :return:
        """
        op_inds = np.ones(self.operation_num, dtype=np.int32) * -1
        time_arr = np.ones(self.job_num) * np.inf
        for i in range(self.job_num):
            for j in range(self.operation_num):
                if process_time_table[i, j] != 0:
                    time_arr[i] = process_time_table[i, j]
                    op_inds[i] = j
                    break
        job_ind = np.argmin(time_arr)
        return job_ind, op_inds[job_ind]

    def lpt(self, process_time_table: np.ndarray, **kwargs):
        """
        Select the job with the longest processing time.
        :param process_time_table:
        :return:
        """
        op_inds = np.ones(self.operation_num, dtype=np.int32) * -1
        time_arr = np.zeros(self.job_num)
        for i in range(self.job_num):
            for j in range(self.operation_num):
                if process_time_table[i, j] != 0:
                    time_arr[i] = process_time_table[i, j]
                    op_inds[i] = j
                    break
        job_ind = np.argmax(time_arr)
        return job_ind, op_inds[job_ind]

    def spt_plus_sso(self, process_time_table: np.ndarray, **kwargs):
        """
        Select the job with the minimum sum of the processing time of the current and subsequent operation.
        :param process_time_table:
        :return:
        """
        op_inds = np.ones(self.operation_num, dtype=np.int32) * -1
        time_arr = np.ones(self.job_num) * np.inf
        for i in range(self.job_num):
            for j in range(self.operation_num):
                if process_time_table[i, j] != 0:
                    time_arr[i] = (
                        process_time_table[i, j]
                        if j == self.operation_num - 1
                        else process_time_table[i, j] + process_time_table[i, j + 1]
                    )
                    op_inds[i] = j
                    break
        job_ind = np.argmin(time_arr)
        return job_ind, op_inds[job_ind]

    def lpt_plus_lso(self, process_time_table: np.ndarray, **kwargs):
        """
        Select the job with the maximum sum of the processing time of the current and subsequent operation.
        :param process_time_table:
        :return:
        """
        op_inds = np.ones(self.operation_num, dtype=np.int32) * -1
        time_arr = np.zeros(self.job_num)
        for i in range(self.job_num):
            for j in range(self.operation_num):
                if process_time_table[i, j] != 0:
                    time_arr[i] = (
                        process_time_table[i, j]
                        if j == self.operation_num - 1
                        else process_time_table[i, j] + process_time_table[i, j + 1]
                    )
                    op_inds[i] = j
                    break
        job_ind = np.argmax(time_arr)
        return job_ind, op_inds[job_ind]

    def spt_multiply_twk(self, process_time_table: np.ndarray, **kwargs):
        """
        Select the job with the minimum product of current processing time and total working time.
        :param process_time_table:
        :return:
        """
        op_inds = np.ones(self.operation_num, dtype=np.int32) * -1
        time_arr = np.ones(self.job_num) * np.inf
        for i in range(self.job_num):
            for j in range(self.operation_num):
                if process_time_table[i, j] != 0:
                    time_arr[i] = process_time_table[i, j] * self.total_working_times[i]
                    op_inds[i] = j
                    break
        job_ind = np.argmin(time_arr)
        return job_ind, op_inds[job_ind]

    def lpt_multiply_twk(self, process_time_table: np.ndarray, **kwargs):
        """
        Select the job with the maximum product of current processing time and total working time.
        :param process_time_table:
        :return:
        """
        op_inds = np.ones(self.operation_num, dtype=np.int32) * -1
        time_arr = np.zeros(self.job_num)
        for i in range(self.job_num):
            for j in range(self.operation_num):
                if process_time_table[i, j] != 0:
                    time_arr[i] = process_time_table[i, j] * self.total_working_times[i]
                    op_inds[i] = j
                    break
        job_ind = np.argmax(time_arr)
        return job_ind, op_inds[job_ind]

    def spt_divide_twk(self, process_time_table: np.ndarray, **kwargs):
        """
        Select the job with the minimum ratio of current processing time and total working time.
        :param process_time_table:
        :return:
        """
        op_inds = np.ones(self.operation_num, dtype=np.int32) * -1
        time_arr = np.ones(self.job_num) * np.inf
        for i in range(self.job_num):
            for j in range(self.operation_num):
                if process_time_table[i, j] != 0:
                    time_arr[i] = process_time_table[i, j] / self.total_working_times[i]
                    op_inds[i] = j
                    break
        job_ind = np.argmin(time_arr)
        return job_ind, op_inds[job_ind]

    def lpt_divide_twk(self, process_time_table: np.ndarray, **kwargs):
        """
        Select the job with the maximum ratio of current processing time and total working time.
        :param process_time_table:
        :return:
        """
        op_inds = np.ones(self.operation_num, dtype=np.int32) * -1
        time_arr = np.zeros(self.job_num)
        for i in range(self.job_num):
            for j in range(self.operation_num):
                if process_time_table[i, j] != 0:
                    time_arr[i] = process_time_table[i, j] / self.total_working_times[i]
                    op_inds[i] = j
                    break
        job_ind = np.argmax(time_arr)
        return job_ind, op_inds[job_ind]

    def spt_multiply_twkr(self, process_time_table: np.ndarray, **kwargs):
        """
        Select the job with the minimum product of current processing time and total working time remaining.
        :param process_time_table:
        :return:
        """
        op_inds = np.ones(self.operation_num, dtype=np.int32) * -1
        time_arr = np.ones(self.job_num) * np.inf
        for i in range(self.job_num):
            for j in range(self.operation_num):
                if process_time_table[i, j] != 0:
                    total_time_remaining = np.sum(process_time_table[i, j:])
                    time_arr[i] = process_time_table[i, j] * total_time_remaining
                    op_inds[i] = j
                    break
        job_ind = np.argmin(time_arr)
        return job_ind, op_inds[job_ind]

    def lpt_multiply_twkr(self, process_time_table: np.ndarray, **kwargs):
        """
        Select the job with the maximum product of current processing time and total working time remaining.
        :param process_time_table:
        :return:
        """
        op_inds = np.ones(self.operation_num, dtype=np.int32) * -1
        time_arr = np.zeros(self.job_num)
        for i in range(self.job_num):
            for j in range(self.operation_num):
                if process_time_table[i, j] != 0:
                    total_time_remaining = np.sum(process_time_table[i, j:])
                    time_arr[i] = process_time_table[i, j] * total_time_remaining
                    op_inds[i] = j
                    break
        job_ind = np.argmax(time_arr)
        return job_ind, op_inds[job_ind]

    def spt_divide_twkr(self, process_time_table: np.ndarray, **kwargs):
        """
        Select the job with the minimum ratio of current processing time and total working time remaining.
        :param process_time_table:
        :return:
        """
        op_inds = np.ones(self.operation_num, dtype=np.int32) * -1
        time_arr = np.ones(self.job_num) * np.inf
        for i in range(self.job_num):
            for j in range(self.operation_num):
                if process_time_table[i, j] != 0:
                    total_time_remaining = np.sum(process_time_table[i, j:])
                    time_arr[i] = process_time_table[i, j] / total_time_remaining
                    op_inds[i] = j
                    break
        job_ind = np.argmin(time_arr)
        return job_ind, op_inds[job_ind]

    def lpt_divide_twkr(self, process_time_table: np.ndarray, **kwargs):
        """
        Select the job with the maximum ratio of current processing time and total working time remaining.
        :param process_time_table:
        :return:
        """
        op_inds = np.ones(self.operation_num, dtype=np.int32) * -1
        time_arr = np.zeros(self.job_num)
        for i in range(self.job_num):
            for j in range(self.operation_num):
                if process_time_table[i, j] != 0:
                    total_time_remaining = np.sum(process_time_table[i, j:])
                    time_arr[i] = process_time_table[i, j] / total_time_remaining
                    op_inds[i] = j
                    break
        job_ind = np.argmax(time_arr)
        return job_ind, op_inds[job_ind]

    def srm(self, process_time_table: np.ndarray, **kwargs):
        """
        Select the job with the shortest remaining machining time not including current operation processing time.
        :param process_time_table:
        :return:
        """
        make_span = kwargs.get("make_span")
        # 每个operation对应的机器编号
        machine_nos = kwargs.get("machine_nos")
        machine_times = kwargs.get("machine_times")

        op_inds = np.ones(self.operation_num, dtype=np.int32) * -1
        time_arr = np.ones(self.job_num) * np.inf
        for i in range(self.job_num):
            for j in range(self.operation_num):
                if process_time_table[i, j] != 0:
                    time_arr[i] = make_span - machine_times[machine_nos[i, j]]
                    op_inds[i] = j
                    break
        job_ind = np.argmin(time_arr)
        return job_ind, op_inds[job_ind]

    def lrm(self, process_time_table: np.ndarray, **kwargs):
        """
        Select the job with the shortest remaining machining time not including current operation processing time.
        :param process_time_table:
        :return:
        """
        make_span = kwargs.get("make_span")
        # 每个operation对应的机器编号
        machine_nos = kwargs.get("machine_nos")
        machine_times = kwargs.get("machine_times")

        op_inds = np.ones(self.operation_num, dtype=np.int32) * -1
        time_arr = np.zeros(self.job_num)
        for i in range(self.job_num):
            for j in range(self.operation_num):
                if process_time_table[i, j] != 0:
                    time_arr[i] = make_span - machine_times[machine_nos[i, j]]
                    op_inds[i] = j
                    break
        job_ind = np.argmax(time_arr)
        return job_ind, op_inds[job_ind]

    def srpt(self, process_time_table: np.ndarray, **kwargs):
        """
        Select the job with the shortest remaining processing time.
        :param process_time_table:
        :return:
        """
        op_inds = np.ones(self.operation_num, dtype=np.int32) * -1
        time_arr = np.ones(self.job_num) * np.inf
        for i in range(self.job_num):
            for j in range(self.operation_num):
                if process_time_table[i, j] != 0:
                    time_arr[i] = np.sum(process_time_table[i, j:])
                    op_inds[i] = j
                    break
        job_ind = np.argmin(time_arr)
        return job_ind, op_inds[job_ind]

    def lrpt(self, process_time_table: np.ndarray, **kwargs):
        """
        Select the job with the longest remaining processing time.
        :param process_time_table:
        :return:
        """
        op_inds = np.ones(self.operation_num, dtype=np.int32) * -1
        time_arr = np.zeros(self.job_num)
        for i in range(self.job_num):
            for j in range(self.operation_num):
                if process_time_table[i, j] != 0:
                    time_arr[i] = np.sum(process_time_table[i, j:])
                    op_inds[i] = j
                    break
        job_ind = np.argmax(time_arr)
        return job_ind, op_inds[job_ind]

    def sso(self, process_time_table: np.ndarray, **kwargs):
        """
        Select the job with the shortest processing time of subsequent operation.
        :param process_time_table:
        :return:
        """
        op_inds = np.ones(self.operation_num, dtype=np.int32) * -1
        time_arr = np.ones(self.job_num) * np.inf
        for i in range(self.job_num):
            for j in range(self.operation_num):
                if process_time_table[i, j] != 0:
                    time_arr[i] = 0 if j == self.operation_num - 1 else process_time_table[i, j + 1]
                    op_inds[i] = j
                    break
        job_ind = np.argmin(time_arr)
        return job_ind, op_inds[job_ind]

    def lso(self, process_time_table: np.ndarray, **kwargs):
        """
        Select the job with the longest processing time of subsequent operation.
        :param process_time_table:
        :return:
        """
        op_inds = np.ones(self.operation_num, dtype=np.int32) * -1
        time_arr = np.zeros(self.job_num)
        for i in range(self.job_num):
            for j in range(self.operation_num):
                if process_time_table[i, j] != 0:
                    time_arr[i] = 0 if j == self.operation_num - 1 else process_time_table[i, j + 1]
                    op_inds[i] = j
                    break
        job_ind = np.argmax(time_arr)
        return job_ind, op_inds[job_ind]


if __name__ == "__main__":
    initial_process_time_table = np.array([[2, 2, 3], [1, 4, 1], [3, 5, 4]])
    # [7 6 12]
    print(np.sum(initial_process_time_table, axis=1))
    rule = ScheduleRule(initial_process_time_table)

    process_time_table = copy.deepcopy(initial_process_time_table)
    i, j = rule.spt(process_time_table)
    assert i == 1 and j == 0
    i, j = rule.lpt(process_time_table)
    assert i == 2 and j == 0
    i, j = rule.spt_plus_sso(process_time_table)
    assert i == 0 and j == 0
    i, j = rule.lpt_plus_lso(process_time_table)
    assert i == 2 and j == 0
    i, j = rule.lpt_plus_lso(process_time_table)
    assert i == 2 and j == 0
    i, j = rule.spt_multiply_twk(process_time_table)
    assert i == 1 and j == 0
    i, j = rule.lpt_multiply_twk(process_time_table)
    assert i == 2 and j == 0
    i, j = rule.spt_multiply_twkr(process_time_table)
    assert i == 1 and j == 0
    i, j = rule.lpt_multiply_twkr(process_time_table)
    assert i == 2 and j == 0
    i, j = rule.spt_divide_twkr(process_time_table)
    assert i == 1 and j == 0
    i, j = rule.lpt_divide_twkr(process_time_table)
    assert i == 0 and j == 0
