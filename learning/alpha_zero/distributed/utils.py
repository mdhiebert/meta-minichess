from tqdm import tqdm
from multiprocessing import Pool

class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

def run_apply_async_multiprocessing(func, argument_list, num_processes, desc=''):
    '''https://leimao.github.io/blog/Python-tqdm-Multiprocessing/'''
    pool = Pool(processes=num_processes)

    jobs = [pool.apply_async(func=func, args=(*argument,)) if isinstance(argument, tuple) else pool.apply_async(func=func, args=(argument,)) for argument in argument_list]
    pool.close()
    result_list_tqdm = []
    for job in tqdm(jobs, desc=desc):
        result_list_tqdm.append(job.get())

    return result_list_tqdm

def run_apply_async_multiprocessing_no_visual(func, argument_list, num_processes, desc=''):
    '''https://leimao.github.io/blog/Python-tqdm-Multiprocessing/'''
    pool = Pool(processes=num_processes)

    jobs = [pool.apply_async(func=func, args=(*argument,)) if isinstance(argument, tuple) else pool.apply_async(func=func, args=(argument,)) for argument in argument_list]
    pool.close()
    result_list_tqdm = []
    for job in jobs:
        result_list_tqdm.append(job.get())

    return result_list_tqdm
