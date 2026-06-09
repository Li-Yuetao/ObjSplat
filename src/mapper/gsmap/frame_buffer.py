import torch
from torch.multiprocessing import Value

class FrameBuffer:
    def __init__(self, config, image_size):
        
        self.counter = Value('i', 0)
        self.ready = Value('i', 0)
        self.ht = ht = image_size[0]
        self.wd = wd = image_size[1]
        self.config = config
        
        ### state attributes ###
        self.tstamp = torch.zeros(config['pre_num'], device="cuda", dtype=torch.float).share_memory_()
        self.rgbs = torch.zeros(config['pre_num'], 3, ht, wd, device="cpu", dtype=torch.float).share_memory_()
        self.depths = torch.ones(config['pre_num'], 1, ht, wd, device="cpu", dtype=torch.float).share_memory_()
        self.masks = torch.zeros(config['pre_num'], 1, ht, wd, device="cpu", dtype=torch.float).share_memory_()
        self.poses = torch.zeros(config['pre_num'], 4, 4, device="cuda", dtype=torch.float).share_memory_()
        self.gt_poses = torch.zeros(config['pre_num'], 4, 4, device="cuda", dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(config['pre_num'], 3, 3, device="cuda", dtype=torch.float).share_memory_()
        
    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        # self.dirty[index] = True
        self.tstamp[index] = item[0]
        self.rgbs[index] = item[1]

        if item[2] is not None:
            self.depths[index] = item[2]

        if item[3] is not None:
            self.masks[index] = item[3]

        if item[4] is not None:
            self.poses[index] = item[4]
        
        if item[5] is not None:
            self.gt_poses[index] = item[5]

        if item[6] is not None:
            self.intrinsics[index] = item[6]
        else:
            self.intrinsics[index] = self.intrinsics[0].clone()

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)
        
        
        
        