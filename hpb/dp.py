import torch
import pynvml
import math
from torch.utils.data import DataLoader
from .utils import *
from queue import Queue


class OnDeviceDataLoader():

    def __init__(self, dataset, batch_size, device='cpu', shuffle=False, num_workers=4, **kwargs):
        self.ds = dataset
        self.device = device
        self.batchsize = batch_size
        self.shuffle = shuffle
        self.mem_approx()

        self.batch_inds = self.batch_inds_sep()
        self.batch_inds_q = Queue()
        self.inds, self.inps, self.labels, self.ds_size = self.load_data()
        self.batch_count = math.ceil(self.ds_size / self.batchsize)
        plog('On-Device dataset initialized')
    
    def mem_approx(self):
        sample_dl = DataLoader(self.ds, batch_size=1)
        batch_input, batch_label = iter(sample_dl).next()
        sample_size = get_tensor_size(batch_input) + get_tensor_size(batch_label)
        gig_size = sample_size * len(self.ds) / (2 ** 30)
        plog('Dataset size in memory: {:.4g}G'.format(gig_size))
        return gig_size

    def set_batchsize(self, batchsize=100):
        self.batchsize = batchsize
        self.batch_inds = self.batch_inds_sep()
        self.batch_inds_q = Queue()
        self.batch_count = math.ceil(self.ds_size / self.batchsize)
        plog("Modified batchsize to {}".format(self.batchsize))
        return
    
    def __len__(self):
        return self.batch_count

    def load_data(self):
        sample_dl = DataLoader(self.ds, batch_size=2048, num_workers=4)
        inds = np.arange(len(self.ds))
        inps_stack, labels_stack = [], []
        for (inp, label) in sample_dl:
            inps_stack.append(inp.to(self.device))
            labels_stack.append(label.to(self.device))
        
        inps, labels = torch.cat(inps_stack), torch.cat(labels_stack) # pylint: disable=no-member
        ds_size = len(inps)
        return inds, inps, labels, ds_size

    def batch_inds_sep(self):
        batch_inds = []
        for i in np.arange(0, len(self.ds) - 1, self.batchsize):
            batch_inds.append([i, i + self.batchsize])
        batch_inds[-1][-1] = len(self.ds) - 1
        return batch_inds
    
    def init_batch_inds_q(self):
        self.batch_inds_q = Queue()
        for x in self.batch_inds:
            self.batch_inds_q.put(x)
    
    def __next__(self):
        if self.batch_inds_q.empty():
            raise StopIteration
        batch_marks = self.batch_inds_q.get()
        batch_inds = list(range(batch_marks[0], batch_marks[1]))
        inp = self.inps[self.inds[batch_inds]]
        labels = self.labels[self.inds[batch_inds]]
        return [inp, labels]
    
    def __iter__(self):
        self.init_batch_inds_q()
        if self.shuffle:
            np.random.shuffle(self.inds)
        return self
    
    next = __next__

class DataLoader_filtered():

    def __init__(self, dataloader, remain_labels=None):
        self.dl = dataloader
        self.dl_iter = iter(self.dl)
        self.remain_labels = remain_labels
        self.batch_count = len(self.dl)

    def __iter__(self):
        self.dl_iter = iter(self.dl)
        return self
    
    def set_remain_labels(self, remain_labels):
        self.remain_labels = remain_labels
        return
    
    def reset_remain_labels(self):
        self.remain_labels = None
        return
    
    def __next__(self):
        if self.remain_labels is None:
            return self.dl_iter.next()
        else:
            non_emp = False
            while non_emp is False:
                data = self.dl_iter.next()
                data = filter_label(data, self.remain_labels)
                if data[0] is not None:
                    non_emp = True
            return data

    next = __next__

class HDC_iterator():

    def __init__(self, dataloader, sample_func, device, comp_layers, out_device):

        self.dl = dataloader
        self.dl_iter = iter(self.dl)
        self.f = sample_func # H_sample(self, comp_layers, sample_count=1, out_device='cpu', inputs=None, batch_sum=False)
        self.device = device
        self.out_device = out_device
        self.comp_layers = comp_layers
        plog('HDC {} initialized with batchsize {}'.format(sample_func.__name__, dataloader.batch_size))
        self.batch_count = self.dl.batch_count
        self.report_inds = list(range(1, self.batch_count, max(1, int(self.batch_count/10))))

    def __iter__(self):
        return self
    
    def __next__(self):
        inputs, _ = self.dl_iter.next()
        inputs = inputs.to(self.device)
        return self.f(self.comp_layers, out_device=self.out_device, inputs=inputs, batch_sum=False)
    
    next = __next__


def crop_dataset(dataset, crop=1, seed=0):

    if crop != 1:
        assert crop < 1 and crop > 0
        crop_index = int(len(dataset) * crop)
        if seed is not None:
            torch.random.manual_seed(seed)
        dataset, _ = torch.utils.data.random_split(dataset, [crop_index, len(dataset) - crop_index])
        # dataset = dataset[0: crop_index]
        torch.random.manual_seed(np.random.randint(1))

    # plog('Dataset size {}'.format(len(dataset)))
    return dataset

def sample_input(dataset, count=1, remain_labels=None):
    dl = DataLoader(dataset, batch_size=count, shuffle=True, num_workers=1)
    return iter(dl).next()[0]

def filter_label(data, inds):
    inputs, labels = data
    remains = sum([labels == ind for ind in inds]).nonzero().view(-1)
    labels = labels[remains]
    inputs = inputs[remains]
    return inputs, labels
