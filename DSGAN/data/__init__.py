import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataLoader(opt, isTrain):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt, isTrain)
    return data_loader


def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'aligned':
        from data.aligned_dataset import AlignedDataset
        dataset = AlignedDataset()
    elif opt.dataset_mode == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == 'single':
        from data.single_dataset import SingleDataset
        dataset = SingleDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, isTrain):
        BaseDataLoader.initialize(self, opt, isTrain)
        self.dataset = CreateDataset(opt)

        if(isTrain == 'train'):
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                shuffle=not opt.serial_batches,
                # shuffle=False,
                num_workers=int(opt.nThreads))
        else:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                shuffle=False,
                num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batchSize >= self.opt.max_dataset_size:
                break
            yield data
