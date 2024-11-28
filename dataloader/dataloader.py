# from dataset import
from dataloader.collate_fn import collate_fn

from torch.utils.data import DataLoader


class DemoDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_train_dataloader(self):
        pass

    def get_val_dataloader(self):
        pass


# def build_dataloader(dataset,):
#     pass


if __name__ == '__main__':
    # demo_dataloader = build_dataloader(dataset)
    pass
