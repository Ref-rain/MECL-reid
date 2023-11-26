from torch.utils.data import DataLoader

from .data import transforms as T
from .data.preprocessor import PreProcessor
from .data.sampler import DistributedRandomIdentitySampler
from .image_layer import Image_Layer


class DataBuilder(object):
    def __init__(self, args):
        super(DataBuilder, self).__init__()
        self.args = args
        self.normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    def _build_train_loader(self, dataset):
        train_transformer = T.Compose([
            #T.ImageNetPolicy(self.args.iters),
            T.Resize((self.args.height, self.args.width)),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop((self.args.height, self.args.width)),
            T.RandomSizedEarserImage(),
            T.ToTensor(),
            self.normalizer
        ])
        sampler = DistributedRandomIdentitySampler(dataset,
                                                   self.args.batch_size,
                                                   self.args.num_instances,
                                                   max_iter=self.args.iters
                                                   )

        train_loader = DataLoader(PreProcessor(dataset, root=self.args.root, transform=train_transformer),
                                  batch_size=self.args.batch_size,
                                  num_workers=self.args.workers,
                                  sampler=sampler,
                                  shuffle=False,
                                  pin_memory=False
                                  )

        return train_loader

    def _build_test_loader(self, query_dataset, gallery_dataset):
        test_transformer = T.Compose([
            T.Resize((self.args.height, self.args.width)),
            T.ToTensor(),
            self.normalizer
        ])

        test_set = list(set(query_dataset) | set(gallery_dataset))
        test_loader = DataLoader(PreProcessor(test_set, root=self.args.root, transform=test_transformer),
                                 batch_size=self.args.batch_size,
                                 num_workers=self.args.workers,
                                 shuffle=False,
                                 pin_memory=False)

        return test_loader

    def build_data(self, is_train, image_list=None):
        if image_list is not None:
            dataset = Image_Layer(image_list, is_train=True)
            data_loader = self._build_train_loader(dataset.data)
            return data_loader, dataset

        # default
        if is_train:
            train_dataset = Image_Layer(self.args.train_list, is_train=True)
            train_loader = self._build_train_loader(train_dataset.data)
            return train_loader, train_dataset

        query_dataset = Image_Layer(self.args.query_list, is_query=True)
        gallery_dataset = Image_Layer(self.args.gallery_list, is_gallery=True)
        test_loader = self._build_test_loader(query_dataset.data, gallery_dataset.data)
        return test_loader, query_dataset, gallery_dataset
