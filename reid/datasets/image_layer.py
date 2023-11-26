from __future__ import print_function, absolute_import

from reid.datasets.data.base_dataset import BaseImageDataset


class Image_Layer(BaseImageDataset):
    def __init__(self, image_list, is_train=False, is_query=False, is_gallery=False, verbose=True):
        super(Image_Layer, self).__init__()
        imgs, pids, cams = [], [], []
        with open(image_list) as f:
            for line in f.readlines():
                info = line.strip('\n').split(" ")
                imgs.append(info[0])
                if len(info) == 1:
                    pids.append(0)
                else:
                    pids.append(int(info[1]))

                if len(info) == 3:
                    cams.append(int(info[2]))
                    continue

                # fake data
                if is_train: cams.append(0); continue
                if is_query: cams.append(-1); continue
                if is_gallery: cams.append(1); continue

        if is_train:
            pids = self._relabel(pids)

        self.data = list(zip(imgs, pids, cams))
        self.num_classes, self.num_imgs, self.num_cams = self.get_imagedata_info(self.data)

        if verbose:
            print("=> Dataset information has been loaded.")
            if is_train:
                self.print_dataset_statistics(self.data, 'train')
            if is_gallery:
                self.print_dataset_statistics(self.data, 'gallery')
            if is_query:
                self.print_dataset_statistics(self.data, 'query')



