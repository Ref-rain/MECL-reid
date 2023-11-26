
class BaseDataset(object):
    """
    Base class of reid dataset
    """

    @staticmethod
    def get_imagedata_info(data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, dataset, dataset_type):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(dataset)

        print("Dataset statistics:")
        print("  ------------------------------------------")
        print("  {:<9s}| {:^5s} | {:^8s} | {:^9s}".format('subset', '# ids', '# images', '# cameras'))
        print("  ------------------------------------------")
        print("  {:<9s}| {:^5d} | {:^8d} | {:^9d}".format(dataset_type, num_train_pids, num_train_imgs, num_train_cams))
        print("  ------------------------------------------")

    @staticmethod
    def _relabel(label_list):
        sorted_pids = sorted(list(set(label_list)))
        label_dict = dict()
        for idx, pid in enumerate(sorted_pids):
            if pid in label_dict.keys():
                continue
            label_dict[pid] = idx

        relabeled_list = [label_dict[pid] for pid in label_list]
        return relabeled_list
