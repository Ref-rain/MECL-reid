from __future__ import absolute_import

import os

from PIL import Image
from torch.utils.data import Dataset

# import transforms as T
memory_cache = False
try:
    import mc, io

    memory_cache = True
    print("using memory cache")
except ModuleNotFoundError:
    print("missing memory cache")
    pass


class PreProcessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(PreProcessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.initialized = False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    @staticmethod
    def _pil_loader(img_str):
        buff = io.BytesIO(img_str)
        with Image.open(buff) as img:
            img = img.convert('RGB')
        return img

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname

        if self.root is not None:
            fpath = os.path.join(self.root, fname)

        global memory_cache
        if not memory_cache:
            img = Image.open(fpath).convert('RGB')
        else:
            self._init_memcached()
            value = mc.pyvector()
            self.mclient.Get(fpath, value)
            value_str = mc.ConvertBuffer(value)
            img = self._pil_loader(value_str)

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid, index
