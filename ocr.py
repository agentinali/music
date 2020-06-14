# import os
#
# root = 'E:/OCR/CASIA/'
# print(os.listdir(root))
import os
import sys
import zipfile
import struct
import pandas as pd
import numpy as np
import tables as tb
import time


def getZ(filename):
    name, end = os.path.splitext(filename)
    if end == '.rar':
        Z = rarfile.RarFile(filename)
    elif end == '.zip':
        Z = zipfile.ZipFile(filename)
    return Z


class Bunch(dict):

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.__dict__ = self


class MPF:

    def __init__(self, fp):
        self.fp = fp
        header_size = struct.unpack('l', self.fp.read(4))[0]
        self.code_format = self.fp.read(8).decode('ascii').rstrip('\x00')
        self.text = self.fp.read(header_size - 62).decode().rstrip('\x00')
        self.code_type = self.fp.read(20).decode('latin-1').rstrip('\x00')
        self.code_length = struct.unpack('h', self.fp.read(2))[0]
        self.data_type = self.fp.read(20).decode('ascii').rstrip('\x00')
        self.nrows = struct.unpack('l', self.fp.read(4))[0]
        self.ndims = struct.unpack('l', self.fp.read(4))[0]

    def __iter__(self):
        m = self.code_length + self.ndims
        for i in range(0, m * self.nrows, m):
            label = self.fp.read(self.code_length).decode('gbk')
            data = np.frombuffer(self.fp.read(self.ndims), np.uint8)
            yield data, label


class MPFBunch(Bunch):

    def __init__(self, root, set_name, *args, **kwds):
        super().__init__(*args, **kwds)
        filename, end = os.path.splitext(set_name)

        if 'HW' in filename and end == '.zip':
            if '_' not in filename:
                self.name = filename
                Z = getZ(f'{root}{set_name}')
                self._get_dataset(Z)
        else:
            # print(f'{filename}不是我们需要的文件！')
            pass

    def _get_dataset(self, Z):
        for name in Z.namelist():
            if name.endswith('.mpf'):
                writer_ = f"writer{os.path.splitext(name)[0].split('/')[1]}"

                with Z.open(name) as fp:
                    mpf = MPF(fp)
                    self.text = mpf.text
                    self.nrows = mpf.nrows
                    self.ndims = mpf.ndims
                    db = Bunch({label: data for data, label in iter(mpf)})
                    self[writer_] = pd.DataFrame.from_dict(db).T


class BunchHDF5(Bunch):
    '''
    pd.read_hdf(path, wname) 可以直接获取 pandas 数据
    '''

    def __init__(self, mpf, *args, **kwds):
        super().__init__(*args, **kwds)

        if 'name' in mpf:
            print(f' {mpf.name} 写入进度条：')

        start = time.time()
        for i, wname in enumerate(mpf.keys()):
            if wname.startswith('writer'):
                _dir = f'{root}mpf/'
                if not os.path.exists(_dir):
                    os.mkdir(_dir)
                self.path = f'{_dir}{mpf.name}.h5'
                mpf[wname].to_hdf(self.path, key=wname, complevel=7)
                k = sys.getsizeof(mpf)  # mpf 的内存使用量
                print('-' * (1 + int((time.time() - start) / k)), end='')

            if i == len(mpf.keys()) - 1:
                print('\n')


class XCASIA(Bunch):

    def __init__(self, root, *args, **kwds):
        super().__init__(*args, **kwds)
        self.paths = []
        print('开始写入磁盘')
        start = time.time()
        for filename in os.listdir(root):
            self.mpf = MPFBunch(root, filename)
            BunchHDF5(self.mpf)
        print(f'总共花费时间 {time.time() - start} 秒。')


root = 'E:/OCR/CASIA/'
# %%time
# xa = XCASIA(root)


import os
import pandas as pd
import tables as tb

root = 'E:/OCR/CASIA/'
print(os.listdir(f'{root}mpf/'))

mpf_root = f'{root}mpf/'
for filename in os.listdir(mpf_root):
    with tb.open_file(f'{mpf_root}{filename}') as h5:
        print(h5.root)
    break

mpf_root = f'{root}mpf/'
for filename in os.listdir(mpf_root):
    h5 = tb.open_file(f'{mpf_root}{filename}')
    break
df = pd.read_hdf(f'{mpf_root}{filename}', key='writer001')
print(df)


