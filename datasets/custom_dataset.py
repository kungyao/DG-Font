import os
import os.path
import sys
import math

import numpy as np
import torch.utils.data as data
from PIL import Image, ImageOps, ImageFilter

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

# def class_maper(ch):
#     if ch in "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン":
#         return 1
#     elif ch in "ガギグゲゴザジズゼゾダヂヅデドバビブベボ":
#         return 2
#     elif ch in "パピプペポ":
#         return 3
#     return 0

# dakuten
def class_maper(idx):
    if idx <= 45:
        return 1
    elif idx <= 65:
        return 2
    elif idx <= 70:
        return 3
    return 0

def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (
                        path, 
                        class_to_idx[target], 
                        # dakuten
                        class_maper(int(fname.split('.')[0]))
                    )
                    images.append(item)
    return images


class AugmentOperator(object):
    def __init__(self):
        pass

    @staticmethod
    def do_scale(img, mask, scale):
        w, h = img.size
        new_size = (int(w*scale), int(h*scale))
        new_img = img.resize(new_size, resample=Image.NEAREST)
        new_mask = mask.resize(new_size, resample=Image.NEAREST)
        return new_img, new_mask

    @staticmethod
    def do_rotate(img, mask, angle):
        new_img = img.rotate(angle, resample=Image.NEAREST, expand=True, fillcolor=(255, 255, 255))
        new_mask = mask.rotate(angle, resample=Image.NEAREST, expand=True)
        # new_img, new_mask = recalculate_bounding_box(new_img, new_mask)
        return new_img, new_mask

    @staticmethod
    def do_shear(img, mask, shear):
        def pil_space_filled_shear_(img_, shear, mode, fill):
            def calculate_size_(width, height, shear):
                # add abs for minus shear
                new_w = width + abs(int(shear*height))
                new_h = height
                return new_w, new_h
            w, h = img_.size
            new_w, new_h = calculate_size_(w, h, shear)
            new_img = Image.new(mode, (new_w, new_h), color=fill)
            # add if-else for minus shear
            new_img.paste(img_, ((new_w - w) if shear >= 0 else 0, 0))
            new_img = new_img.transform((new_w, new_h), Image.AFFINE, data=(1, shear, 0, 0, 1, 0), resample=Image.NEAREST, fillcolor=fill)
            return new_img
        new_img = pil_space_filled_shear_(img, shear, "RGB", (255, 255, 255))
        new_mask = pil_space_filled_shear_(mask, shear, "L", (0))
        # new_img, new_mask = recalculate_bounding_box(new_img, new_mask)
        return new_img, new_mask

    @staticmethod
    def do_white_edge(img, mask, kernel_size):
        # kernel_size should bigger than zero and odd.
        if kernel_size <= 0 or kernel_size%2 == 0:
            return img, mask
        # Padding image (kernel_size) pixel
        new_img = ImageOps.expand(img, border=kernel_size, fill=(255, 255, 255))
        new_mask = ImageOps.expand(mask, border=kernel_size)
        # Expand mask black pixel which mean the truly label area, so we can get white edge under original image.
        new_mask = new_mask.filter(ImageFilter.MaxFilter(kernel_size))
        return new_img, new_mask

    @staticmethod
    def recalculate_bounding_box(img, mask):
        # get non-zero bounding box.
        true_box = mask.getbbox()
        new_img = img.crop(true_box)
        new_mask = mask.crop(true_box)
        return new_img, new_mask

    @staticmethod
    def to_n_by_n(img, mask):
        w, h = img.size
        if w > h:
            anchor = (0, (w - h)//2)
            new_img = Image.new("RGB", (w, w), color=(255, 255, 255))
            new_img.paste(img, anchor)
            new_mask = Image.new("L", (w, w), color=(0))
            new_mask.paste(mask, anchor)
        elif h > w:
            anchor = ((h - w)//2, 0)
            new_img = Image.new("RGB", (h, h), color=(255, 255, 255))
            new_img.paste(img, anchor)
            new_mask = Image.new("L", (h, h), color=(0))
            new_mask.paste(mask, anchor)
        else:
            new_img = img
            new_mask = mask
        return new_img, new_mask

    def __call__(self, img, mask, params):
        '''
        Parameters
        ----------
        img (PIL.Image): Character image.
        mask (PIL.Image): Character mask image.
        params (Dict[str]) : Augmentation parameters dictionary.
            {
                "angle" : (float)
                    Do rotate coefficient.
                "shear" : (float)
                    Do shear coefficient.
                "kernel_size" : (odd int)
                    Do white edge coefficient.
            }
        '''
        img, mask = self.recalculate_bounding_box(img, mask)
        if 'angle' in params:
            img, mask = self.do_rotate(img, mask, params['angle'])
        if 'shear' in params:
            img, mask = self.do_shear(img, mask, params['shear'])
        if 'kernel_size' in params:
            img, mask = self.do_white_edge(img, mask, params['kernel_size'])
        img, mask = self.recalculate_bounding_box(img, mask)
        img, mask = self.to_n_by_n(img, mask)
        return img, mask


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions, transform=None, target_transform=None, use_stn=False):
        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

        self.augment = None
        if use_stn:
            self.augment = AugmentOperator()

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        classes.sort(key= lambda x:int(x[3:]))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def get_style_sample(self, sample):
        def get_sample_mask(sample: Image.Image):
            sample_mask = sample.copy()
            sample_mask = sample_mask.convert("L")
            sample_mask = sample_mask.point(lambda p: p<200 and 255)
            return sample_mask
        if self.augment is not None:
            params = {
                'scale' : np.random.uniform(0.707, 1.414),
                'angle' : np.random.uniform(-10, 10),
                'shear' : np.random.uniform(-0.2, 0.2),
                # round to digit zero
                'kernel_size' : (int(round(np.random.uniform(8, 21), 0)) // 2) + 1
            }
            style_sample, style_sample_mask = self.augment(sample, get_sample_mask(sample), params)
            # sample.save(f"./samples/{params['scale']}_{params['angle']}_{params['shear']}_{params['kernel_size']}_ori.png")
            # style_sample.save(f"./samples/{params['scale']}_{params['angle']}_{params['shear']}_{params['kernel_size']}_stl.png")
            # style_sample_mask.save(f"./samples/{params['scale']}_{params['angle']}_{params['shear']}_{params['kernel_size']}_stl_mask.png")
            return style_sample
        else:
            return 0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, jp_class = self.samples[index]
        sample = self.loader(path)
        style_sample = self.get_style_sample(sample)
        if self.transform is not None:
            sample = self.transform(sample)          
            if style_sample is not 0:
                style_sample = self.transform(style_sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        imgname = path.split('/')[-1].replace('.JPEG', '')
        return sample, style_sample, target, jp_class, imgname

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# def accimage_loader(path):
#     import accimage
#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        # return accimage_loader(path)
        print("Not support accimage for loading image.")
        return pil_loader(path)
    else:
        return pil_loader(path)


class ImageFolerRemap(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, remap_table=None, with_idx=False, use_stn=False):
        super(ImageFolerRemap, self).__init__(root, loader, IMG_EXTENSIONS, transform=transform, target_transform=target_transform, use_stn=use_stn)

        self.imgs = self.samples
        self.class_table = remap_table
        self.with_idx = with_idx

    def __getitem__(self, index):
        path, target, jp_class= self.samples[index]
        sample = self.loader(path)
        style_sample = self.get_style_sample(sample)
        if self.transform is not None:
            sample = self.transform(sample)
            if style_sample is not 0:
                style_sample = self.transform(style_sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = self.class_table[target]
        if self.with_idx:
            return sample, index, target, jp_class, style_sample
        return sample, style_sample, target, jp_class


class CrossdomainFolder(data.Dataset):
    def __init__(self, root, data_to_use=['photo', 'monet'], transform=None, loader=default_loader, extensions='jpg'):
        self.data_to_use = data_to_use
        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir() and d.name in self.data_to_use]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) and d in self.data_to_use]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class ImageReader(data.Dataset):
    '''
        folder/xxx.ext
        folder/xxy.ext
        folder/xxz.ext
    '''
    def __init__(self, folder_path, transform=None, loader=default_loader):
        super(ImageReader, self).__init__()
        self.imgs = self.collect_image_paths(folder_path)
        self.loader = loader
        self.transform = transform

    def collect_image_paths(self, folder_path):
        imgs = []
        for file in os.listdir(folder_path):
            imgs.append(os.path.join(folder_path, file))
        return imgs

    def __getitem__(self, index):
        img = self.loader(self.imgs[index])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)
