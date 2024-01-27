import os
import json
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import color

class RainBase(Dataset):
    def __init__(self,
                 json_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_paths = json_file
        self.data_root = data_root
        with open(self.data_paths,"r") as f:
            self.image_paths = json.load(f)
        self._length = len(self.image_paths)
        self.labels = {
            "background_path": [os.path.join(self.data_root, l[0])
                        for l in self.image_paths],
            "mask_path": [os.path.join(self.data_root, l[1])
                        for l in self.image_paths],       
            "rain_path": [os.path.join(self.data_root, l[2])
                        for l in self.image_paths],            
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def loadImgColor(self,path):
        image = loadImgRGB(path)
        return image

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = self.loadImgColor(example["background_path"])
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)
        example["raw_image"] = np.array(image)

        # image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)

        image_gt = self.loadImgColor(example["rain_path"])
        if self.size is not None:
            image_gt = image_gt.resize((self.size, self.size), resample=self.interpolation)

        example["raw_image_gt"] = np.array(image_gt)
        # image_gt = self.flip(image_gt)
        image_gt = np.array(image_gt).astype(np.uint8)
        example["image_gt"] = (image_gt / 127.5 - 1.0).astype(np.float32)

        mask = loadImgGray(example["mask_path"])
        if self.size is not None:
            mask = mask.resize((self.size, self.size), resample=self.interpolation)
        mask = np.array(mask).astype(np.float32)/255.0
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        example["mask"] = mask
        mask = mask.reshape(self.size,self.size,1)

        masked_image = (1-mask)*image
        example["masked_image"] = (masked_image / 127.5 - 1.0).astype(np.float32)

        return example

class RainBaseList(RainBase):
    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = self.loadImgColor(example["background_path"])
        image_gt = self.loadImgColor(example["rain_path"])
        mask = loadImgGray(example["mask_path"])
        for size_idx, size_hw in enumerate(self.size):    
            image_reszie = image.resize((size_hw, size_hw), resample=self.interpolation)
            example["raw_image_%d" % size_idx] = np.array(image_reszie)
            image_reszie = np.array(image_reszie).astype(np.uint8)
            example["image_%d" % size_idx] = (image_reszie / 127.5 - 1.0).astype(np.float32)

            image_gt_reszie = image_gt.resize((size_hw, size_hw), resample=self.interpolation)
            example["raw_image_gt_%d" % size_idx] = np.array(image_gt_reszie)
            image_gt_reszie = np.array(image_gt_reszie).astype(np.uint8)
            example["image_gt_%d" % size_idx] = (image_gt_reszie / 127.5 - 1.0).astype(np.float32)

            mask_reszie = mask.resize((size_hw, size_hw), resample=self.interpolation)
            mask_reszie = np.array(mask_reszie).astype(np.float32)/255.0
            mask_reszie[mask_reszie < 0.5] = 0
            mask_reszie[mask_reszie >= 0.5] = 1
            example["mask_%d" % size_idx] = mask_reszie
            mask_reszie = mask_reszie.reshape(size_hw,size_hw,1)

            masked_image = (1-mask_reszie)*image_reszie
            example["masked_image_%d" % size_idx] = (masked_image / 127.5 - 1.0).astype(np.float32)

        return example
    
class RainBaseLab(RainBase):
    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = self.loadImgColor(example["background_path"])
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)
        example["raw_image"] = np.array(image)

        # image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] =  color.rgb2lab(image)

        image_gt = self.loadImgColor(example["rain_path"])
        if self.size is not None:
            image_gt = image_gt.resize((self.size, self.size), resample=self.interpolation)

        example["raw_image_gt"] = np.array(image_gt)
        # image_gt = self.flip(image_gt)
        image_gt = np.array(image_gt).astype(np.uint8)
        example["image_gt"] = color.rgb2lab(image_gt)

        mask = loadImgGray(example["mask_path"])
        if self.size is not None:
            mask = mask.resize((self.size, self.size), resample=self.interpolation)
        mask = np.array(mask).astype(np.float32)/255.0
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        example["mask"] = mask
        mask = mask.reshape(self.size,self.size,1)

        masked_image = (1-mask)*image
        example["masked_image"] = color.rgb2lab(masked_image)

        return example
    
class RainBaseListLab(RainBase):
    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = self.loadImgColor(example["background_path"])
        image_gt = self.loadImgColor(example["rain_path"])
        mask = loadImgGray(example["mask_path"])
        for size_idx, size_hw in enumerate(self.size):    
            image_reszie = image.resize((size_hw, size_hw), resample=self.interpolation)
            example["raw_image_%d" % size_idx] = image_reszie
            image_reszie = np.array(image_reszie).astype(np.uint8)
            example["image_%d" % size_idx] = color.rgb2lab(image_reszie)

            image_gt_reszie = image_gt.resize((size_hw, size_hw), resample=self.interpolation)
            example["raw_image_gt_%d" % size_idx] = image_gt_reszie
            image_gt_reszie = np.array(image_gt_reszie).astype(np.uint8)
            example["image_gt_%d" % size_idx] = color.rgb2lab(image_gt_reszie)

            mask_reszie = mask.resize((size_hw, size_hw), resample=self.interpolation)
            mask_reszie = np.array(mask_reszie).astype(np.float32)/255.0
            mask_reszie[mask_reszie < 0.5] = 0
            mask_reszie[mask_reszie >= 0.5] = 1
            example["mask_%d" % size_idx] = mask_reszie
            mask_reszie = mask_reszie.reshape(size_hw,size_hw,1)

            masked_image = (1-mask_reszie)*image_reszie
            example["masked_image_%d" % size_idx] = color.rgb2lab(masked_image)

        return example

def loadImgRGB(path):
    image = Image.open(path)
    if not image.mode == "RGB":
        image = image.convert("RGB")

    # default to score-sde preprocessing
    img = np.array(image).astype(np.uint8)
    crop = min(img.shape[0], img.shape[1])
    h, w, = img.shape[0], img.shape[1]
    img = img[(h - crop) // 2:(h + crop) // 2,
            (w - crop) // 2:(w + crop) // 2]

    image = Image.fromarray(img)

    return image


def loadImgGray(path):
    image = Image.open(path)
    if not image.mode == "L":
        image = image.convert("L")

    # default to score-sde preprocessing
    img = np.array(image).astype(np.uint8)
    crop = min(img.shape[0], img.shape[1])
    h, w, = img.shape[0], img.shape[1]
    img = img[(h - crop) // 2:(h + crop) // 2,
            (w - crop) // 2:(w + crop) // 2]

    image = Image.fromarray(img)

    return image