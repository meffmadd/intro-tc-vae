from typing import Callable, List, Tuple
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import os
from os.path import join
from PIL import Image, ImageOps
import random
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from torch.utils.data import DataLoader


class WrappedDataLoader:
    def __init__(self, data_loader: DataLoader, pre_process: Callable):
        self.dl = data_loader
        self.func = pre_process

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


class DisentanglementDataset(data.Dataset):
    @property
    def latent_indices(self) -> List[int]:
        raise NotImplementedError()
    
    @property
    def factor_sizes(self) -> List[int]:
        raise NotImplementedError()

# TODO: Test dataset
class MPI3D(DisentanglementDataset):
    def __init__(self, arr, resize: int = 64) -> None:
        self.imgs = arr['images'] * 255
        self.latents_values = np.arange(self.imgs.shape[0])
        self.factor_bases = np.divide(
            np.prod(self.factor_sizes), np.cumprod(self.factor_sizes)
        ).astype(int)
        self.resize = resize
        self.input_transform = transforms.Compose([transforms.ToTensor()])
    
    @classmethod
    def load_data(cls, resize: int = 64) -> "DisentanglementDataset":
        data_dir = os.path.expanduser("~/mpi3d-dataset")
        data_path = os.path.join(data_dir, "mpi3d_toy.npz")
        arr = np.load(data_path)
        return MPI3D(arr, resize=resize)
    
    def _index_to_factor(self, idx: int) -> np.ndarray:
        """Get factor array from index

        Parameters
        ----------
        idx : int
            Index to convert

        Returns
        -------
        np.ndarray
            Factor index array of shape (7,)
        """
        bucket_pos = np.floor_divide(idx, self.factor_bases)
        return np.mod(bucket_pos, self.factor_sizes)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, np.ndarray]:
        img = Image.fromarray(self.imgs[index])
        label = self._index_to_factor(self.latents_values[index])
        if self.resize != 64:
            img = img.resize((self.resize, self.resize), Image.BICUBIC)
        img = self.input_transform(img)
        return img, label 
    
    @property
    def latent_indices(self) -> List[int]:
        return [0, 1, 2, 3, 4, 5, 6]
    
    @property
    def factor_sizes(self) -> List[int]:
        return [6,6,2,3,3,40,40]


class DSprites(DisentanglementDataset):
    def __init__(self, arr, resize: int = 64):
        self.imgs = arr['imgs'] * 255
        self.latents_values = arr['latents_values']
        self.resize = resize
        self.input_transform = transforms.Compose([transforms.ToTensor()])
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, np.ndarray]:
        img = Image.fromarray(self.imgs[index])
        label = self.latents_values[index]
        if self.resize != 64:
            img = img.resize((self.resize, self.resize), Image.BICUBIC)
        img = self.input_transform(img)
        return img, label 
    
    @property
    def latent_indices(self) -> List[int]:
        return [1, 2, 3, 4, 5]
    
    @property
    def factor_sizes(self) -> List[int]:
        return [1, 3, 6, 40, 32, 32]

    @classmethod
    def load_data(cls, resize: int = 64) -> "DisentanglementDataset":
        data_dir = os.path.expanduser("~/dsprites-dataset")
        data_path = os.path.join(data_dir, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
        arr = np.load(data_path)
        return DSprites(arr, resize=resize)

def get_spaced_elements(arr, n):
    """Returns n evenly spaced values from the unique values of the array.

    Parameters
    ----------
    arr : np.array
        The array to select the elements from.
    n : int
        The number of elements to select from the unique values in the array.
    """
    unique_values = np.unique(arr)
    idx =  np.round(np.linspace(0, len(unique_values) - 1, n)).astype(int)
    return unique_values[idx]

class DSpritesSmall(DSprites):
    def __init__(self, arr, resize: int = 64):
        self.latents_values = arr['latents_values']
        # reduce number of unique values for orientation, x position and y position
        rotation_mask = np.isin(self.latents_values[:,3], get_spaced_elements(self.latents_values[:,3], 4))
        x_mask = np.isin(self.latents_values[:,4], get_spaced_elements(self.latents_values[:,4], 3))
        y_mask = np.isin(self.latents_values[:,5], get_spaced_elements(self.latents_values[:,5], 3))
        mask = np.logical_and(rotation_mask, x_mask, y_mask)
        self.latents_values = self.latents_values[mask]
        self.imgs = arr['imgs'][mask] * 255
        self.resize = resize
        self.input_transform = transforms.Compose([transforms.ToTensor()])
    
    @property
    def factor_sizes(self) -> List[int]:
        return [1, 3, 6, 40, 2, 2]

    @classmethod
    def load_data(cls, resize: int = 64) -> "DisentanglementDataset":
        data_dir = os.path.expanduser("~/dsprites-dataset")
        data_path = os.path.join(data_dir, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
        arr = np.load(data_path)
        return DSpritesSmall(arr, resize=resize)


if __name__ == "__main__":
    MPI3D.load_data()

class UkiyoE(data.Dataset):
    def __init__(self, root, df, category, resize=256):
        self.root = root
        self.labels = df[category].astype("category")
        self.category = category
        self.resize = resize

        self.entries = [
            tuple(r)
            for r in zip(df["filename"], self.labels.cat.codes)
            if os.path.exists(os.path.join(self.root, r[0]))
        ]
        self.input_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index) -> Tuple[torch.Tensor, np.ndarray]:
        image_filename, label = self.entries[index]
        image_filepath = os.path.join(self.root, image_filename)
        image = load_image(
            image_filepath,
            input_height=256,
            output_height=self.resize,
            is_mirror=False,
            is_random_crop=False,
        )
        image = self.input_transform(image)
        return image, np.array(label)

    def get_label(self, index) -> str:
        code = self.labels.cat.codes[index]
        return self.labels.cat.categories[code]

    @classmethod
    def load_data(cls, resize: int = 256) -> "DisentanglementDataset":
        data_dir = os.path.expanduser("~/arc-ukiyoe-faces/scratch")
        image_dir = data_dir + "/arc_extracted_face_images"
        return UkiyoE(image_dir, UkiyoE.load_labels(data_dir), "Painter", resize=resize)

    @classmethod
    def load_labels(cls, data_dir) -> pd.DataFrame:
        labels: pd.DataFrame = pd.read_csv(
            data_dir + "/arc_extracted_face_metadata.csv"
        )
        labels.columns = [
            "ACNo.",
            "Print title",
            "Picture name",
            "Official title",
            "Text",
            "Publisher",
            "Format",
            "Direction",
            "Seal",
            "Painter",
            "revised seals",
            "Year in A.D.",
            "Year in Japanese Calender",
            "Region",
            "Theater",
            "Title of play",
            "Reading of Title of play",
            "Performed title",
            "Reading of Performed title",
            "Main performed title",
            "Classification title",
            "Library",
            "Text",
            "homeURL",
            "SmallImageURL",
            "LargeImageURL",
            "filename",
        ]
        labels = labels[["Painter", "Year in A.D.", "Region", "filename"]]
        labels["Painter"] = labels["Painter"].astype(str)
        return labels


def load_image(
    file_path,
    input_height=128,
    input_width=None,
    output_height=128,
    output_width=None,
    crop_height=None,
    crop_width=None,
    is_random_crop=True,
    is_mirror=True,
    is_gray=False,
):
    if input_width is None:
        input_width = input_height
    if output_width is None:
        output_width = output_height
    if crop_width is None:
        crop_width = crop_height

    img = Image.open(file_path)
    if not is_gray and img.mode != "RGB":
        img = img.convert("RGB")
    if is_gray and img.mode != "L":
        img = img.convert("L")

    if is_mirror and random.randint(0, 1) == 0:
        img = ImageOps.mirror(img)

    if input_height is not None:
        img = img.resize((input_width, input_height), Image.BICUBIC)

    if crop_height is not None:
        [w, h] = img.size
        if is_random_crop:
            # print([w,cropSize])
            cx1 = random.randint(0, w - crop_width)
            cx2 = w - crop_width - cx1
            cy1 = random.randint(0, h - crop_height)
            cy2 = h - crop_height - cy1
        else:
            cx2 = cx1 = int(round((w - crop_width) / 2.0))
            cy2 = cy1 = int(round((h - crop_height) / 2.0))
        img = ImageOps.crop(img, (cx1, cy1, cx2, cy2))

    img = img.resize((output_width, output_height), Image.BICUBIC)
    return img
