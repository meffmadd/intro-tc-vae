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


class DisentanglementDataset(data.Dataset):
    @classmethod
    def load_data(cls, resize: int = 256) -> "DisentanglementDataset":
        raise NotImplementedError()

class DSprites(DisentanglementDataset):
    def __init__(self, arr, resize: int = 64) -> None:
        self.imgs = arr['imgs']
        self.latents_values = arr['latents_values']
        self.latents_classes = arr['latents_classes']
        self.resize = resize
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img = Image.fromarray(self.imgs[index])
        label = self.latents_values[index]
        if self.resize != 64:
            img.resize((self.resize, self.resize), Image.BICUBIC)
        return img, label 

    @classmethod
    def load_data(cls, resize: int = 64) -> "DisentanglementDataset":
        data_dir = os.path.expanduser("~/dsprites-dataset")
        data_path = os.path.join(data_dir, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
        arr = np.load(data_path)
        return DSprites(arr, resize=resize)

class UkiyoE(DisentanglementDataset):
    def __init__(self, root, df, category, resize=256):
        self.root = root
        self.labels = df
        self.category = category
        self.resize = resize

        self.entries = [
            tuple(r)
            for r in df[["filename", category]].to_numpy()
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

    def __getitem__(self, index):
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
        return image, label

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
    if is_gray is False and img.mode is not "RGB":
        img = img.convert("RGB")
    if is_gray and img.mode is not "L":
        img = img.convert("L")

    if is_mirror and random.randint(0, 1) is 0:
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
