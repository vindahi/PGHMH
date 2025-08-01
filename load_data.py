from open_clip.simple_tokenizer import SimpleTokenizer
import os
import numpy as np
import scipy.io as scio

from torch.utils.data import Dataset
import torch
import random
from PIL import Image
from torchvision.transforms import Compose, RandomCrop, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode, RandomResizedCrop, RandomHorizontalFlip
from utilss import *


dataset_root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasetsave")

class BaseDataset(Dataset):
    def __init__(self,
                 captions: dict,
                 indexs: dict,
                 labels: dict,
                 img_path: str,
                 is_train=True,
                 tokenizer=SimpleTokenizer(),
                 maxWords=77,
                 imageResolution=224,
                 class_name_list=None
                 ):

        if class_name_list is None:
            class_name_list = []
        self.captions = captions
        self.indexs = indexs
        self.labels = labels
        self.maxWords = maxWords
        self.tokenizer = tokenizer
        self.img_path = img_path
        self.class_name_list = class_name_list
        self.prompt_prefix = "a photo of a"
        self.is_train = is_train

        self.transform = Compose([
            Resize(256, interpolation=InterpolationMode.BICUBIC),
            RandomHorizontalFlip(),
            RandomCrop(imageResolution),
            ToTensor(),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ]) if is_train else Compose([
            Resize(256, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(imageResolution),
            ToTensor(),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

        self.__length = len(self.indexs)

    def __len__(self):
        return self.__length

    def _load_image(self, index: int) -> torch.Tensor:
        image_path = os.path.join(self.img_path, self.indexs[index].split("/")[-1].strip())
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image

    def _load_text(self, index: int):
        captions = self.captions[index]
        use_cap = captions[random.randint(0, len(captions) - 1)]
        words = self.tokenizer.tokenize(use_cap)
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.maxWords - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
        caption = self.tokenizer.convert_tokens_to_ids(words)
        while len(caption) < self.maxWords:
            caption.append(0)
        caption = torch.tensor(caption)
        key_padding_mask = (caption == 0)
        return caption, key_padding_mask

    def _load_label(self, index: int) -> torch.Tensor:
        label = self.labels[index]
        label = torch.from_numpy(label)
        return label

    def get_all_label(self):
        labels = torch.zeros([self.__length, len(self.labels[0])], dtype=torch.int64)
        for i, item in enumerate(self.labels):
            labels[i] = torch.from_numpy(item)
        return labels

    def __getitem__(self, index):
        image = self._load_image(index)
        caption, key_padding_mask = self._load_text(index)
        label = self._load_label(index)
        return image, caption, key_padding_mask, label, index


def split_data(captions, indexs, labels, query_num, train_num, seed=None):
    np.random.seed(seed=seed)

    random_index = np.random.permutation(range(len(indexs)))
    query_index = random_index[: query_num]
    train_index = random_index[query_num: query_num + train_num]
    retrieval_index = random_index[query_num:]

    query_indexs = indexs[query_index]
    query_captions = captions[query_index]
    query_labels = labels[query_index]

    train_indexs = indexs[train_index]
    train_captions = captions[train_index]
    train_labels = labels[train_index]

    retrieval_indexs = indexs[retrieval_index]
    retrieval_captions = captions[retrieval_index]
    retrieval_labels = labels[retrieval_index]

    split_indexs = (query_indexs, train_indexs, retrieval_indexs)
    split_captions = (query_captions, train_captions, retrieval_captions)
    split_labels = (query_labels, train_labels, retrieval_labels)

    return split_indexs, split_captions, split_labels


def generate_dataset(captionFile: str,
                     indexFile: str,
                     labelFile: str,
                     dataset_name: str,
                     maxWords=77,
                     imageResolution=224,
                     query_num=2000,
                     train_num=10000,
                     seed=None,
                     ):
    npy = False
    if captionFile.endswith("mat"):
        
        captions = scio.loadmat(captionFile)
        if "caption" in captions:
            captions = captions["caption"]
        elif "tags" in captions:
            captions = captions["tags"]
        elif "YAll" in captions:
            captions = captions["YAll"]
        else:
            raise RuntimeError("text file is not support, we only read the keys of [caption, tags, YAll].")
        captions = captions[0] if captions.shape[0] == 1 else captions
    elif captionFile.endswith("txt"):
        with open(captionFile, "r") as f:
            captions = f.readlines()
        captions = np.asarray([[item.strip()] for item in captions])
    else:
        raise ValueError("the format of 'captionFile' doesn't support, only support [txt, mat] format.")
    
    if indexFile.endswith("mat"):
        npy = False
        indexs = scio.loadmat(indexFile)
        if "index" in indexs:
            indexs = indexs["index"]
        elif "imgs" in indexs:
            indexs = indexs["imgs"]
        elif "FAll" in indexs:
            indexs = indexs["FAll"]
        else:
            raise RuntimeError("image file is not support, we only read the keys of [caption, tags, YAll].")
    elif indexFile.endswith("npy"):
        npy = True
        indexs = np.load(indexFile)
    else:
        npy = False
        raise RuntimeError("index file is not support, we only read the keys of [*.mat, *.npy].")
    labels = scio.loadmat(labelFile)
    if "category" in labels:
        labels = labels["category"]
    elif "LAll" in labels:
        labels = labels["LAll"]
    elif "labels" in labels:
        labels = labels["labels"]
    elif "label" in labels:
        labels = labels["label"]
    else:
        raise RuntimeError("label file is not support, we only read the keys of [caption, tags, YAll].")

    if dataset_name == 'flickr25k':
        img_path = flickr25k_img_path
        class_name_list = flickr25k_class_name_list
    elif dataset_name == 'nuswide':
        img_path = nuswide_img_path
        class_name_list = nuswide_class_name_list
    else:
        img_path = coco_img_path
        class_name_list = coco_class_name_list

    split_indexs, split_captions, split_labels = split_data(captions, indexs, labels, query_num=query_num,
                                                            train_num=train_num, seed=seed)
    


    

    query_data = BaseDataset(captions=split_captions[0], indexs=split_indexs[0], labels=split_labels[0],
                             img_path=img_path, maxWords=maxWords, imageResolution=imageResolution, class_name_list=class_name_list, is_train=False)
    train_data = BaseDataset(captions=split_captions[1], indexs=split_indexs[1], labels=split_labels[1],
                             img_path=img_path, maxWords=maxWords, imageResolution=imageResolution, class_name_list=class_name_list)
    retrieval_data = BaseDataset(captions=split_captions[2], indexs=split_indexs[2], labels=split_labels[2],
                                 img_path=img_path, maxWords=maxWords, imageResolution=imageResolution, class_name_list=class_name_list, is_train=False)



    return train_data, query_data, retrieval_data


