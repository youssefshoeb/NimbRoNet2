# Imports
import os
import math
import torch
import random
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from torchvision import transforms, utils
from scipy.stats import multivariate_normal
from torch.utils.data import Dataset, DataLoader
from utils import reverse_normalize, read_image, default_transforms, convert_image_to_label, convert_label_to_image, resize_image


class DetectionDataset(Dataset):
    def __init__(self, label_data: str, image_folder: str = '', h=480, w=640):
        """
        Takes as input the path to a csv file and the path to the folder where the images are located

        :param: label_data:
            path to a CSV file, the file should have the following columns in
            order: 'filename', 'width', 'height', 'class', 'xmin',
            'ymin', 'xmax', 'ymax' and 'image_id'
        :param: image_folder:
            path to the folder containing the images.
        :param: transform(Optional):
            a torchvision transforms object containing all transformations to be applied on all elements of the dataset
            (all box coordinates are also automatically adjusted to match the modified image-only horizontal flip
            and scaling are supported-)
        """

        # Width and height of the image
        self.width = w
        self.height = h

        # Root directory
        self.root_dir = image_folder

        # ImageNet based normalization
        self.transform = default_transforms()

        # Read CSV with info
        self.labels_dataframe = pd.read_csv(label_data)

        # Labels data
        self.label_map = {"ball": 0, "robot": 1, "goalpost": 2}
        self.label_radius = {'ball': 120, 'robot': 200, 'goalpost': 120}

    def set_resolution(self, h: int, w: int):
        """
        Set new resolution for the dataset.

        :param h: height of the image
        :param w: width of the image
        """
        self.width = w
        self.height = h

    def get_probmap(self, boxes: list, labels: list, original_height: int, original_width: int) -> np.array:
        """
        Takes as input the boxes and labels
        for a given image and returns the
        respective probability map

        :param boxes: bouding boxes coordinates
        :param labels: labels of the boundig boxes
        :param size: probability map size
        :return: torch tensor representing a 2D image
        """
        # Probability map is 1/4 of resolution
        width = self.width
        height = self.height

        # Probability map
        probmap = np.zeros((height, width, 3), dtype='float32')

        # Populate probability map
        for idx, coordinates in enumerate(boxes):
            xmin, xmax = coordinates[0], coordinates[2]
            ymin, ymax = coordinates[1], coordinates[3]

            # Invalid object
            if xmin == xmax == ymin == ymax == -1:
                continue

            # Transform coordinate to new resolution
            xmin = ((xmin * width) // original_width)
            xmax = ((xmax * width) // original_width)
            ymin = ((ymin * height) // original_height)
            ymax = ((ymax * height) // original_height)

            # Radius and center of the object
            radius = self.label_radius[labels[idx]]

            if labels[idx] == "robot" or labels[idx] == "goalpost":
                center = np.array([int(ymin + (ymax-ymin)/2), int(xmin + (xmax-xmin)/2)])

            if labels[idx] == "ball":
                center = np.array([ymin + (ymax-ymin)/2, xmin + (xmax-xmin)/2])

            # Distribution
            idxs = np.meshgrid(np.arange(0, height), np.arange(0, width))
            idxs = np.array(idxs).T.reshape(-1, 2)
            dist = multivariate_normal.pdf(idxs, center, [radius, radius])

            # Normalize distribution values between 0 and 1
            dist = (dist - dist.min()) / (dist.max() - dist.min())

            # Populate probability map
            probmap[:, :, self.label_map[labels[idx]]] += dist.reshape(height, width)

        # Cap values above one to 1
        probmap = np.where(probmap > 1, 1, probmap)

        return probmap

    def __len__(self) -> int:
        """
        :return: the length of the dataset
        """
        return len(self.labels_dataframe['image_id'].unique().tolist())

    def __getitem__(self, idx: int) -> (torch.tensor, dict):
        """
        :param idx: index of image we want to get
        :return: tuple containing image and target dictionary
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Read image
        object_entries = self.labels_dataframe.loc[self.labels_dataframe['image_id'] == idx]
        image = read_image(os.path.join(self.root_dir, object_entries.iloc[0, 0]))

        # Bounding boxes labels and positions
        boxes, labels = [], []
        for object_idx, row in object_entries.iterrows():
            boxes.append(self.labels_dataframe.iloc[object_idx, 4:8])
            labels.append(self.labels_dataframe.iloc[object_idx, 3])

        # Get relative probability map
        probmap = self.get_probmap(boxes, labels, image.shape[0], image.shape[1])

        # Resize image to target resolution
        image = transforms.ToPILImage()(image)
        image = resize_image(image, self.height, self.width)

        # Normalize image and target to tensor
        image = self.transform(image)
        probmap = transforms.ToTensor()(probmap)

        return image, probmap


class SegmentationDataset(Dataset):
    def __init__(self, base_dir: str, h=480, w=640):
        """
        Takes as input the path to the directory containing the images and labels
        and any transformations to be applied on the images.

        :param: base_dir:
            path to the folder containing the image folder which contains the images
            and target folder which contains the segmentation results.
        :param: transform(Optional):
            a torchvision transforms object containing all transformations to be applied on all elements of the dataset
            (all box coordinates are also automatically adjusted to match the modified image-only horizontal flip
            and scaling are supported-)
        """

        # Get image and labels paths for segmentation data
        self.img_paths = glob(base_dir + "/**/image/*.jpg", recursive=True)
        self.lbl_paths = glob(base_dir + "/**/target/*.png", recursive=True)

        # Sort lists to have a 1:1 mapping
        self.img_paths.sort()
        self.lbl_paths.sort()

        # Make sure we have the same size for both
        assert len(self.img_paths) == len(self.lbl_paths)

        # Defaults transformations
        self.transform = default_transforms()

        # Orignal size
        self.h = h
        self.w = w

    def set_resolution(self, h: int, w: int):
        """
        Set the height and width for the images in the dataset
        :param h: height of image
        :param w: width of the image
        """
        self.w = w
        self.h = h

    def __len__(self) -> int:
        """
        :return: the length of the dataset
        """
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        """
        :param idx: index of image we want to get
        :return: tuple containing image and segmentation label
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image
        image = read_image(str(self.img_paths[idx]))

        # Get label
        label = read_image(str(self.lbl_paths[idx]))

        # Resize images and labels
        image = transforms.ToPILImage()(image)
        label = transforms.ToPILImage()(label)
        image = resize_image(image, self.h, self.w)
        label = resize_image(label, self.h, self.w)

        # Convert image to one channel labels
        label = convert_image_to_label(label)

        # Perform transformations
        image = self.transform(image)
        label = transforms.ToTensor()(label)

        return image, label
