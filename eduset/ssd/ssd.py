# source: https://debuggercafe.com/train-ssd300-vgg16/
# source: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
import torchvision
import os
import torch

import cv2 as cv
import numpy as np
import glob as glob
import matplotlib.pyplot as plt
import albumentations as a

from xml.etree import ElementTree as Et
from torchvision.models.detection._utils import retrieve_out_channels
from torchvision.models.detection import SSD300_VGG16_Weights
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.metrics import confusion_matrix
from albumentations.pytorch import ToTensorV2


class VOCDetection(Dataset):
    def __init__(self, root: str, classes: list[str], width=300, height=300, transforms=None, name=None):
        self.transforms = transforms
        self.root = root
        self.height = height
        self.width = width
        self.classes = classes
        self.name = name
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm', '*.JPG']
        self.all_image_paths = []

        # Get all the image paths in sorted order.
        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(os.path.join(self.root, file_type)))
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
        self.all_images = sorted(self.all_images)

    def __len__(self):
        return len(self.all_images)

    def __repr__(self):
        return (f"Name:\t\t{self.name}\n"
                f"Classes:\t{self.classes}\n"
                f"Items:\t\t{self.__len__()}\n")

    def __getitem__(self, idx):
        # Capture the image name and the full image path.
        image_name = self.all_images[idx]
        image_path = os.path.join(self.root, image_name)

        # Read and preprocess the image.
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32)
        image_resized = (cv.resize(image, dsize=(self.width, self.height))) / 255

        # Capture the corresponding XML file for getting the annotations.
        annot_filename = os.path.splitext(image_name)[0] + '.xml'
        annot_file_path = os.path.join(self.root, annot_filename)

        boxes = []
        labels = []
        tree = Et.parse(annot_file_path)
        root = tree.getroot()

        # Original image width and height.
        image_width = image.shape[1]
        image_height = image.shape[0]

        # Box coordinates for xml files are extracted
        # and corrected for image size given.
        for member in root.findall('object'):
            # Get label and map the `classes`.
            labels.append(self.classes.index(member.find('name').text))

            # Left corner x-coordinates.
            x_min = int(member.find('bndbox').find('xmin').text)
            # Right corner x-coordinates.
            x_max = int(member.find('bndbox').find('xmax').text)
            # Left corner y-coordinates.
            y_min = int(member.find('bndbox').find('ymin').text)
            # Right corner y-coordinates.
            y_max = int(member.find('bndbox').find('ymax').text)

            # Resize the bounding boxes according
            # to resized image `width`, `height`.
            x_min_final = (x_min / image_width) * self.width
            x_max_final = (x_max / image_width) * self.width
            y_min_final = (y_min / image_height) * self.height
            y_max_final = (y_max / image_height) * self.height

            # Check that all coordinates are within the image.
            if x_max_final > self.width:
                x_max_final = self.width
            if y_max_final > self.height:
                y_max_final = self.height

            boxes.append([x_min_final, y_min_final, x_max_final, y_max_final])

        # Bounding box to tensor.
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Area of the bounding boxes.
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 \
            else torch.as_tensor(boxes, dtype=torch.float32)
        # No crowd instances.
        is_crowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # Labels to tensor.
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Prepare the final `target` dictionary.
        target = {"boxes": boxes,
                  "labels": labels,
                  "area": area,
                  "is_crowd": is_crowd}

        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.transforms:
            sample = self.transforms(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        if np.isnan((target["boxes"]).numpy()).any() or target["boxes"].shape == torch.Size([0]):
            target["boxes"] = torch.zeros((0, 4), dtype=torch.int64)

        return image_resized, target

    def vizualize(self, idx: int, classes_vis: dict) -> None:
        image, target = self.__getitem__(idx)
        image = cv.cvtColor(image.permute(1, 2, 0).numpy(), cv.COLOR_RGB2BGR)

        for box_count in range(len(target["boxes"])):
            box = target['boxes'][box_count]
            label = classes_vis[(target['labels'][box_count]).item() - 1]["name"]
            color = classes_vis[(target['labels'][box_count]).item() - 1]["color"]

            cv.rectangle(
                image,
                (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                color=color,
                thickness=2
            )
            cv.putText(
                image,
                label,
                org=(int(box[0]), int(box[1] - 5)),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=color,
                thickness=3
            )

        plt.imshow(image)


class VOCDataLoader(DataLoader):
    def __init__(self,
                 dataset: VOCDetection,
                 batch_size=16,
                 shuffle=False,
                 num_workers=0,
                 drop_last=True):
        super(VOCDataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )


def collate_fn(batch):
    return tuple(zip(*batch))


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


class Model:
    def __init__(self, epochs: int, train_dataloader: DataLoader | None, val_dataloader: DataLoader | None, out_dir: str):
        self.epochs = epochs
        self.device = torch.device('cpu')
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.out_dir = out_dir

        self.params = 0
        self.num_classes = 0
        self.train_loss_history = Averager()
        self.optimizer = None
        self.scheduler = None
        self.model = None
        self.name = "ssd_model"

    def create(self, num_classes=91, size=300) -> None:
        self.num_classes = num_classes
        # Load the Torchvision pretrained model.
        self.model = torchvision.models.detection.ssd300_vgg16(
            weights=SSD300_VGG16_Weights.COCO_V1,
        )
        # Retrieve the list of input channels.
        in_channels = retrieve_out_channels(self.model.backbone, size=(size, size))

        # List containing number of anchors based on aspect ratios.
        num_anchors = self.model.anchor_generator.num_anchors_per_location()

        # The classification head.
        self.model.head.classification_head = torchvision.models.detection.ssd.SSDClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
        )

        # Image size for transforms.
        self.model.transform.min_size = (size,)
        self.model.transform.max_size = size,

        self.get_params()

    def switch_gpu(self) -> None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        self.device = device
        print(f"Model switched to {device}")

    def train(self) -> None:
        # Metrics monitor
        train_loss_score = np.array([])
        map_score = np.array([])
        map50_score = np.array([])
        confusion_matrix = []

        prev_map = 0
        for epoch in range(self.epochs):
            print("\n")
            self.train_loss_history.reset()
            train_loss = self.train_epoch(self.train_dataloader)
            metric_summary = self.val_epoch(self.val_dataloader)

            print(f"Epoch #{epoch + 1} train loss: {self.train_loss_history.value:.3f}")
            print(f"Epoch #{epoch + 1} mAP: {metric_summary['map']}")

            # Update metrics monitor
            train_loss_score = np.append(train_loss_score, train_loss)
            map_score = np.append(map_score, metric_summary["map"])
            map50_score = np.append(map50_score, metric_summary["map_50"])
            # confusion_matrix.append(metric_summary["confusion_matrix"])

            if metric_summary["map"] > prev_map:
                self.save(epoch+1, metric_summary["map"], train_loss)
                prev_map = metric_summary["map"]

            self.scheduler.step()

        self.save_metrics(train_loss_score, map_score, map50_score)

    def save_metrics(self, loss_score: np.array, map_score: np.array, map50_score: np.array,
                     confusion_matrix=None) -> None:
        out_dir = f"{self.out_dir}/model"
        os.makedirs(name=out_dir, exist_ok=True)

        graphs = {"Loss score": loss_score, "mAP": map_score, "mAP50": map50_score}
        for key, value in graphs.items():
            x = np.arange(1, len(value)+1)
            plt.scatter(x, value)
            plt.plot(x, value)
            plt.ylabel(key)
            plt.xlabel("epoch")

            plt.savefig(f"{out_dir}/{key}.svg", format="svg")
            plt.clf()

        # TODO: add confusion matrix

        print(f"Metrics have been saved to {out_dir}")

    def save(self, epoch: int, map_score: float, loss_score: float) -> None:
        out_dir = f"{self.out_dir}/model"
        os.makedirs(name=out_dir, exist_ok=True)

        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'mAP': map_score,
                    'loss': loss_score},
                   f=f'{out_dir}/{self.name}.pth')

        print(f"Epoch #{epoch}\tModel was saved")

    def val_epoch(self, data_loader) -> dict:
        target = []
        predictions = []

        print("Validating")
        self.model.eval()

        progress_bar = tqdm(total=len(data_loader))
        for data in data_loader:
            images, targets = data

            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                outputs = self.model(images, targets)

            for i in range(len(images)):
                true_dict = dict()
                prediction_dict = dict()
                true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
                true_dict['labels'] = targets[i]['labels'].detach().cpu()
                prediction_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
                prediction_dict['scores'] = outputs[i]['scores'].detach().cpu()
                prediction_dict['labels'] = outputs[i]['labels'].detach().cpu()
                predictions.append(prediction_dict)
                target.append(true_dict)

            progress_bar.set_description(desc="Validating")
            progress_bar.update(1)

        progress_bar.close()

        metric = MeanAveragePrecision()
        metric.update(predictions, target)
        metric_summary = metric.compute()

        return metric_summary

    def train_epoch(self, data_loader) -> float:
        loss_value = 0

        print("Training")
        self.model.train()

        progress_bar = tqdm(total=len(data_loader))
        for data in data_loader:
            self.optimizer.zero_grad()
            images, targets = data

            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            self.train_loss_history.send(loss_value)

            losses.backward()
            self.optimizer.step()

            # update the loss value beside the progress bar for each iteration
            progress_bar.set_description(desc=f"Loss: {loss_value:.4f}")
            progress_bar.update(1)

        progress_bar.close()

        return loss_value

    def get_params(self) -> None:
        self.params = [p for p in self.model.parameters() if p.requires_grad]

    def set_optimizer(self, optimizer) -> None:
        self.optimizer = optimizer

    def set_scheduler(self, scheduler) -> None:
        self.scheduler = scheduler


def get_train_transform():
    return a.Compose([
        a.Blur(blur_limit=3, p=0.1),
        a.MotionBlur(blur_limit=3, p=0.1),
        a.MedianBlur(blur_limit=3, p=0.1),
        a.ToGray(p=0.3),
        a.RandomBrightnessContrast(p=0.3),
        a.ColorJitter(p=0.3),
        a.RandomGamma(p=0.3),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


def get_valid_transform():
    return a.Compose([
        a.Blur(blur_limit=3, p=0.1),
        a.MotionBlur(blur_limit=3, p=0.1),
        a.MedianBlur(blur_limit=3, p=0.1),
        a.ToGray(p=0.3),
        a.RandomBrightnessContrast(p=0.3),
        a.ColorJitter(p=0.3),
        a.RandomGamma(p=0.3),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })
