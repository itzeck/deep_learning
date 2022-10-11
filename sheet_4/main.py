import torch
from torch import nn
from torch.nn import LogSoftmax

import numpy as np
import idx2numpy

from torchvision import models, transforms

from PIL import Image

from captum.attr import (
    GuidedBackprop,
    Occlusion,
    Saliency,
    IntegratedGradients,
)

from captum.attr import visualization as vis
from matplotlib import pyplot as plt

class mnist_model(nn.Module):
    def __init__(self):
        super(mnist_model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(20, 64, kernel_size=(3, 3))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(3, 3))
        self.fc1 = nn.Linear(64 * 2 * 2, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        # print(x.shape)
        x = x.view(-1, 64 * 2 * 2)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.logSoftmax(x)
        return x


#source: https://boscoj2008.github.io/customCNN/
normalize = transforms.Normalize(
    mean=0.2859,
    std=0.3530,
)
convert = transforms.ToTensor()

# Prepare network.
model = mnist_model()
model.load_state_dict(torch.load("../sheet_1/model_sheet_1.pth"))
model.eval()

file = "../sheet_1/data/FashionMNIST/raw/t10k-images-idx3-ubyte"
arr_decoded = idx2numpy.convert_from_file(file)

#make writable
images = np.copy(arr_decoded)

plt.imsave("trouser1.jpg",images[2])
trouser1 = Image.open("trouser1.jpg")
trouser1 = trouser1.convert('L')
tensor_trouser1 = normalize(convert(trouser1))[None, ...]

plt.imsave("trouser2.jpg",images[3])
trouser2 = Image.open("trouser2.jpg")
trouser2 = trouser2.convert('L')
tensor_trouser2 = normalize(convert(trouser2))[None, ...]

plt.imsave("pullover1.jpg",images[46])
pullover1 = Image.open("pullover1.jpg")
pullover1 = pullover1.convert('L')
tensor_pullover1= normalize(convert(pullover1))[None, ...]

plt.imsave("pullover2.jpg",images[20])
pullover2 = Image.open("pullover2.jpg")
pullover2 = pullover2.convert('L')
tensor_pullover2= normalize(convert(pullover2))[None, ...]

label_names = ["T-shirt/top",
               "Trouser",
               "Pullover",
               "Dress",
               "Coat",
               "Sandal",
               "Shirt",
               "Sneaker",
               "Bag",
               "Ankle boot"]


def get_top_predictions():

    predictions = model(tensor_pullover2).squeeze()
    values = torch.softmax(predictions, 0)
    top_values, top_idx = torch.topk(values, 10)
    print(top_idx)

    guided = GuidedBackprop(model)

    figure, axes = plt.subplots(
        5, 2, figsize=(2 * 3.8, 5 * 3.8)
    )

    for axis, i, v in zip(
            axes.flatten(),
            top_idx.cpu().squeeze(),
            top_values.cpu().squeeze(),

    ):
        attribution = (
            guided.attribute(tensor_pullover2, target=int(i))
            .numpy()
            .squeeze(0)

        )

        vis.visualize_image_attr(
            np.transpose(attribution, (1, 2, 0)),
            original_image=Image.open("pullover2.jpg"),
            method="heat_map",
            sign="positive",
            plt_fig_axis=(figure, axis),
            use_pyplot=False,
        )

        axis.set_title(
            label_names[i].split(",")[0] + f" {v:0.3f}"
        )

    plt.tight_layout()
    #plt.show()
    figure.savefig("pullover2_top_10.svg", bbox_inches="tight")

def get_local_explanations():
    methods = [
        (Saliency(model), {}),
        (GuidedBackprop(model), {}),
        (IntegratedGradients(model), {}),
    ]

    figure, axes = plt.subplots(
        1, 3, figsize=(12,16)
    )

    for axis, (method, params) in zip(
            axes.flatten(), methods
    ):
        attribution = (
            method.attribute(
                tensor_trouser1, target=model(tensor_trouser1).argmax().item(), **params
            )
            .cpu()
            .numpy()
            .squeeze(0)
        )

        vis.visualize_image_attr(
            np.transpose(attribution, (1, 2, 0)),
            original_image=Image.open("trouser1.jpg"),
            method="heat_map",
            sign="positive",
            show_colorbar=True,
            plt_fig_axis=(figure, axis),
            use_pyplot=False,
        )
        axis.set_title(type(method).__name__)
    plt.tight_layout()
    figure.savefig("local_explanations_before_attack.svg", bbox_inches="tight")

#Attack
trouser_image = images[2]
trouser_image[15:24,5:17] = 0
plt.imshow(trouser_image)
plt.show()

plt.imsave("trouser1_attacked.jpg", trouser_image)
trouser1_attacked = Image.open("trouser1_attacked.jpg")
trouser1_attacked = trouser1_attacked.convert('L')
tensor_trouser1_attacked = normalize(convert(trouser1_attacked))[None, ...]

def get_local_explanations_attacked():

    print(f"Predicted class of attacked trouser image: {model(tensor_trouser1_attacked).argmax().item()}")

    methods = [
        (Saliency(model), {}),
        (GuidedBackprop(model), {}),
        (IntegratedGradients(model), {}),
    ]

    figure, axes = plt.subplots(
        1, 3, figsize=(12, 16)
    )

    for axis, (method, params) in zip(
            axes.flatten(), methods
    ):
        attribution = (
            method.attribute(
                tensor_trouser1_attacked, target=model(tensor_trouser1_attacked).argmax().item(), **params
            )
            .cpu()
            .numpy()
            .squeeze(0)
        )

        vis.visualize_image_attr(
            np.transpose(attribution, (1, 2, 0)),
            original_image=Image.open("trouser1_attacked.jpg"),
            method="heat_map",
            sign="positive",
            show_colorbar=True,
            plt_fig_axis=(figure, axis),
            use_pyplot=False,
        )
        axis.set_title(type(method).__name__)
    plt.tight_layout()
    figure.savefig("local_explanations_after_attack.svg", bbox_inches="tight")

get_local_explanations_attacked()