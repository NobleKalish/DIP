# import cv2 as cv
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as functional
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import matplotlib.pyplot as plt


data_dir = 'data2/train'

model_name = "checkpoint.pth.tar"


def get_random_images(num, test_transforms):
    data = datasets.ImageFolder(data_dir, transform=test_transforms)
    classes = data.classes
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels, classes


def predict_image(image, test_transforms, device, model):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index


def main():
    test_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
    new_state_dict = dict()
    for k, _ in model_checkpoint['state_dict'].items():
        key = k[7:]
        new_state_dict[key] = model_checkpoint['state_dict'][k]

    model_checkpoint['state_dict'] = new_state_dict
    model = models.__dict__["resnet18"](pretrained=True)
    model.load_state_dict(model_checkpoint['state_dict'])
    model.eval()

    image = None
    to_pil = transforms.ToPILImage()
    images, labels, classes = get_random_images(5, test_transforms)
    fig = plt.figure(figsize=(10, 10))
    for ii in range(len(images)):
        image = to_pil(images[ii])
        index = predict_image(image, test_transforms, device, model)
        sub = fig.add_subplot(1, len(images), ii + 1)
        res = int(labels[ii]) == index
        sub.set_title(str(classes[index]) + ":" + str(res))
        plt.axis('off')
        plt.imshow(image)
    plt.show()
    predict_image(image, test_transforms, device, model)


if __name__ == '__main__':
    main()
