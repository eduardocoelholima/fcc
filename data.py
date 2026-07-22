import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

def load_inception():
    global save_path
    # Load pre-saved INCEPTION features
    print('Loading Inception_v3 embeddings for CIFAR-10 and CIFAR-100...', end="")
    cifar10_features_path = './data/my_cifar10_features.pt'
    cifar100_features_path = './data/my_cifar100_features.pt'
    features_cifar10 = torch.load(cifar10_features_path)
    features_cifar100 = torch.load(cifar100_features_path)
    print('done')
    save_path = './data/tiny_imagenet_inception.pt'
    print(f"Loading TinyImagenet Inception features loaded from {save_path}...", end="")
    extracted_features = torch.load(save_path)
    print("done")
    return features_cifar10, features_cifar100, extracted_features


def load_dinov3():
    global save_path
    # Load pre-saved DINOv3 embeddings
    print('Loading Dino_v3 embeddings for CIFAR-10 and CIFAR-100...', end="")
    dino_features_cifar10_path = './data/my_cifar10_dinov3.pt'
    dino_features_cifar100_path = './data/my_cifar100_dinov3.pt'
    dino_features_cifar10 = torch.load(dino_features_cifar10_path)
    dino_features_cifar100 = torch.load(dino_features_cifar100_path)
    print('done')
    print(f"Loading TinyImagenet Dinov3 features loaded from {save_path}...", end="")
    save_path = './data/tiny_imagenet_dinov3.pt'
    tiny_dino_extracted_features = torch.load(save_path)
    print("done")
    return dino_features_cifar10, dino_features_cifar100, tiny_dino_extracted_features


def load_cifar_pytorch_datasets():
    # Load the CIFAR-10,100 pytorch training dataset
    print('Loading CIFAR-10 and CIFAR-100 Torch Datasets...', end="")
    transform = transforms.Compose([transforms.ToTensor()])  # No need of transforms
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    trainset100 = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
    print('done')
    return trainset,trainset100


def load_tinyimagenet_metadata():
    # Load tinyimagenet METADATA
    print(f"Loading TinyImagenet classes...", end="")
    tiny_imagenet_cls_names = torch.load('./data/tiny_imagenet_cls_names.pt')
    tiny_imagenet_cls_index = torch.load('./data/tiny_imagenet_cls_index.pt')
    tiny_imagenet_cls_idx_to_name_dict = torch.load('./data/tiny_imagenet_cls_idx_to_name_dict.pt')
    tiny_imagenet_cls_index_to_idx_dict = torch.load('./data/tiny_imagenet_cls_index_to_idx_dict.pt')
    print("done")
    return ( tiny_imagenet_cls_names, tiny_imagenet_cls_index, tiny_imagenet_cls_idx_to_name_dict, tiny_imagenet_cls_index_to_idx_dict )


def sanity_check():
    print(f'CIFAR-10 classes: {trainset.classes}')
    print(f'CIFAR-100 classes: {trainset100.classes}')
    print(f'Tiny-Imagenet-200 classes: {np.unique(tiny_imagenet_cls_names)}')


def main():
    load_inception()
    load_dinov3()
    load_cifar_pytorch_datasets()
    load_tinyimagenet_metadata()
    # sanity_check()

# main()
