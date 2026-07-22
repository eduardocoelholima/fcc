import torch
import numpy as np
import data
import util


def c100_c10_inception_distance(trainset, trainset100, features_cifar10, features_cifar100 ):
    # distances betweeen c10-classes and c-100-whole-dataset
    fids = []
    for class_name in trainset.classes:
        # create a tensor with the image inception features for selected class
        cifar_labels = torch.tensor(trainset.targets)
        class_index = trainset.classes.index(class_name)
        selected_features_cifar10 = features_cifar10[cifar_labels == class_index]

        # calculate metric
        fid_score_subset = util.fid(features_cifar100, selected_features_cifar10)
        fids.append(fid_score_subset)
        print(
            f"The Fréchet Inception Distance (FID) between C10-'{class_name}' and 'C100-whole-dataset' features is: {fid_score_subset}")
    print(f'fids mean={np.mean(fids)} max={np.max(fids)} class_coverage={np.mean(fids) / np.max(fids)}')


def c10_c100_inception_distance(trainset, trainset100, features_cifar10, features_cifar100):
    # distances betweeen c100-classes and c-10-whole-dataset
    fids = []
    for class_name100 in trainset100.classes:
        # create a tensor with the image inception features for selected class
        cifar_labels100 = torch.tensor(trainset100.targets)
        class_index100 = trainset100.classes.index(class_name100)
        selected_features_cifar100 = features_cifar100[cifar_labels100 == class_index100]

        # calculate metric
        fid_score_subset = util.fid(features_cifar10, selected_features_cifar100)
        fids.append(fid_score_subset)
        print(
            f"The Fréchet Inception Distance (FID) between C100-'{class_name100}' and 'C10-whole-dataset' features is: {fid_score_subset}")
    print(f'fids mean={np.mean(fids)} max={np.max(fids)} class_coverage={np.mean(fids) / np.max(fids)}')


def c100_c10_dinov3_distance(trainset, trainset100, dino_features_cifar10, dino_features_cifar100):
    import torch
    import numpy as np

    # Distances between CIFAR-10 classes and the whole CIFAR-100 dataset using DINO embeddings
    fids_dino = []
    cifar10_labels = torch.tensor(trainset.targets)

    for class_name in trainset.classes:
        # Get the DINO features for the specific CIFAR-10 class
        class_index = trainset.classes.index(class_name)
        selected_dino_c10 = dino_features_cifar10[cifar10_labels == class_index]

        # calculate metric (comparing whole C100 DINO vs specific C10 class DINO)
        fid_score = util.fid(dino_features_cifar100, selected_dino_c10)
        fids_dino.append(fid_score)
        print(
            f"The Frechet DINO Distance (FDD) between C10-'{class_name}' and 'C100-whole-dataset' using DINO is: {fid_score}")

    print(
        f'\nDINO FDDs: mean={np.mean(fids_dino):.4f} max={np.max(fids_dino):.4f} coverage={np.mean(fids_dino) / np.max(fids_dino):.4f}')

def c10_c100_dinov3_distances(trainset, trainset100, dino_features_cifar10, dino_features_cifar100):
    import torch
    import numpy as np

    # Distances between CIFAR-10 classes and the whole CIFAR-100 dataset using DINO embeddings
    fids_dino = []
    cifar10_labels = torch.tensor(trainset.targets)

    c10_whole_mean, c10_whole_cov = util.calculate_mean_covariance(dino_features_cifar10)
    for class_name100 in trainset100.classes:
        # Get the DINO features for the specific CIFAR-100 class
        cifar_labels100 = torch.tensor(trainset100.targets)
        class_index100 = trainset100.classes.index(class_name100)
        selected_dino_c100 = dino_features_cifar100[cifar_labels100 == class_index100]

        # Calculate metric (comparing whole CIFAR-10 DINO vs specific CIFAR-100 class DINO)
        c100_selected_mean, c100_selected_cov = util.calculate_mean_covariance(selected_dino_c100)
        fid_score = util.calculate_fid(c10_whole_mean, c10_whole_cov, c100_selected_mean, c100_selected_cov)
        fids_dino.append(fid_score)
        print(
            f"The Fréchet DINO Distance (FDD) between C100-'{class_name100}' and 'C10-whole-dataset' using DINO is: {fid_score}")

    print(
        f'\nDINO FDDs: mean={np.mean(fids_dino):.4f} max={np.max(fids_dino):.4f} coverage={np.mean(fids_dino) / np.max(fids_dino):.4f}')


def tiny_c10_inception_distance(trainset, features_cifar10, extracted_features):
    # distances betweeen c10-classes and tiny-imagenet-whole-dataset
    fids = []
    for class_name in trainset.classes:
        # create a tensor with the image inception features for selected class
        cifar_labels = torch.tensor(trainset.targets)
        class_index = trainset.classes.index(class_name)
        selected_features_cifar10 = features_cifar10[cifar_labels == class_index]

        # calculate metric
        fid_score_subset = util.fid(extracted_features, selected_features_cifar10)
        fids.append(fid_score_subset)
        print(
            f"The Fréchet Inception Distance (FID) between C10-'{class_name}' and 'tiny-imagenet-whole-dataset' features is: {fid_score_subset}")
    print(f'fids mean={np.mean(fids)} max={np.max(fids)} class_coverage={np.mean(fids) / np.max(fids)}')


def tiny_c10_dinov3_distance(trainset, dino_features_cifar10, tiny_dino_extracted_features):
    import torch
    import numpy as np

    # distances between c10-classes and tiny-imagenet-whole-dataset using DINO embeddings
    fids_dino = []
    cifar10_labels = torch.tensor(trainset.targets)
    
    for class_name in trainset.classes:
        # create a tensor with the image DINO features for selected class
        class_index = trainset.classes.index(class_name)
        selected_dino_c10 = dino_features_cifar10[cifar10_labels == class_index]

        # calculate metric
        fid_score = util.fid(tiny_dino_extracted_features, selected_dino_c10)
        fids_dino.append(fid_score)
        print(
            f"The Frechet DINO Distance (FDD) between C10-'{class_name}' and 'tiny-imagenet-whole-dataset' using DINO is: {fid_score}")
            
    print(
        f'\nDINO FDDs: mean={np.mean(fids_dino):.4f} max={np.max(fids_dino):.4f} coverage={np.mean(fids_dino) / np.max(fids_dino):.4f}')


def c10_tiny_inception_distance(features_cifar10, extracted_features, tiny_metadata):
    # create code to do the other way around: classes form tiny against whole-c10
    # distances betweeen tiny-imagenet-classes and c10-whole-dataset
    fids = []
    tiny_imagenet_cls_names, tiny_imagenet_cls_index, tiny_imagenet_cls_idx_to_name_dict, tiny_imagenet_cls_index_to_idx_dict = tiny_metadata
    for class_index in np.unique(tiny_imagenet_cls_index):
        # create a tensor with the image inception features for selected class
        # cifar_labels = torch.tensor(trainset.targets)
        # class_index = trainset.classes.index(class_name)
        selected_features_tiny = extracted_features[tiny_imagenet_cls_index == class_index]
        class_name = tiny_imagenet_cls_idx_to_name_dict[tiny_imagenet_cls_index_to_idx_dict[class_index]]
        # calculate metric
        fid_score_subset = util.fid(features_cifar10, selected_features_tiny)
        fids.append(fid_score_subset)
        print(
            f"The Fréchet Inception Distance (FID) between Tiny-'{class_name}' and 'C10-whole-dataset' features is: {fid_score_subset}")
    print(f'fids mean={np.mean(fids)} max={np.max(fids)} class_coverage={np.mean(fids) / np.max(fids)}')


def c10_tiny_dinov3_distance(dino_features_cifar10, tiny_dino_extracted_features, tiny_metadata):
    import torch
    import numpy as np

    # distances between tiny-imagenet-classes and c10-whole-dataset using DINO embeddings
    fids_dino = []
    tiny_imagenet_cls_names, tiny_imagenet_cls_index, tiny_imagenet_cls_idx_to_name_dict, tiny_imagenet_cls_index_to_idx_dict = tiny_metadata
    for class_index in np.unique(tiny_imagenet_cls_index):
        # create a tensor with the image DINO features for selected class
        selected_dino_tiny = tiny_dino_extracted_features[tiny_imagenet_cls_index == class_index]
        class_name = tiny_imagenet_cls_idx_to_name_dict[tiny_imagenet_cls_index_to_idx_dict[class_index]]
        # calculate metric
        fid_score = util.fid(dino_features_cifar10, selected_dino_tiny)
        fids_dino.append(fid_score)
        print(
            f"The Frechet DINO Distance (FDD) between Tiny-'{class_name}' and 'C10-whole-dataset' using DINO is: {fid_score}")
    print(
        f'\nDINO FDDs: mean={np.mean(fids_dino):.4f} max={np.max(fids_dino):.4f} coverage={np.mean(fids_dino) / np.max(fids_dino):.4f}')


def tiny_c100_inception_distance(trainset100, features_cifar100, extracted_features):
    # distances betweeen c100-classes and tiny-imagenet-whole-dataset
    fids = []
    for class_name100 in trainset100.classes:
        # create a tensor with the image inception features for selected class
        cifar_labels100 = torch.tensor(trainset100.targets)
        class_index100 = trainset100.classes.index(class_name100)
        selected_features_cifar100 = features_cifar100[cifar_labels100 == class_index100]

        # calculate metric
        fid_score_subset = util.fid(extracted_features, selected_features_cifar100)
        fids.append(fid_score_subset)
        print(
            f"The Fréchet Inception Distance (FID) between C100-'{class_name100}' and 'tiny-imagenet-whole-dataset' features is: {fid_score_subset}")
    print(f'fids mean={np.mean(fids)} max={np.max(fids)} class_coverage={np.mean(fids) / np.max(fids)}')


def tiny_c100_dinov3_distance(trainset100, dino_features_cifar100, tiny_dino_extracted_features):
    import torch
    import numpy as np

    # distances between c100-classes and tiny-imagenet-whole-dataset using DINO embeddings
    fids_dino = []
    cifar100_labels = torch.tensor(trainset100.targets)
    
    for class_name100 in trainset100.classes:
        # create a tensor with the image DINO features for selected class
        class_index100 = trainset100.classes.index(class_name100)
        selected_dino_c100 = dino_features_cifar100[cifar100_labels == class_index100]

        # calculate metric
        fid_score = util.fid(tiny_dino_extracted_features, selected_dino_c100)
        fids_dino.append(fid_score)
        print(
            f"The Frechet DINO Distance (FDD) between C100-'{class_name100}' and 'tiny-imagenet-whole-dataset' using DINO is: {fid_score}")
            
    print(
        f'\nDINO FDDs: mean={np.mean(fids_dino):.4f} max={np.max(fids_dino):.4f} coverage={np.mean(fids_dino) / np.max(fids_dino):.4f}')


def c100_tiny_inception_distance(features_cifar100, extracted_features, tiny_metadata):
    # create code to do the other way around: classes form tiny against whole-c10
    # distances betweeen tiny-imagenet-classes and c10-whole-dataset
    fids = []
    tiny_imagenet_cls_names, tiny_imagenet_cls_index, tiny_imagenet_cls_idx_to_name_dict, tiny_imagenet_cls_index_to_idx_dict = tiny_metadata
    for class_index in np.unique(tiny_imagenet_cls_index):
        # create a tensor with the image inception features for selected class
        # cifar_labels = torch.tensor(trainset.targets)
        # class_index = trainset.classes.index(class_name)
        selected_features_tiny = extracted_features[tiny_imagenet_cls_index == class_index]
        class_name = tiny_imagenet_cls_idx_to_name_dict[tiny_imagenet_cls_index_to_idx_dict[class_index]]
        # calculate metric
        fid_score_subset = util.fid(features_cifar100, selected_features_tiny)
        fids.append(fid_score_subset)
        print(
            f"The Fréchet Inception Distance (FID) between Tiny-'{class_name}' and 'C100-whole-dataset' features is: {fid_score_subset}")
    print(f'fids mean={np.mean(fids)} max={np.max(fids)} class_coverage={np.mean(fids) / np.max(fids)}')


def c100_tiny_dinov3_distance(dino_features_cifar100, tiny_dino_extracted_features, tiny_metadata):
    import torch
    import numpy as np

    # distances between tiny-imagenet-classes and c100-whole-dataset using DINO embeddings
    fids_dino = []
    tiny_imagenet_cls_names, tiny_imagenet_cls_index, tiny_imagenet_cls_idx_to_name_dict, tiny_imagenet_cls_index_to_idx_dict = tiny_metadata
    for class_index in np.unique(tiny_imagenet_cls_index):
        # create a tensor with the image DINO features for selected class
        selected_dino_tiny = tiny_dino_extracted_features[tiny_imagenet_cls_index == class_index]
        class_name = tiny_imagenet_cls_idx_to_name_dict[tiny_imagenet_cls_index_to_idx_dict[class_index]]
        # calculate metric
        fid_score = util.fid(dino_features_cifar100, selected_dino_tiny)
        fids_dino.append(fid_score)
        print(
            f"The Frechet DINO Distance (FDD) between Tiny-'{class_name}' and 'C100-whole-dataset' using DINO is: {fid_score}")
    print(
        f'\nDINO FDDs: mean={np.mean(fids_dino):.4f} max={np.max(fids_dino):.4f} coverage={np.mean(fids_dino) / np.max(fids_dino):.4f}')

def main():
    features_cifar10, features_cifar100, extracted_features = data.load_inception()
    dino_features_cifar10, dino_features_cifar100, tiny_dino_extracted_features = data.load_dinov3()
    trainset, trainset100 = data.load_cifar_pytorch_datasets()
    tiny_metadata = data.load_tinyimagenet_metadata()

    # c100_c10_inception_distance(trainset, trainset100, features_cifar10, features_cifar100)
    # c10_c100_inception_distance(trainset, trainset100, features_cifar10, features_cifar100)
    # c100_c10_dinov3_distance(trainset, trainset100, dino_features_cifar10, dino_features_cifar100)
    # c10_c100_dinov3_distances(trainset, trainset100, dino_features_cifar10, dino_features_cifar100)
    # tiny_c10_inception_distance(trainset, features_cifar10, extracted_features)
    # c10_tiny_inception_distance(features_cifar10, extracted_features, tiny_metadata)
    # tiny_c10_dinov3_distance(trainset, dino_features_cifar10, tiny_dino_extracted_features)
    c10_tiny_dinov3_distance(dino_features_cifar10, tiny_dino_extracted_features, tiny_metadata)
    # tiny_c100_inception_distance(trainset100, features_cifar100, extracted_features)
    # c100_tiny_inception_distance(features_cifar10, extracted_features, tiny_metadata)
    # tiny_c100_dinov3_distance(trainset100, dino_features_cifar100, tiny_dino_extracted_features)
    # c100_tiny_dinov3_distance(dino_features_cifar100, tiny_dino_extracted_features, tiny_metadata)

main()
