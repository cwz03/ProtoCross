# -*- coding: utf-8 -*-
import argparse
import os
import sys
from typing import Any
import json

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from mydataset import Transforms1, MyDataSetMriPetClinical, Transforms2
from model.mynet import create_vit_backbone, MriClassifier, PetClassifier
from kmeans_prototype_manager import KMeansPrototypeManager

from utils.utils import get_subjects_labels, set_seed
from utils.utils import cosine_similarity
from loss.MetaWeightContrastiveLoss import WeightSupConLoss
from loss.loss import calculate_my_loss
from loss.subtype_aware_loss import SubtypeAwareProtoLoss, SubtypeAwareMWCL


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', type=str, default='ADNI-UniCross+')
    parser.add_argument('--class_names', type=str, default='CN,AD',
                        choices=['CN,AD', 'sMCI,pMCI'], help='names of the classes.')

    parser.add_argument('--backbone', type=str, default='vit',
                        choices=['vit'], help='names of the backbone.')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for computation, e.g., "cpu", "cuda:0"')

    parser.add_argument('--optim_type', type=str, default='SGD', choices=['SGD'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=50)

    parser.add_argument('--use_Shared_Classifier', type=bool, default=True,
                        help='Shared_Classifier')
    parser.add_argument('--use_WeightSupConLoss', type=bool, default=True,
                        help='Weight supervised contrastive (SupCon) loss')

    parser.add_argument('--temperature', type=float, default=0.07)

    parser.add_argument('--num_subtypes', type=int, default=3, help='Number of AD subtypes')
    parser.add_argument('--lambda_mwcl', type=float, default=1.0, help='Weight for subtype-aware MWCL loss')
    parser.add_argument('--lambda_proto', type=float, default=1.0, help='Weight for subtype-aware proto loss')
    parser.add_argument('--proto_update_interval', type=int, default=5, help='Update prototypes every N epochs')
    parser.add_argument('--start_mwcl_epoch', type=int, default=10, help='Start MWCL or MAPL loss from this epoch')

    return parser.parse_args()


def save_prototypes(prototype_manager, filepath):
    prototypes_data = {
        'mri': {},
        'pet': {},
        'cn_prototypes': {},
        'subtype_mapping_mri': {},
        'subtype_mapping_pet': {},
        'current_stage': int(prototype_manager.current_stage)
    }

    # save subtype prototype
    for modality in ['mri', 'pet']:
        for subtype in range(prototype_manager.num_subtypes):
            if prototype_manager.prototypes[modality][subtype] is not None:
                proto_tensor = prototype_manager.prototypes[modality][subtype].cpu()
                proto_list = [float(x) for x in proto_tensor.tolist()]
                prototypes_data[modality][str(subtype)] = proto_list

    # CN
    for modality in ['mri', 'pet']:
        if prototype_manager.cn_prototypes[modality] is not None:
            proto_tensor = prototype_manager.cn_prototypes[modality].cpu()
            proto_list = [float(x) for x in proto_tensor.tolist()]
            prototypes_data['cn_prototypes'][modality] = proto_list

    for subject_id, subtype in prototype_manager.subtype_mapping_mri.items():
        prototypes_data['subtype_mapping_mri'][subject_id] = int(subtype)

    for subject_id, subtype in prototype_manager.subtype_mapping_pet.items():
        prototypes_data['subtype_mapping_pet'][subject_id] = int(subtype)

    with open(filepath, 'w') as f:
        json.dump(prototypes_data, f, indent=2)


def load_prototypes(prototype_manager, filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            prototypes_data = json.load(f)

        # AD
        for modality in ['mri', 'pet']:
            for subtype_str in prototypes_data[modality]:
                subtype = int(subtype_str)
                prototype_vector = torch.tensor(prototypes_data[modality][subtype_str])
                prototype_manager.prototypes[modality][subtype] = prototype_vector

        # CN
        for modality in ['mri', 'pet']:
            if modality in prototypes_data['cn_prototypes']:
                prototype_manager.cn_prototypes[modality] = torch.tensor(prototypes_data['cn_prototypes'][modality])

        prototype_manager.subtype_mapping_mri = prototypes_data['subtype_mapping_mri']
        prototype_manager.subtype_mapping_pet = prototypes_data['subtype_mapping_pet']

        prototype_manager.current_stage = prototypes_data['current_stage']
        prototype_manager.initialized = True

        print(f"Loaded prototypes from {filepath}")


def save_models(fold, class_names, experiment_name, model_mri, model_pet, classifier,
                acc_mri, acc_pet, best_acc, best_accs, previous_filepaths):
    """
    Save model function
    Args:
        fold: Current fold number
        class_names: List of class names
        experiment_name: Name of experiment
        model_mri: MRI model
        model_pet: PET model
        classifier: Shared classifier
        acc_mri: MRI accuracy
        acc_pet: PET accuracy
        best_acc: Previous best accuracy
        best_accs: best_acc fold's mri acc and pet acc
        previous_filepaths: Dictionary containing paths of previous models
    Returns:
        current_acc: current best accuracy
        new_filepaths: Dictionary containing paths of new models
        best_accs: Dictionary containing best accuracies
    """
    current_acc = acc_mri + acc_pet
    if current_acc < best_acc:
        return best_acc, previous_filepaths, best_accs

    # make save dir
    save_dir = os.path.join('checkpoints', f'{class_names[0]}_{class_names[1]}', f'{experiment_name}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # make save path
    new_filepaths = {
        'mri': os.path.join(save_dir, f'fold-{fold}_model_mri.pth'),
        'pet': os.path.join(save_dir, f'fold-{fold}_model_pet.pth'),
    }
    if classifier is not None:
        new_filepaths['shared'] = os.path.join(save_dir, f'fold-{fold}_model_shared_head.pth')

    # deleted old path
    for model_type, old_path in previous_filepaths.items():
        if old_path and os.path.exists(old_path):
            os.remove(old_path)
            print(f"Deleted previous {model_type} model file: {old_path}")

    # save model
    torch.save(model_mri.state_dict(), new_filepaths['mri'])
    print(f"Saved new best mri model to: {new_filepaths['mri']}")

    torch.save(model_pet.state_dict(), new_filepaths['pet'])
    print(f"Saved new best pet model to: {new_filepaths['pet']}")

    if classifier is not None:
        torch.save(classifier.state_dict(), new_filepaths['shared'])
        print(f"Saved new best shared classifier to: {new_filepaths['shared']}")

    return current_acc, new_filepaths, {'mri': acc_mri, 'pet': acc_pet}


def collect_training_features(model_mri, model_pet, train_dataloader, device):
    model_mri.eval()
    model_pet.eval()

    all_mri_features = []
    all_pet_features = []
    all_labels = []
    all_subject_ids = []
    ad_indices = []

    with torch.no_grad():
        for batch_idx, (mri_data, pet_data, _, labels, subject_ids) in enumerate(train_dataloader):
            mri_data = mri_data.to(device)
            pet_data = pet_data.to(device)

            mri_feature, _ = model_mri(mri_data)
            pet_feature, _ = model_pet(pet_data)

            all_mri_features.append(mri_feature.cpu())
            all_pet_features.append(pet_feature.cpu())
            all_labels.append(labels.cpu())
            all_subject_ids.extend(subject_ids)

            batch_ad_mask = (labels == 1)
            batch_ad_indices = torch.where(batch_ad_mask)[0] + batch_idx * train_dataloader.batch_size
            ad_indices.extend(batch_ad_indices.tolist())

    all_mri_features = torch.cat(all_mri_features, dim=0)
    all_pet_features = torch.cat(all_pet_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_mri_features, all_pet_features, all_labels, all_subject_ids, ad_indices


def update_prototypes(prototype_manager, model_mri, model_pet, train_dataloader, device):
    print(f"🔄 Updating prototypes with current encoder features...")

    all_mri_features, all_pet_features, all_labels, all_subject_ids, ad_indices = collect_training_features(
        model_mri, model_pet, train_dataloader, device
    )

    prototype_manager.initialize_with_kmeans(
        all_mri_features, all_pet_features, all_labels, all_subject_ids, ad_indices
    )

    print(f"✅ Prototypes updated successfully!")


def forward_with_subtype_learning(args, criterion, prototype_manager, subtype_mwcl_criterion, subtype_proto_criterion,
                                  train_data, model_mri, model_pet, classifier, clinical_encoder, device,
                                  lambda_mwcl, lambda_proto, epoch):
    # get data
    mri_data, pet_data, clinical_data, labels, subject_ids = train_data
    mri_data = mri_data.to(device)
    pet_data = pet_data.to(device)
    clinical_data = clinical_data.to(device)
    labels = labels.to(device)

    # forward
    mri_feature, out_mri = model_mri(mri_data)
    pet_feature, out_pet = model_pet(pet_data)
    clinical_feature = clinical_encoder(clinical_data)

    shared_out_mri = classifier(mri_feature)
    shared_out_pet = classifier(pet_feature)

    # calculate loss
    loss_align_mri = criterion(shared_out_mri, labels)
    loss_align_pet = criterion(shared_out_pet, labels)
    loss_mri = criterion(out_mri, labels)
    loss_pet = criterion(out_pet, labels)

    loss_subtype_mwcl = torch.tensor(0.0, device=device)
    loss_subtype_proto = torch.tensor(0.0, device=device)

    if prototype_manager.initialized and epoch >= args.start_mwcl_epoch:
        with torch.no_grad():
            subtype_labels_mri = prototype_manager.get_batch_subtype_labels(subject_ids, labels, 'mri')
            subtype_labels_pet = prototype_manager.get_batch_subtype_labels(subject_ids, labels, 'pet')

        clinical_similarity = cosine_similarity(clinical_feature.detach())
        loss_subtype_mwcl = subtype_mwcl_criterion(
            mri_feature, pet_feature, labels, subtype_labels_mri, subtype_labels_pet, clinical_similarity
        )

        loss_subtype_proto_mri = subtype_proto_criterion(
            mri_feature,
            prototype_manager.prototypes['mri'],
            subtype_labels_mri,
            labels,
            prototype_manager.cn_prototypes['mri']
        )
        loss_subtype_proto_pet = subtype_proto_criterion(
            pet_feature,
            prototype_manager.prototypes['pet'],
            subtype_labels_pet,
            labels,
            prototype_manager.cn_prototypes['pet']
        )
        loss_subtype_proto = loss_subtype_proto_mri + loss_subtype_proto_pet

    total_loss = (loss_mri + loss_pet + loss_align_mri + loss_align_pet +
                  lambda_mwcl * loss_subtype_mwcl + lambda_proto * loss_subtype_proto)

    return total_loss, loss_mri, loss_pet, loss_align_mri, loss_align_pet, loss_subtype_mwcl, loss_subtype_proto


def forward_basic(args, criterion, weight_supcon_criterion, train_data, model_mri, model_pet, classifier,
                  clinical_encoder, device):
    # get data
    mri_data, pet_data, clinical_data, labels, subject_ids = train_data
    mri_data = mri_data.to(device)
    pet_data = pet_data.to(device)
    clinical_data = clinical_data.to(device)
    labels = labels.to(device)

    # forward
    mri_feature, out_mri = model_mri(mri_data)
    pet_feature, out_pet = model_pet(pet_data)
    clinical_feature = clinical_encoder(clinical_data)

    shared_out_mri = classifier(mri_feature)
    shared_out_pet = classifier(pet_feature)

    # calculate loss
    loss_align_mri = criterion(shared_out_mri, labels)
    loss_align_pet = criterion(shared_out_pet, labels)
    loss_mri = criterion(out_mri, labels)
    loss_pet = criterion(out_pet, labels)

    #similarity = cosine_similarity(clinical_feature)
    #loss_contrastive = calculate_my_loss(mri_feature, pet_feature, labels, weight_supcon_criterion, similarity)
    loss_contrastive = 0

    total_loss = loss_mri + loss_pet + loss_align_mri + loss_align_pet + loss_contrastive

    return total_loss, loss_mri, loss_pet, loss_align_mri, loss_align_pet, loss_contrastive


def train_and_test(args, train_dataloader, test_dataloader, model_mri, model_pet, classifier,
                   device, optimizer, lr_scheduler, fold, clinical_encoder):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    weight_supcon_criterion = WeightSupConLoss(temperature=args.temperature)

    prototype_manager = KMeansPrototypeManager(
        num_subtypes=args.num_subtypes,
        feature_dim=768,
        device=device
    )

    subtype_mwcl_criterion = SubtypeAwareMWCL(temperature=args.temperature)
    subtype_proto_criterion = SubtypeAwareProtoLoss(temperature=args.temperature)

    best_acc = 0.0
    best_accs = {'mri': 0.0, 'pet': 0.0}
    previous_filepaths = {'mri': None, 'pet': None, 'shared': None, 'prototypes': None}

    for epoch in range(args.epochs):
        ########################################################################
        if epoch < args.start_mwcl_epoch:  # 阶段1: 基础表征学习
            prototype_manager.set_stage(1)
            lambda_mwcl, lambda_proto = 0.0, 0.0
            use_subtype_forward = False

        elif epoch == args.start_mwcl_epoch:  # 阶段2: 初始亚型发现（K-means聚类）
            prototype_manager.set_stage(2)
            print(f"\n=== Fold {fold} Epoch {epoch + 1}: 开始亚型发现阶段 ===")

            # 收集训练集特征并进行K-means聚类
            all_mri_features, all_pet_features, all_labels, all_subject_ids, ad_indices = collect_training_features(
                model_mri, model_pet, train_dataloader, device
            )

            # 使用K-means初始化亚型
            prototype_manager.initialize_with_kmeans(
                all_mri_features, all_pet_features, all_labels, all_subject_ids, ad_indices
            )

            lambda_mwcl, lambda_proto = 0.0, 0.0
            use_subtype_forward = False

        else:  # 阶段3: 精细化学习
            prototype_manager.set_stage(3)

            # 每5个epoch更新一次原型
            if epoch > args.start_mwcl_epoch and (epoch - args.start_mwcl_epoch) % args.proto_update_interval == 0:
                update_prototypes(prototype_manager, model_mri, model_pet, train_dataloader, device)

            if epoch >= args.start_mwcl_epoch:
                progress = min(1.0, (epoch - args.start_mwcl_epoch) / 10)
                lambda_mwcl = args.lambda_mwcl * progress
                lambda_proto = args.lambda_proto * progress
                use_subtype_forward = True
            else:
                lambda_mwcl, lambda_proto = 0.0, 0.0
                use_subtype_forward = False

        ########################################################################
        # train
        model_mri.train()
        model_pet.train()
        classifier.train()
        clinical_encoder.train()

        train_loss = 0.0
        _loss_mri = 0.0
        _loss_pet = 0.0
        _loss_align_mri = 0.0
        _loss_align_pet = 0.0
        _loss_subtype_mwcl = 0.0
        _loss_subtype_proto = 0.0
        _loss_contrastive = 0.0

        train_bar = tqdm(train_dataloader, file=sys.stdout)
        for step, train_data in enumerate(train_bar):
            optimizer.zero_grad()

            if use_subtype_forward:
                total_loss, loss_mri, loss_pet, loss_align_mri, loss_align_pet, loss_subtype_mwcl, loss_subtype_proto = forward_with_subtype_learning(
                    args, criterion, prototype_manager, subtype_mwcl_criterion, subtype_proto_criterion,
                    train_data, model_mri, model_pet, classifier, clinical_encoder, device,
                    lambda_mwcl, lambda_proto, epoch
                )
                loss_contrastive = torch.tensor(0.0)
            else:
                total_loss, loss_mri, loss_pet, loss_align_mri, loss_align_pet, loss_contrastive = forward_basic(
                    args, criterion, weight_supcon_criterion,
                    train_data, model_mri, model_pet, classifier, clinical_encoder, device
                )
                loss_subtype_mwcl = torch.tensor(0.0)
                loss_subtype_proto = torch.tensor(0.0)

            total_loss.backward()
            optimizer.step()


            train_loss += total_loss.item()
            _loss_align_mri += loss_align_mri.item()
            _loss_align_pet += loss_align_pet.item()
            _loss_mri += loss_mri.item()
            _loss_pet += loss_pet.item()
            _loss_subtype_mwcl += loss_subtype_mwcl.item()
            _loss_subtype_proto += loss_subtype_proto.item()

            stage_desc = {
                1: "基础学习",
                2: "亚型发现",
                3: "精细化学习"
            }[prototype_manager.current_stage]

            train_bar.desc = (
                f"fold {fold}: train epoch[{epoch + 1}/{args.epochs}] "
                f"loss:{total_loss.item():.3f} stage:{stage_desc} "
                f"λ_mwcl:{lambda_mwcl:.2f} λ_proto:{lambda_proto:.2f}"
            )

        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f'fold {fold}: Current Learning Rate: {current_lr:.6f}')

        ########################################################################
        # test
        model_mri.eval()
        model_pet.eval()

        all_preds_mri = []
        all_preds_pet = []
        all_labels = []

        with torch.no_grad():
            for test_data in test_dataloader:
                mri_data, pet_data, _, labels, _ = test_data
                mri_data = mri_data.to(device)
                pet_data = pet_data.to(device)
                labels = labels.to(device)

                _, out_mri = model_mri(mri_data)
                _, out_pet = model_pet(pet_data)

                _, preds_mri = torch.max(out_mri, 1)
                _, preds_pet = torch.max(out_pet, 1)
                all_preds_mri.extend(preds_mri.cpu().numpy())
                all_preds_pet.extend(preds_pet.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        num_batches = len(train_dataloader)
        train_loss /= num_batches
        _loss_align_mri /= num_batches
        _loss_align_pet /= num_batches
        _loss_mri /= num_batches
        _loss_pet /= num_batches
        _loss_subtype_mwcl /= num_batches
        _loss_subtype_proto /= num_batches

        acc_mri = accuracy_score(all_labels, all_preds_mri)
        acc_pet = accuracy_score(all_labels, all_preds_pet)

        print(
            f'fold {fold}: Epoch {epoch + 1}/{args.epochs}, '
            f'Train Loss: {train_loss:.4f}, '
            f'MRI: {_loss_mri:.4f}+{_loss_align_mri:.4f}, '
            f'PET: {_loss_pet:.4f}+{_loss_align_pet:.4f}, '
            f'Subtype MWCL: {_loss_subtype_mwcl:.4f}, '
            f'Subtype Proto: {_loss_subtype_proto:.4f}, '
            f'MRI Acc: {acc_mri:.4f}, '
            f'PET Acc: {acc_pet:.4f}'
        )

        # save best model
        best_acc, previous_filepaths, best_accs = save_models(
            fold=fold,
            class_names=args.class_names.split(','),
            experiment_name=args.experiment_name,
            model_mri=model_mri,
            model_pet=model_pet,
            classifier=classifier,
            acc_mri=acc_mri,
            acc_pet=acc_pet,
            best_acc=best_acc,
            best_accs=best_accs,
            previous_filepaths=previous_filepaths
        )

        print('-------------------------------------------------------------------------------------------------------')

    print('Finished Training and validating')


def main():
    args = get_arguments()
    set_seed(args.seed)
    class_names = args.class_names.split(',')
    class_num = len(class_names)

    nw = args.num_workers # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    if 'cuda' in args.device and not torch.cuda.is_available():
        print("CUDA is not available on this machine. Switching to CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    ####################################################################################################################
    # Folder path
    if os.name == 'nt':  # Windows
        mri_path = r"D:\ADNI_PROCESSED_version1/ADNI_MRI_T1_LINEAR"
        pet_path = r"D:\ADNI_PROCESSED_version1/ADNI_PET_T1_LINEAR_SMOOTH"
        clinical_path = r"D:\ADNI_PROCESSED_version1/clinical_pt"
    elif os.name == 'posix':  # Linux
        mri_path = '/home/data/ADNI_PROCESSED_version1/ADNI_MRI_T1_LINEAR'
        pet_path = '/home/data/ADNI_PROCESSED_version1/ADNI_PET_T1_LINEAR_SMOOTH'
        clinical_path = '/home/data/ADNI_PROCESSED_version1/clinical_pt'
    else:
        raise ValueError("Unsupported operating system!")

    mri_img_name_list = os.listdir(pet_path)
    subject_list_file = 'Data/Group_Subject_MRI_PET.csv'
    df = pd.read_csv(subject_list_file)

    ####################################################################################################################
    # k-fold
    selected_columns = [col for col in df.columns if col in class_names]
    subjects, labels = get_subjects_labels(df, selected_columns)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    for fold, (train_index, test_index) in enumerate(skf.split(subjects, labels)):
        print(f"\n{'=' * 80}")
        print(f"START Fold {fold}")
        print(f"{'=' * 80}")

        ################################################################################################################
        # model
        if args.backbone == 'vit':
            mri_backbone = create_vit_backbone(pretrained=True)
            pet_backbone = create_vit_backbone(pretrained=True)

            model_mri = MriClassifier(mri_backbone, out_feature_dim=768, class_num=class_num).to(device)
            model_pet = PetClassifier(pet_backbone, out_feature_dim=768, class_num=class_num).to(device)
            classifier = nn.Linear(768, class_num).to(device)  # shared classifier
        else:
            raise ValueError("Unsupported backbone!")

        clinical_encoder = nn.Linear(4, 32).to(device)

        models = [model_mri, model_pet, clinical_encoder, classifier]
        parameters = [p for model in models for p in model.parameters()]

        # optimizer
        if args.optim_type == 'SGD':
            optimizer = optim.SGD(parameters,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  momentum=args.momentum)
        else:
            raise ValueError("Unsupported optimizer!")

        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=3, eta_min=0.00001)

        ################################################################################################################
        train_subjects = subjects[train_index]
        test_subjects = subjects[test_index]

        train_dataset = MyDataSetMriPetClinical(
            mri_dir_path=mri_path,
            pet_dir_path=pet_path,
            clinical_dir_path=clinical_path,
            img_name_list=mri_img_name_list,
            subject_list=train_subjects,
            transform=Transforms1,
            class_names=class_names
        )
        test_dataset = MyDataSetMriPetClinical(
            mri_dir_path=mri_path,
            pet_dir_path=pet_path,
            clinical_dir_path=clinical_path,
            img_name_list=mri_img_name_list,
            subject_list=test_subjects,
            transform=Transforms2,
            class_names=class_names
        )

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=nw, pin_memory=True, drop_last=False)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=nw, pin_memory=True, drop_last=False)

        print("fold {}: {} subjects for training, {} subjects for test.".format(fold, len(train_subjects),
                                                                                len(test_subjects)))

        # train and test
        train_and_test(args, train_dataloader, test_dataloader, model_mri, model_pet,classifier,
                       device, optimizer, lr_scheduler, fold, clinical_encoder)


if __name__ == '__main__':
    main()