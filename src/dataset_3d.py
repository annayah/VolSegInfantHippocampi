# Dependencies imports
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Callable, Any
import nibabel as nib

# MONAI imports
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    SpatialPadd,
    CenterSpatialCropd,
    # RandCropByPosNegLabeld,
    # RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    # RandRotated,
    # RandZoomd,
    # RandShiftIntensityd,
    # RandScaleIntensityd,
    # RandGaussianNoised,
    # RandGaussianSmoothd,
    NormalizeIntensityd,
    ToTensord,
    MapTransform,
)
from monai.data import CacheDataset, list_data_collate
from monai.config import KeysCollection
from sklearn.model_selection import KFold


class BinarizeLabeld(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        threshold: float = 0.5,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.threshold = threshold

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key in self.key_iterator(d):
            if not isinstance(d[key], torch.Tensor):
                d[key] = torch.as_tensor(d[key])
            dtype = d[key].dtype
            d[key] = (d[key] > self.threshold).to(dtype)
        return d


class RepeatChanneld(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        repeats: int = 3,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.repeats = repeats

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = d[key].repeat(self.repeats, 1, 1, 1)
        return d
    

# Dataset-specific statistics
DATASET_STATS = {
    'albert': {
        'intensity_range': (175, 666),
        'target_spacing': (0.8, 0.8, 0.8),
        'global_mean': 323.3083959,
        'global_std': 80.4516610,
    },
    'lisa': {
        'intensity_range': (2.19, 10.70),
        'target_spacing': (1.0, 1.0, 1.0),
        'global_mean': 4.7462048,
        'global_std': 1.6160563,
    },
}


# Transform pipelines
def get_train_transforms(
    rand_crop_size: Tuple[int, int, int] = (128, 128, 128),
    intensity_range: Tuple[float, float] = (-100, 400),
    global_mean: float = 100.0,
    global_std: float = 80.0,
    target_spacing: Optional[Tuple[float, float, float]] = None,
    use_flip_rot: bool = True,
) -> Compose:
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ]

    if target_spacing is not None:
        transforms += [
            Spacingd(
                keys=["image", "label"],
                pixdim=target_spacing,
                mode=("trilinear", "nearest"),
                align_corners=False,
            )
        ]

    transforms += [

        ScaleIntensityRanged(
            keys=["image"],
            a_min=intensity_range[0],
            a_max=intensity_range[1],
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        
        NormalizeIntensityd(
            keys=["image"],
            subtrahend=global_mean / (intensity_range[1] - intensity_range[0]),
            divisor=global_std / (intensity_range[1] - intensity_range[0]),
        ),

        CropForegroundd(
            keys=["image","label"], 
            source_key="image", 
            margin=10
        ),
        
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=rand_crop_size,
        ),

        CenterSpatialCropd(
            keys=["image", "label"],
            roi_size=rand_crop_size,
        )
    ]

    if use_flip_rot:
        transforms += [
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        ]

    transforms += [
        BinarizeLabeld(keys=["label"]),
        RepeatChanneld(keys=["image"], repeats=3),
        ToTensord(keys=["image", "label"]),
    ]
    
    return Compose(transforms)


def get_val_transforms(
    rand_crop_size: Tuple[int, int, int] = (128, 128, 128),
    intensity_range: Tuple[float, float] = (-100, 400),
    global_mean: float = 100.0,
    global_std: float = 80.0,
    target_spacing: Optional[Tuple[float, float, float]] = None
) -> Compose:
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ]

    if target_spacing is not None:
        transforms += [
            Spacingd(
                keys=["image", "label"],
                pixdim=target_spacing,
                mode=("trilinear", "nearest"),
                align_corners=False,
            )
        ]

    transforms += [
        ScaleIntensityRanged(
            keys=["image"],
            a_min=intensity_range[0],
            a_max=intensity_range[1],
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),

        NormalizeIntensityd(
            keys=["image"],
            subtrahend=global_mean / (intensity_range[1] - intensity_range[0]),
            divisor=global_std / (intensity_range[1] - intensity_range[0]),
        ),

        CropForegroundd(
            keys=["image","label"], 
            source_key="image", 
            margin=10
        ),

        SpatialPadd(
            keys=["image", "label"],
            spatial_size=rand_crop_size,
        ),

        CenterSpatialCropd(
        keys=["image", "label"],
        roi_size=rand_crop_size,
        ),

        BinarizeLabeld(keys=["label"]),
        RepeatChanneld(keys=["image"], repeats=3),
        ToTensord(keys=["image", "label"]),
    ]

    
    return Compose(transforms)


def get_test_transforms(
    intensity_range: Tuple[float, float] = (-100, 400),
    global_mean: float = 100.0,
    global_std: float = 80.0,
    target_spacing: Optional[Tuple[float, float, float]] = None
) -> Compose:
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ]

    if target_spacing is not None:
        transforms += [
            Spacingd(
                keys=["image", "label"],
                pixdim=target_spacing,
                mode=("trilinear", "nearest"),
                align_corners=False,
            )
        ]

    transforms += [

        ScaleIntensityRanged(
            keys=["image"],
            a_min=intensity_range[0],
            a_max=intensity_range[1],
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),

        NormalizeIntensityd(
            keys=["image"],
            subtrahend=global_mean / (intensity_range[1] - intensity_range[0]),
            divisor=global_std / (intensity_range[1] - intensity_range[0]),
        ),

        BinarizeLabeld(keys=["label"]),
        RepeatChanneld(keys=["image"], repeats=3),
        ToTensord(keys=["image", "label"]),
    ]
    
    return Compose(transforms)


# Dataset class
class HippocampusDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        dataset_name: str = 'albert',
        split: str = 'train',
        transform: Optional[Compose] = None,
        fold: int = 0,
        num_folds: int = 5,
    ):
        self.data_dir = data_dir
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.transform = transform
        self.fold = fold
        self.num_folds = num_folds
        
        self.stats = DATASET_STATS.get(self.dataset_name, DATASET_STATS['albert'])
        
        self.data_list = self._load_data_list()
        
    def _load_data_list(self) -> List[Dict[str, str]]:
        data_list = []
        
        image_dir = os.path.join(self.data_dir, self.dataset_name, 'images')
        label_dir = os.path.join(self.data_dir, self.dataset_name, 'labels')
        
        if not os.path.exists(image_dir):
            image_dir = os.path.join(self.data_dir, self.dataset_name)
            label_dir = os.path.join(self.data_dir, self.dataset_name)
        
        image_files = []
        for f in os.listdir(image_dir):
            if f.endswith('.nii.gz') or f.endswith('.nii'):
                if 'label' not in f.lower() and 'seg' not in f.lower() and 'mask' not in f.lower():
                    image_files.append(f)
        
        image_files.sort()
        
        for img_file in image_files:
            base_name = img_file.replace('.nii.gz', '').replace('.nii', '')
            label_candidates = [
                f"{base_name}_label.nii.gz",
                f"{base_name}_seg.nii.gz",
                f"{base_name}_mask.nii.gz",
                f"{base_name}_label.nii",
                f"{base_name}_seg.nii",
            ]
            
            label_file = None
            for candidate in label_candidates:
                if os.path.exists(os.path.join(label_dir, candidate)):
                    label_file = candidate
                    break
            
            if label_file is not None:
                data_list.append({
                    'image': os.path.join(image_dir, img_file),
                    'label': os.path.join(label_dir, label_file),
                    'case_id': base_name,
                })
        
        if len(data_list) > 0:
            data_list = self._apply_cv_split(data_list)
        
        return data_list
    
    def _apply_cv_split(self, data_list: List[Dict]) -> List[Dict]:
        if self.num_folds <= 1:
            return data_list
        
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        indices = list(range(len(data_list)))
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
            if fold_idx == self.fold:
                if self.split == 'train':
                    return [data_list[i] for i in train_idx]
                else:
                    return [data_list[i] for i in val_idx]
        
        return data_list
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Dict:
        data = self.data_list[idx]
        
        if self.transform is not None:
            data = self.transform(data)
            if isinstance(data, list):
                data = data[0]
        
        if "image_meta_dict" in data and "pixdim" in data["image_meta_dict"]:
            spacing = np.asarray(data["image_meta_dict"]["pixdim"])[1:4].astype(np.float32)
        elif self.stats.get("target_spacing") is not None:
            spacing = np.asarray(self.stats["target_spacing"], dtype=np.float32)
        else:
            img_nib = nib.load(self.data_list[idx]["image"])
            spacing = np.array(img_nib.header.get_zooms()[:3], dtype=np.float32)

        data["spacing"] = torch.tensor(spacing, dtype=torch.float32)  
        
        if "case_id" not in data and "case_id" in self.data_list[idx]:
            data["case_id"] = self.data_list[idx]["case_id"]
            
        if "image_path" not in data and "image" in self.data_list[idx]:
            data["image_path"] = self.data_list[idx]["image"]
        if "label_path" not in data and "label" in self.data_list[idx]:
            data["label_path"] = self.data_list[idx]["label"]

        preproc_affine = None
        img_obj = data.get("image", None)
        if img_obj is not None and hasattr(img_obj, "affine") and img_obj.affine is not None:
            preproc_affine = torch.as_tensor(img_obj.affine, dtype=torch.float32)
        elif "image_meta_dict" in data and "affine" in data["image_meta_dict"]:
            preproc_affine = torch.as_tensor(data["image_meta_dict"]["affine"], dtype=torch.float32)
        if preproc_affine is not None:
            data["preproc_affine"] = preproc_affine
            
        return data
    

def create_dataloaders(
    data_dir: str,
    dataset_name: str = 'albert',
    rand_crop_size: Tuple[int, int, int] = (128, 128, 128),
    batch_size: int = 1,
    num_workers: int = 4,
    fold: int = 0,
    num_folds: int = 5,
    cache_rate: float = 0.0,
) -> Tuple[DataLoader, DataLoader]:

    stats = DATASET_STATS.get(dataset_name.lower(), DATASET_STATS['albert'])

    train_transforms = get_train_transforms(
        rand_crop_size=rand_crop_size,
        intensity_range=stats['intensity_range'],
        global_mean=stats['global_mean'],
        global_std=stats['global_std'],
        target_spacing=stats['target_spacing'],
    )
    
    val_transforms = get_val_transforms(
        rand_crop_size=rand_crop_size,
        intensity_range=stats['intensity_range'],
        global_mean=stats['global_mean'],
        global_std=stats['global_std'],
        target_spacing=stats['target_spacing'],
    )
    
    train_dataset = HippocampusDataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        split='train',
        transform=train_transforms,
        fold=fold,
        num_folds=num_folds,
    )
    
    val_dataset = HippocampusDataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        split='val',
        transform=val_transforms,
        fold=fold,
        num_folds=num_folds,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=True,
        drop_last=False,
    )
    
    return train_loader, val_loader


def create_test_dataloader(
    data_dir: str,
    dataset_name: str = 'albert',
    batch_size: int = 1,
    num_workers: int = 4,
) -> DataLoader:

    stats = DATASET_STATS.get(dataset_name.lower(), DATASET_STATS['albert'])

    test_transforms = get_test_transforms(
        intensity_range=stats['intensity_range'],
        global_mean=stats['global_mean'],
        global_std=stats['global_std'],
        target_spacing=stats['target_spacing'],
    )

    test_dataset = HippocampusDataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        split='test',
        transform=test_transforms,
        fold=0,
        num_folds=1,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=True,
        drop_last=False,
    )
    
    return test_loader