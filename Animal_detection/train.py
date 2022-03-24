'''
    Train ResNet50_vd_ssld with Animals

    author: guopingpan
    email: 731061720@qq.com
            or panguoping02@gmail.com

'''

import matplotlib
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx
from paddlex.cls import transforms

train_transforms = transforms.Compose([
    transforms.RandomCrop(crop_size=224),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize()
])
eval_transforms = transforms.Compose([
    transforms.ResizeByShort(short_size=256),
    transforms.CenterCrop(crop_size=224),
    transforms.Normalize()
])

train_dataset = pdx.datasets.ImageNet(
    data_dir='Animals-10',
    file_list='Animals-10/train_list.txt',
    label_list='Animals-10/labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.ImageNet(
    data_dir='Animals-10',
    file_list='Animals-10/val_list.txt',
    label_list='Animals-10/labels.txt',
    transforms=eval_transforms)


num_classes = len(train_dataset.labels)
model = pdx.cls.ResNet50_vd_ssld(num_classes=num_classes)
model.train(num_epochs = 10,
            save_interval_epochs = 2,
            train_dataset = train_dataset,
            train_batch_size = 256,
            eval_dataset = eval_dataset,
            learning_rate = 0.025,
            warmup_steps = 32,
            warmup_start_lr = 0.0001,
            lr_decay_epochs=[2, 4, 8],
            lr_decay_gamma = 0.025,    
            save_dir='./output',
            use_vdl=True)