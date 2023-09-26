import glob
import random
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from options import TrainOptions

def load_list(dataset_name, data_root):

    images = []
    labels = []

    img_root = data_root + dataset_name + '/train_A/'
    img_files = os.listdir(img_root)

    for img in img_files:

        images.append(img_root + img[:-4]+'.png')
        labels.append(img_root.replace('/train_A/', '/train_B/') + img[:-4]+'.png')
        

    img_root = data_root + 'DSC' + '/train_A/'
    img_files = os.listdir(img_root)

    for img in img_files:

        images.append(img_root + img[:-4]+'.jpg')
        labels.append(img_root.replace('/train_A/', '/train_B/') + img[:-4]+'.jpg')
    
    img_root = data_root + 'DSC2' + '/train_A/'
    img_files = os.listdir(img_root)

    for img in img_files:

        images.append(img_root + img[:-4]+'.png')
        labels.append(img_root.replace('/train_A/', '/train_B/') + img[:-4]+'.png')


    img_root = data_root + 'ISTD2/ISTD2' + '/train_A/'
    img_files = os.listdir(img_root)

    for img in img_files:

        images.append(img_root + img[:-4]+'.png')
        labels.append(img_root.replace('/train_A/', '/train_B/') + img[:-4]+'.png')

    # img_root = data_root + dataset_name + '/train_C/'
    # img_files = os.listdir(img_root)

    # for img in img_files:

    #     images.append(img_root + img[:-4]+'.png')
    #     labels.append(img_root.replace('/train_C/', '/train_D/') + 'shadow.png')

    return images, labels


def load_test_list(test_path, data_root):

    images = []


    # img_root = data_root + test_path + '/test_D/'

    # img_files = os.listdir(img_root)

    # for img in img_files:
    #     images.append(img_root + img[:-4] + '.png')
    
    # img_root = data_root + 'DSC2' + '/train_A/'
    # img_files = os.listdir(img_root)
    # for img in img_files:

    #     images.append(img_root + img[:-4]+'.png')
    
    
    img_root = data_root + 'test_new' + '/test_A/'
    # img_root = data_root + 'ISTD' + '/test/test_A/'
    # img_root = data_root + 'DSC2' + '/train_A/'

    img_files = os.listdir(img_root)

    for img in img_files:
        images.append(img_root + img[:-4] + '.png')

    return images


class ImageDataset(Dataset):
    def __init__(self, args, root, transforms_=None, mode='train'):
        if mode == 'train':
            self.image_path, self.label_path = load_list('ISTD/ISTD', './Data/')
        else:
            self.image_path = load_test_list('ISTD/test', './Data/')
        self.mode = mode
        self.transform = transforms.Compose(transforms_)
        ts = [transforms.Resize((args.img_height, args.img_width), Image.BICUBIC),
              transforms.ToTensor()]
        ts = transforms.Compose(ts)
        self.ts = ts
        self.args = args
        print(len(self.image_path))
        #input()

    def __getitem__(self, item):
        fn = self.image_path[item].split('/')

        filename = fn[-1]
        if self.mode == 'train':
            image = Image.open(self.image_path[item]).convert('RGB')
            label = Image.open(self.label_path[item]).convert('L')
                
            img_A = self.transform(image)
            img_B = self.ts(label)
        
            return {'A': img_A, 'B': img_B, 'C': item}
        else:
            image = Image.open(self.image_path[item]).convert('RGB')
            img_A = self.transform(image)

            return{'A': img_A, 'B': filename}
    # def __getitem__(self, index):
    #
    #     img = Image.open(self.files[index])
    #     w, h = img.size
    #
    #     if self.args.which_direction == 'AtoB':
    #         img_A = img.crop((0, 0, w/2, h))
    #         img_B = img.crop((w/2, 0, w, h))
    #     else:
    #         img_B = img.crop((0, 0, w/2, h))
    #         img_A = img.crop((w/2, 0, w, h))
    #
    #
    #     if np.random.random() < 0.5:
    #         img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], 'RGB')
    #         img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], 'RGB')
    #
    #     img_A = self.transform(img_A)
    #     img_B = self.transform(img_B)
    #
    #     return {'A': img_A, 'B': img_B}

    def __len__(self):
        return len(self.image_path)



def Get_dataloader(args):
    transforms_ = [ transforms.Resize((args.img_height, args.img_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

    train_dataloader = DataLoader(ImageDataset(args, "%s/%s" % (args.data_root,args.dataset_name), transforms_=transforms_,mode='train'),
                        batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)

    test_dataloader = DataLoader(ImageDataset(args, "%s/%s" % (args.data_root,args.dataset_name), transforms_=transforms_, mode='test'),
                            batch_size=1, shuffle=True, num_workers=1, drop_last=True)

    val_dataloader = DataLoader(ImageDataset(args, "%s/%s" % (args.data_root,args.dataset_name), transforms_=transforms_, mode='val'),
                            batch_size=10, shuffle=True, num_workers=0, drop_last=True)

    return train_dataloader, test_dataloader, val_dataloader