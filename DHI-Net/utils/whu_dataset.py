import level_inter as li

class whu_dataset(li.Dataset):
    def __init__(self,root,transform=None, crop_size=(512,512)):
        self.img_path = li.os.path.join(root,'img')
        self.lab_path = li.os.path.join(root,'lab')
        self.transform = transform
        self.crop_size = crop_size
        self.ids = [file.split('.')[0] for file in li.listdir(self.img_path)]
        if not self.ids:
            raise RuntimeError(f'No input file found in {self.img_path}, make sure you put your images there')
        li.logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __getitem__(self, item):
        filename = self.ids[item]
        img = li.cv2.imread(li.os.path.join(self.img_path, filename + '.png'))
        img = li.Image.fromarray(li.cv2.cvtColor(img, li.cv2.COLOR_BGR2RGB)).convert('RGB')
        lab = li.cv2.imread(li.os.path.join(self.lab_path, filename + '.png'))
        lab = li.Image.fromarray(li.cv2.cvtColor(lab, li.cv2.COLOR_BGR2GRAY)).convert('1')


        if self.transform is not None:
            img = self.transform(img)
            lab= self.transform(lab)
            # body_lab = self.transform(body_lab)
            # edge_lab = self.transform(edge_lab)
        return img, lab

    def __len__(self):
        return len(self.ids)

class test_dataset(li.Dataset):
    def __init__(self,root,transform=None, crop_size=(512,512)):
        self.img_path = li.os.path.join(root,'img')
        self.lab_path = li.os.path.join(root,'lab')
        self.transform = transform
        self.crop_size = crop_size
        self.ids = [file.split('.')[0] for file in li.listdir(self.img_path)]
        if not self.ids:
            raise RuntimeError(f'No input file found in {self.img_path}, make sure you put your images there')
        li.logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __getitem__(self, item):
        filename = self.ids[item]
        img = li.cv2.imread(li.os.path.join(self.img_path, filename + '.png'))
        img = li.Image.fromarray(li.cv2.cvtColor(img, li.cv2.COLOR_BGR2RGB)).convert('RGB')
        lab = li.cv2.imread(li.os.path.join(self.lab_path, filename + '.png'))
        lab = li.Image.fromarray(li.cv2.cvtColor(lab, li.cv2.COLOR_BGR2GRAY)).convert('1')


        if self.transform is not None:
            img = self.transform(img)
            lab= self.transform(lab)
            # body_lab = self.transform(body_lab)
            # edge_lab = self.transform(edge_lab)
        return img, lab

    def __len__(self):
        return len(self.ids)