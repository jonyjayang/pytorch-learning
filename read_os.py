from PIL import Image
from  torch.utils.data import Dataset
import os


class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir =label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)



    def __getitem__(self, item):
        img_name = self.img_path[item]
        img_item_path =os.path.join(self.root_dir,self.label_dir,img_name)
        img =Image.open(img_item_path)
        label = self.label_dir
        return img,label
    def __len__(self):
        return len(self.img_path)




if __name__ == '__main__':
    root_dir = 'Kodak24gray512bmp'
    label_dir = 'img'
    ant_data = MyData(root_dir,label_dir)
    img,label = ant_data[0]
    print(img)
    img.show()
    print(ant_data)