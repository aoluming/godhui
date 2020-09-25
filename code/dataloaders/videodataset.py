import os
from sklearn.model_selection import train_test_split
import torch
import random
import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision
from collections import Counter
from torchvision import transforms
from PIL import Image

# from mypath import Path

def ffmpeg_run(video_name, out_dir):
    com = "ffmpeg -i " + video_name + " -f image2 -vf fps=fps=10 -s 171x128 " + os.path.join(out_dir, "out%3d.png")
    print(com)
    os.system(com)


class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, root_dir='/opt/data/private/dbmeng/Data/Emotion/Caer/Caer',
                 output_dir='/opt/data/private/dbmeng/Data/Emotion/Caer/processCaer',
                 dataset='caer', split='train',
                 clip_len=16, preprocess=False):
        self.root_dir = root_dir
        self.output_dir = output_dir
        folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

        # if (not self.check_preprocess()) or preprocess:
        #   print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
        #  self.preprocess()
        if preprocess:
            self.preprocess()
        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            data_list = os.listdir(os.path.join(folder, label))
            # random.shuffle(data_list)
            count = 0
            for fname in data_list:
                # if count>=1000:
                #     break
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)
                count += 1
        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        # if self.split == 'train':
        #     self.fnames = self.fnames[:3000]
        #     labels = labels[:3000]
        # else:
        #     self.fnames = self.fnames[:300]
        #     labels = labels[:300]
        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)
        basedir = os.path.dirname(__file__)
        txt_path = os.path.join(basedir, dataset + '_labels.txt')
        with open(txt_path, 'w') as f:
            for id, label in enumerate(sorted(self.label2index)):
                f.writelines(str(id + 1) + ' ' + label + '\n')

        print('data distributition:', Counter(labels))
        # if dataset == "ucf101":
        #     if not os.path.exists('dataloaders/ucf_labels.txt'):
        #         with open('dataloaders/ucf_labels.txt', 'w') as f:
        #             for id, label in enumerate(sorted(self.label2index)):
        #                 f.writelines(str(id+1) + ' ' + label + '\n')
        #
        # elif dataset == 'hmdb51':
        #     if not os.path.exists('dataloaders/hmdb_labels.txt'):
        #         with open('dataloaders/hmdb_labels.txt', 'w') as f:
        #             for id, label in enumerate(sorted(self.label2index)):
        #                 f.writelines(str(id+1) + ' ' + label + '\n')
        # elif dataset == "per":
        #     if not os.path.exists('dataloaders/per_labels.txt'):
        #         with open('dataloaders/per_labels.txt', 'w') as f:
        #             for id, label in enumerate(sorted(self.label2index)):
        #                 f.writelines(str(id+1) + ' ' + label + '\n')

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        labels = np.array(self.label_array[index])
        buffer = self.load_frames(self.fnames[index], clip_len=self.clip_len, crop_size=self.crop_size)
        return buffer, torch.from_numpy(labels)
        # # Loading and preprocessing.
        # buffer = self.load_frames(self.fnames[index])
        # buffer = self.crop(buffer, self.clip_len, self.crop_size)
        # labels = np.array(self.label_array[index])
        #
        # if self.split == 'test':
        #     # Perform data augmentation
        #     buffer = self.randomflip(buffer)
        # # buffer = self.normalize(buffer)
        # # buffer = self.to_tensor(buffer)
        # buffer = self.transform(buffer)
        # print(buffer.shape)
        # return torch.from_numpy(buffer), torch.from_numpy(labels)

    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True

    def preprocess(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            os.mkdir(os.path.join(self.output_dir, 'validation'))
            os.mkdir(os.path.join(self.output_dir, 'test'))

        # split the data
        # the root_dir struct is [train,test,vaild]
        for train_test_vaild in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, train_test_vaild)
            category_emtion = [name for name in os.listdir(file_path)]
            for category in tqdm(category_emtion):
                train_dir = os.path.join(self.output_dir, 'train', category)
                val_dir = os.path.join(self.output_dir, 'validation', category)
                test_dir = os.path.join(self.output_dir, 'test', category)
                if not os.path.exists(train_dir):
                    os.mkdir(train_dir)
                if not os.path.exists(val_dir):
                    os.mkdir(val_dir)
                if not os.path.exists(test_dir):
                    os.mkdir(test_dir)
                video_path = os.path.join(file_path, category)
                if train_test_vaild == 'train':
                    train = os.listdir(video_path)
                    for video in train:
                        self.process_video(video, os.path.join('train', category), train_dir)
                elif train_test_vaild == 'test':
                    test = os.listdir(video_path)
                    for video in test:
                        self.process_video(video, os.path.join('test', category), test_dir)
                elif train_test_vaild == 'validation':
                    vaild = os.listdir(video_path)
                    for video in vaild:
                        self.process_video(video, os.path.join('validation', category), val_dir)

            # for video in train:
            #     self.process_video(video, file, train_dir)
            #
            # for video in val:
            #     self.process_video(video, file, val_dir)
            #
            # for video in test:
            #     self.process_video(video, file, test_dir)

        print('Preprocessing finished.')

    def process_video(self, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        # capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))
        ffmpeg_run(video_name=os.path.join(self.root_dir, action_name, video),
                   out_dir=os.path.join(save_dir, video_filename))
        # frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        # frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least 16 frames
        # EXTRACT_FREQUENCY = 4
        # if frame_count // EXTRACT_FREQUENCY <= 16:
        #     EXTRACT_FREQUENCY -= 1
        #     if frame_count // EXTRACT_FREQUENCY <= 16:
        #         EXTRACT_FREQUENCY -= 1
        #         if frame_count // EXTRACT_FREQUENCY <= 16:
        #             EXTRACT_FREQUENCY -= 1
        #
        # count = 0
        # i = 0
        # retaining = True
        #
        # while (count < frame_count and retaining):
        #     retaining, frame = capture.read()
        #     if frame is None:
        #         continue
        #
        #     if count % EXTRACT_FREQUENCY == 0:
        #         if (frame_height != self.resize_height) or (frame_width != self.resize_width):
        #             frame = cv2.resize(frame, (self.resize_width, self.resize_height))
        #         cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
        #         i += 1
        #     count += 1

        # Release the VideoCapture once it is no longer needed
        # capture.release()

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame
        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir, clip_len, crop_size):
        # the ori version from github
        # frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        # frame_count = len(frames)
        # buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        # for i, frame_name in enumerate(frames):
        #     frame = np.array(cv2.imread(frame_name)).astype(np.float64)
        #     buffer[i] = frame
        # return buffer
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        if len(frames) >= clip_len:
            if len(frames) == clip_len:
                load_frames = frames
            else:
                start = np.random.randint(len(frames) - clip_len)
                load_frames = frames[start:start + clip_len]
        else:
            copy_count = int(clip_len - len(frames))
            copy_frame = frames[-1]
            for i in range(copy_count):
                frames.append(copy_frame)
            load_frames = frames
        buffer = torch.empty(clip_len, 3, crop_size, crop_size)
        size_w = np.random.randint(self.resize_width - crop_size)
        size_h = np.random.randint(self.resize_height - crop_size)
        for i, frame_name in enumerate(load_frames):
            img = Image.open(frame_name)
            # img.save('ori_'+str(i) + '.png')
            # print(img.size)
            # img.save(str(i)+'.png')
            img = img.crop((size_w, size_h, size_w + crop_size, size_h + crop_size))
            img = self.transform(img)
            buffer[i] = img
        # print(buffer.size())
       # print(buffer.size())
        buffer = buffer.permute(1, 0, 2, 3)
        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len)

        # centy = buffer.shape[1] // 2
        # centx = buffer.shape[2] // 2
        # buffer = buffer[time_index:time_index + clip_len,
        #          centy-crop_size//2:centy-crop_size//2 + crop_size,
        #          centx-crop_size//2:centx-crop_size//2 + crop_size, :]

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    root_dir = '/opt/data/private/dbmeng/Data/Emotion/Caer/Caer'
    output_dir = '/opt/data/private/dbmeng/Data/Emotion/Caer/processCaer'
    train_data = VideoDataset(dataset='caer', root_dir=root_dir, output_dir=output_dir,
                              split='train', clip_len=16, preprocess=False)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)
    data = next(iter(train_loader))
    print(data[0].size())

    # ffmpeg_run('1070.avi','data')
    # for i, sample in enumerate(train_loader):
    #     inputs = sample[0]
    #     labels = sample[1]
    #     print(inputs.size())
    #     print(labels)
    #     img=inputs.permute(0,2,1,3,4)
    #
    #     t = torchvision.transforms.ToPILImage()
    #     for i in range(len(img[0])):
    #         x=img[0][i]
    #         x=t(x)
    #         basedir = os.path.dirname(__file__)
    #         print(basedir)
    #         x.save(os.path.join(basedir,str(i)+'_.png'))
    #    # print(img.size())
    #     break
