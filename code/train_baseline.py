import argparse
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from evaluate import *

parser = argparse.ArgumentParser()
parser.add_argument("--nepoch", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--lr", type=float, default=2e-4, help="adam: lr")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--root_path", type=str, default='/opt/data/private/data/Caer/Caer-S/train',
                    help="the path of data for train")
parser.add_argument("--save_model_path", type=str, default='./saved_model',
                    help="the path of saved model")
parser.add_argument("--checkpoint_interval", type=int, default=10,
                    help="the path of saved model")

opt = parser.parse_args()
print(opt)

# fix random seeds for reproducibility
# SEED = 123
# torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(SEED)

writer = SummaryWriter(log_dir='./log/baseline/')
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cuda = True if torch.cuda.is_available() else False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# loss function
# optimizer_G = torch.optim.Adam(Net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# model = CEN()

# root = '/Users/arthur/Documents/data/MinCaer/test'
train_data = DataLoader(
    EmotionDataset(opt.root_path, txt_path='./face_info.txt'),
    batch_size=opt.batch_size,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
    num_workers=8
)
vaild_data = DataLoader(
    Evalue_dataloader('./face_test_info.txt'),
    batch_size=1,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
    num_workers=8
)
log_file = open('./log_baseline.txt', mode='w')


def train(train_type='res'):
    os.system('rm -rf ./log/baseline/*')
    if train_type == 'cet':
        model = CEN()
    else:
        model = ResNet()
    criterion = torch.nn.BCEWithLogitsLoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=0,amsgrad=True)
    nepoch = opt.nepoch
    for epoch in range(nepoch):
        #model.train()
        count = 0
        total_count = 0
        for i, batch in enumerate(train_data):
            input_face = batch['face']
            if train_type == 'cet':
                input_img = batch['img']
            input_label = batch['label']
            label_temp = input_label
            input_label = F.one_hot(input_label, 7)
            input_label = input_label.type(Tensor)
            # put data into GPU
            if torch.cuda.is_available():
                input_face = input_face.cuda(non_blocking=True)
                if train_type == 'cet':
                    input_img = input_img.cuda(non_blocking=True)
                input_label = input_label.cuda(non_blocking=True)
            if train_type == 'cet':
                res = model(input_img, input_face)
            else:
                res = model(input_face)
            # print(res)
            # print(input_label)
            loss = criterion(res, input_label)
            optim.zero_grad()
            # loss.requires_grad = True
            loss.backward(retain_graph=True)
            optim.step()
            # size of res is [8 x 6] ->[bs x category_num]
            temp_count = 0
            for xx in range(len(res)):
                if int(res[xx].argmax().item()) == int(label_temp[xx].item()):
                    count += 1
                    temp_count += 1
            total_count += len(label_temp)
            batch_acc = temp_count / len(label_temp)
            total_acc = count / total_count
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] [example ~ argamx_index:%d label:%d][batch_acc:%f total_acc:%f]\n"
                % (
                    epoch + 1,
                    nepoch,
                    i,
                    len(train_data),
                    loss.item(),
                    int(res[0].argmax().item()),
                    int(label_temp[0]),
                    batch_acc,
                    total_acc
                )
            )
        print('for test ......')
        vaild_count = 0
        vaild_distribution = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0}
        #model.eval()
        for index, e in enumerate(vaild_data):
            vaild_img = e['img']
            vaild_face = e['face']
            vaild_label = e['label']
            if torch.cuda.is_available():
                vaild_face = vaild_face.cuda(non_blocking=True)
                if train_type == 'cet':
                    vaild_img = vaild_img.cuda(non_blocking=True)
                vaild_label = vaild_label.cuda(non_blocking=True)
            if train_type == 'cet':
                #with torch.no_grad():
                out = model(vaild_img, vaild_face)
            else:
                #with torch.no_grad():
                out = model(vaild_face)
            for xx in range(len(out)):
                if int(out[xx].argmax().item()) == int(vaild_label[xx].item()):
                    vaild_count += 1
                vaild_distribution[str(out[xx].argmax().item())] += 1
        vaild_acc = vaild_count / vaild_data.__len__()
        writer.add_scalar('loss', loss, epoch)
        writer.add_scalar('acc', total_acc, epoch)
        writer.add_scalar('_test_acc:', vaild_acc, epoch)
        log_file.write('epoch:%d ,train_acc:%f , vaild_acc:%f ,pred_distribution %s\n' % (
            epoch, total_acc, vaild_acc, str(vaild_distribution)))
        if opt.checkpoint_interval != -1 and (epoch + 1) % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(model.state_dict(), "%s/FER_baseline_%d.pth" % (opt.save_model_path, epoch + 1))
    writer.close()
    log_file.close()

if __name__ == '__main__':
    train()
