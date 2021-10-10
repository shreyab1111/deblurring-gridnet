import sys, os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from model import GridNet


def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        'pascal': pascalVOCLoader,
        'camvid': camvidLoader,
        'ade20k': ADE20KLoader,
        'mit_sceneparsing_benchmark': MITSceneParsingBenchmarkLoader,
        'cityscapes': cityscapesLoader,
        'nyuv2': NYUv2Loader,
        'sunrgbd': SUNRGBDLoader,
    }[name]


def get_data_path(name, config_file='config.json'):
    """get_data_path
    :param name:
    :param config_file:
    """
    data = json.load(open(config_file))
    return data[name]['data_path']


import math
import numbers
import random
from PIL import Image, ImageOps

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask):
        img, mask = Image.fromarray(img, mode='RGB'), Image.fromarray(mask, mode='L')            
        assert img.size == mask.size
        for a in self.augmentations:
            img, mask = a(img, mask)
        return np.array(img), np.array(mask, dtype=np.uint8)


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), Image.BILINEAR), mask.resize((self.size, self.size),
                                                                                       Image.NEAREST)

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):
        assert img.size == mask.size

        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        return self.crop(*self.scale(img, mask))


class runningScore(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {'Overall Acc: \t': acc,
                'Mean Acc : \t': acc_cls,
                'FreqW Acc : \t': fwavacc,
                'Mean IoU : \t': mean_iu,}, cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt: # upsample labels
        target = target.unsequeeze(1)
        target = F.upsample(target, size=(h, w), mode='nearest')
        target = target.sequeeze(1)
    elif h < ht and w < wt: # upsample images
        input = F.upsample(input, size=(ht, wt), mode='bilinear')
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(-1, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, ignore_index=250,
                      weight=weight, size_average=False)
    if size_average:
        if mask.is_cuda:
            loss /= mask.data.sum().type(torch.cuda.FloatTensor)
        else:
            loss /= mask.data.sum().type(torch.FloatTensor)
    return loss

def bootstrapped_cross_entropy2d(input, target, K, weight=None, size_average=True):
    
    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=None, size_average=True):
        n, c, h, w = input.size()
        log_p = F.log_softmax(input, dim=1)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
        log_p = log_p.view(-1, c)

        mask = target >= 0
        target = target[mask]
        loss = F.nll_loss(log_p, target, weight=weight, ignore_index=250,
                          reduce=False, size_average=False)
        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(input=torch.unsqueeze(input[i], 0),
                                           target=torch.unsqueeze(target[i], 0),
                                           K=K,
                                           weight=weight,
                                           size_average=size_average)
    return loss / float(batch_size)


def multi_scale_cross_entropy2d(input, target, weight=None, size_average=True, scale_weight=None):
    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight == None: # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp))

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(input=inp, target=target, weight=weight, size_average=size_average)

    return loss


import pdb

def train(args):

    data_aug = Compose([RandomRotate(10),
                        RandomHorizontallyFlip()])

    # Setup dataloader
    data_loader = get_loader(args.dataset)
    # data_path = get_data_path(args.dataset)
    data_path = args.dataset_dir
    t_loader = data_loader(data_path, is_transform = True,
                           img_size = (args.img_rows, args.img_cols),
                           augmentations = data_aug, img_norm = args.img_norm)
    v_loader = data_loader(data_path, is_transform = True,
                           split = 'validation', img_size = (args.img_rows, args.img_cols),
                           img_norm = args.img_norm)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size = args.batch_size, shuffle = True)
                                  # num_workers = args.num_workers, shuffle = True)
    valloader = data.DataLoader(v_loader, batch_size = args.batch_size)
                                # num_workers = args.num_workers)

    # Setup metrics
    running_metrics = runningScore(n_classes)

    # Setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom()
        loss_window = vis.line(X = torch.zeros((1,)).cpu(),
                               Y = torch.zeros((1)).cpu(),
                               opts = dict(xlabel = 'minibatches',
                                           ylabel = 'Loss',
                                           title = 'Training loss',
                                           legend = ['Loss']))

    gpu_id = int(input('input utilize gpu id (-1:cpu) : '))
    device = torch.device(f'cuda:{gpu_id}' if gpu_id >= 0 else 'cpu')

    # Setup model
    model = GridNet(inChannels = 3, outChannels = n_classes)
    model.to(device)
        
    if hasattr(model.modules, 'optimizer'):
        optimizer = model.modules.optimizer
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr = args.l_rate,
                                    momentum = 0.99, weight_decay = 5e-4)

    criterion = nn.NLLLoss()
    criterion.to(device)

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(f'Loading model and optimizer from checkpoint {args.resume}')
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print(f"Loaded checkpoint {args.resume}, epoch {checkpoint['epoch']}")
        else:
            print(f'No checkpoint found at {args.resume}')

    best_iou = -100.
    for epoch in range(args.n_epoch):
        print(f'epoch : {epoch} start')
        
        model.train()
        for i, (images, labels) in enumerate(trainloader):
            print(f'epoch : {epoch}, num_batch : {i} processing')
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            print('infering...')

            outputs = model(images)


            # pdb.set_trace()
            print('loss calculating')
            loss = criterion(outputs, labels)

            print('back propagating')
            loss.backward()
            print('parameter update')
            optimizer.step()

            if args.visdom:
                vis.line(X = torch.ones((1, 1)).cpu() * i,
                         Y = torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
                         win = loss_window,
                         update = 'append')

            if (i+1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{args.n_epoch}] Loss: {loss.data[0]}")

            model.eval()
            for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
                images_val, labels_val = images_val.to(device), labels_val.to(device)

                outputs = model(images_val)
                pred = outputs.data.max(1)[1].cpu().numpy()
                gt = labels_val.data.cpu().numpy()
                running_metrics.update(gt, pred)

                score, class_iou = running_metrics.get_scores()
                for k, v in score.items():
                    print(k, v)
                running_metrics.reset()

                if score['Mean IoU : \t'] >= best_iou:
                    best_iou = score['Mean IoU : \t']
                    state = {'epoch': epoch+1,
                             'model_state': model.state_dict(),
                             'optimizer_state': optimizer.state_dict()}
                    torch.save(state, f'{args.dataset}_best_model.pkl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--dataset', nargs='?', type=str, default='mit_sceneparsing_benchmark',
                        help = 'Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--dataset_dir', required = True, type = str,
                        help = 'Directory containing target dataset')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256,
                        help = 'Height of the input')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256,
                        help = 'Width of input')

    parser.add_argument('--img_norm', dest = 'img_norm', action = 'store_true',
                        help = 'Enable input images scales normalization [0, 1] | True by default')
    parser.add_argument('--no-img_norm', dest = 'img_norm', action = 'store_false',
                        help = 'Disable input images scales normalization [0, 1] | True by Default')
    parser.set_defaults(img_norm = True)
    
    parser.add_argument('--n_epoch', nargs = '?', type = int, default = 100,
                        help = '# of epochs')
    parser.add_argument('--batch_size', nargs = '?', type = int, default = 8,
                        help = 'Batch size')
    parser.add_argument('--l_rate', nargs = '?', type = float, default = 1e-5,
                        help = 'Learning rate [1-e5]')
    parser.add_argument('--resume', nargs = '?', type = str, default = None,
                        help = 'Path to previous saved model to restart from')

    parser.add_argument('--visdom', dest = 'visdom', action = 'store_true',
                        help = 'Enable visualizaion(s) on visdom | False by default')
    parser.add_argument('--no-visdom', dest = 'visdom', action = 'store_false',
                        help = 'Disable visualization(s) in visdom | False by default')
    parser.set_defaults(visdom = False)

    args = parser.parse_args()
    for key, item in vars(args).items():
        print(f'{key} : {item}')
    train(args)
