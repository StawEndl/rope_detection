# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
import numpy as np
import os
import PIL.Image as Image
from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel
import torchvision, cv2
from torchvision.transforms import Resize
import torch.nn.functional as F
import random
from skimage import morphology

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False,device='cuda:1'):
        '''
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

        self.transf = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        '''
        
        # self.binary_cross_entropy_with_logits = nn.functional.binary_cross_entropy_with_logits()
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss()
        # self.mask_loss_txt = open("mask_loss.txt", "w")
        # self.BCE_mask = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        # self.sigmoid = nn.Sigmoid()
        self.BCELoss = nn.BCELoss()
        #self.smooth_l1_loss = nn.smooth_l1_loss()


        self.sobel_h = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False)
        self.sobel_h.weight.data = torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]],
                                                device=device).float()
        self.sobel_h.requires_grad_(False)
        self.sobel_v = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False)
        self.sobel_v.weight.data = torch.tensor([[[[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]]],
                                                device=device).float()
        self.sobel_v.requires_grad_(False)

        self.device = device

        np.seterr(divide='ignore', invalid='ignore')

    def mask_iou(self,img_masks, pred_mask):
        '''
        Computes IoU between two masks
        Input: two 2D array mask
        '''
        img_masks = self.resize_img2mask(img_masks, pred_mask.shape)

        # print("img_masks!!!!!",img_masks_clone.shape, mask_res.shape)
        img_masks = img_masks.mean(dim=1, keepdim=True)
        img_masks[img_masks > 0.2] = 1.
        img_masks[img_masks <= 0.2] = 0.

        pred_mask = pred_mask.clone()
        pred_mask[pred_mask>0.5] = 1
        pred_mask[pred_mask <= 0.5] = 0

        Union = (pred_mask + img_masks) != 0
        Intersection = (pred_mask * img_masks) != 0
        # print(Intersection)
        # print(f"Intersection.sum(){Intersection.sum()},Union.sum(){Union.sum()}")
        res = Intersection.sum() / Union.sum()
        return res

    def _fast_hist(self, row_label, row_image, n_class):
        mask = (row_label >= 0) & (row_label < n_class)
        hist = np.bincount(
            n_class * row_label[mask].astype(int) +
            row_image[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        return hist


    def mask_miou(self,img_masks, pred_mask):
        '''
        Computes IoU between two masks
        Input: two 2D array mask
        '''
        img_masks = self.resize_img2mask(img_masks, pred_mask.shape).squeeze(1)

        # print("img_masks!!!!!",img_masks.shape, pred_mask.shape)
        # img_masks = img_masks.mean(dim=1)
        # img_masks[img_masks > 0.2] = 1.
        # img_masks[img_masks <= 0.2] = 0.

        # mask = (img_masks != 255)
        # mask_res = mask_res[mask]
        # img_masks = img_masks[mask]

        _, index = pred_mask.max(dim=1)
        # print(index)
        index = index.cpu().numpy()
        label = img_masks.cpu().numpy()
        num_classes = pred_mask.shape[1]
        # print(num_classes)
        hist = np.zeros((num_classes, num_classes))

        for single_image, single_label in zip(index, label):
            for row_image, row_label in zip(single_image, single_label):
                hist += self._fast_hist(row_label.flatten(), row_image.flatten(), num_classes)
        # print(hist)
        miou = (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        miou = np.nanmean(np.diag(hist)/miou)

        # miou = (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        # print(miou)
        # miou = np.nanmean(np.diag(hist) / miou)

        # miou = (np.diag(hist) / )#.mean()
        # miou = np.nanmean(miou)

        # print(miou,type(miou),miou.shape,miou.mean())

        return miou

    def iou_mean(self, pred, target):
        # n_classes ：the number of classes in your dataset,not including background
        # for mask and ground-truth label, not probability map
        n_classes = pred.shape[1]
        _, pred = pred.max(dim=1,keepdim=True)
        ious = []
        # iousSum = 0.
        # pred = torch.from_numpy(pred)
        # pred = pred.view(-1)
        # target = np.array(target)
        # target = torch.from_numpy(target)
        # target = target.view(-1)


        # Ignore IoU for background class ("0")
        for cls in range(n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
            pred_inds = pred == cls
            target_inds = target == cls
            intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
            union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
            if union == 0:
                # ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
                continue
            else:
                ious.append(float(intersection) / float(max(union, 1)))
                # iousSum += float(intersection) / float(max(union, 1))
        # print(ious)
        # return iousSum / n_classes
        return sum(ious)/len(ious)

    def resize_img2mask(self,img_masks,shape):
        # print(shape,img_masks.shape)
        _,_,h,w = shape
        return torch.nn.functional.interpolate(img_masks, (h,w), mode='bilinear', align_corners=False)

    def smoothL1_compute(self,img_masks_clone,mask_res):
        # print("mask_res[0].shape@@@@@@@@@",mask_res[0].shape)
        #w = mask_res[0].shape[-2:]
        #img_masks_clone = img_masks[0]
        #for i in range(1,len(img_masks)):
        #    img_masks_clone = torch.cat((img_masks_clone,img_masks[i].resize_(1,1,w,w)),0)

        img_masks_clone = self.resize_img2mask(img_masks_clone,mask_res.shape)
        
        #print("img_masks!!!!!",img_masks_clone.shape, mask_res.shape)
        img_masks_clone = img_masks_clone.mean(dim=1, keepdim=True)
        # img_masks_clone[img_masks_clone > 0.2] = 1.
        # img_masks_clone[img_masks_clone <= 0.2] = 0.
        #print("mask_res, img_masks_clone@#####",len(mask_res),mask_res[0].shape, img_masks_clone.shape)
        mask_loss = nn.functional.smooth_l1_loss(mask_res, img_masks_clone,reduction='mean')
        #mask_loss = self._smooth_l1_loss(mask_res, img_masks_clone)
        mask_losses = torch.zeros(1, device=self.device)+ mask_loss
        return mask_loss,mask_losses


    def sobel(self, img):
        # img = img.float()
        out_h = self.sobel_h(img)
        out_v = self.sobel_v(img)
        # out = abs(out_v)+abs(out_h)
        out = (out_v**2 + out_h**2)**0.5
        # cv2.imwrite(f'sobel/out_{str(i)}_img.jpg', out[0].squeeze(0).permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy())
        # cv2.imwrite(f'sobel/out_{str(i)}_mask.jpg', out[0].squeeze(0).permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy())
        return out


    def Sobel_Loss(self,img_masks, mask_res):
        # i = random.randint(0,100000)
        # img_masks_clone = torch.mean(img_masks, dim=1, keepdim=True)
        # img_masks = img_masks.unsqueeze(1)
        # mask_loss = 0.
        # print("?>>>>>",img_masks)
        # morphology.remove_small_objects(data, min_size=300, connectivity=1)
        sobel_img_masks = self.sobel(img_masks)

        # name = random.randint(0,100000)
        # cv2.imwrite(f"sobel/{name}_{clas}_img.jpg",img_masks[0].permute(1,2,0).repeat((1,1,3)).cpu().numpy())
        # cv2.imwrite(f"sobel/{name}_{clas}_sobel.jpg", sobel_img_masks[0].permute(1, 2, 0).repeat((1, 1, 3)).cpu().numpy())

        # inp = mask_res.max(dim=1, keepdim=True)[1].float()
        # print("::::::",mask_res)
        sobel_mask_res = self.sobel(mask_res)
        # cv2.imwrite(f'sobel/out_{str(i)}_img.jpg', (sobel_img_masks[0]*255.).permute(1, 2, 0).repeat(1, 1, 3).detach().cpu().numpy().astype("int"))
        # cv2.imwrite(f'sobel/out_{str(i)}_mask.jpg', (sobel_mask_res[0]*255.).permute(1, 2, 0).repeat(1, 1, 3).detach().cpu().numpy().astype("int"))
        # print(sobel_img_masks.unique() , sobel_mask_res.unique(),sobel_img_masks-sobel_mask_res)
        mask_loss = (abs(sobel_img_masks-sobel_mask_res)).mean()#.mean()

        return mask_loss #, torch.zeros(1, device=self.device)+ mask_loss


    def remove_small_objects(self, target, min_size=30, connectivity=1):
        masks = []
        for i in range(target.shape[0]):
            mask = target[i,0,:,:]

            mask = morphology.remove_small_objects(mask.cpu().numpy(), min_size=min_size, connectivity=connectivity)
            masks.append(torch.from_numpy(mask).unsqueeze(0).unsqueeze(0))

        return torch.cat(masks,0)



    def Sobel_Loss_multi(self, img_masks, mask_res):
        loss = []
        # name = random.randint(0, 100000)
        # cv2.imwrite(f"sobel/{name}_image.jpg", img_masks[0].permute(1, 2, 0).repeat((1, 1, 3)).cpu().numpy())
        for i in range(mask_res.shape[1]):
            img_masks1 = img_masks.clone()
            mask_res1 = mask_res.max(dim=1, keepdim=True)[1]

            target = img_masks1 == i
            # target = morphology.remove_small_objects(target.cpu().numpy(), min_size=30, connectivity=1)
            # target = self.remove_small_objects(target)
            img_masks1[target] = 255

            img_masks1[~target] = 0
            # img_masks1 += 1

            mask_res1[mask_res1 != i] = -1
            mask_res1[mask_res1 == i] = 254
            mask_res1 += 1
            # print(img_masks.unique(),"??????",img_masks1.unique())
            # print(mask_res.shape, "!!!!!!!", mask_res1.shape)
            loss.append(self.Sobel_Loss(img_masks1.float(), mask_res1.float()))
        # print(loss)
        loss = (sum(loss)/len(loss))

        return loss, torch.zeros(1, device=self.device) + loss

        
    def CEloss_compute(self,img_masks_clone,mask_res):
        #print("mask_res[0].shape@@@@@@@@@", mask_res[0].shape)
        #w = mask_res[0].shape[-1]
        #img_masks_clone = img_masks[0].resize_(1,1,w,w)
        #img_masks_clone = img_masks[0]
        #for i in range(1,len(img_masks)):
        #    img_masks_clone = torch.cat((img_masks_clone,img_masks[i].resize_(1,1,w,w)),0)

        # img_masks_clone = img_masks_clone.unsqueeze(1)
        # print(f"1111,{img_masks_clone.shape}")
        img_masks_clone = self.resize_img2mask(img_masks_clone, mask_res.shape).squeeze(1)
        # print(f"2222,{img_masks_clone.shape}")
        
        #print("img_masks!!!!!",img_masks_clone.shape, mask_res.shape)
        # img_masks_clone = torch.mean(img_masks_clone,dim=1).long()
        # img_masks_clone[img_masks_clone > 0.2] = 1.
        # img_masks_clone[img_masks_clone <= 0.2] = 0.
        #print("@@@@@@@",img_masks_clone.shape, mask_res.shape)
        mask_loss = F.cross_entropy(mask_res.float(), img_masks_clone.long(), ignore_index=255)
        mask_losses = torch.zeros(1, device=self.device)+ mask_loss
        return mask_loss,mask_losses

    def __call__(self, p, targets, mask_res=None, img_masks=None, epoch=None):  # predictions, targets, model
        device = targets.device
        # print("device", device)
        lcls, lbox, lobj = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        '''
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # print("tbox[i]:",i, tbox[i])
                # print("tbox[i]:", tbox[i].size())
                #print("$$$$$$box",pbox)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        
        '''
        bs = 0  # batch size
        # print("^^^^^^^^^^^^^^len(p[0])::", paths)
        if mask_res is not None:
            """
            img_masks = img_masks.mean(dim=1, keepdim=True)
            #print("@@@@@@@",img_masks.shape, mask_res.shape)
            mask_loss = self.BCEWithLogitsLoss(mask_res, img_masks)
            mask_losses = torch.zeros(1, device=device)+ mask_loss
            """

            img_masks = img_masks.mean(dim=1,keepdim=True)

            mask = (img_masks<0) | (img_masks>=mask_res.shape[1])
            # img_masks[mask] = 0

            img_masks[mask] = 255
            # img_masks = img_masks - 1
            # img_masks[img_masks == 254] = 255
            # mask = img_masks.mean(dim=1,keepdim=True) == 254
            # img_masks[mask.repeat(1,img_masks.shape[1],1,1)] = 0
            # mask_res[mask.repeat(1,mask_res.shape[1],1,1)] = 0

            # mask = (img_masks != 255)
            # mask_res = mask_res[mask]
            # img_masks = img_masks[mask]

            # print(img_masks.unique(),img_masks.max(),mask_res.shape)
            mask_loss,mask_losses = self.CEloss_compute(img_masks,mask_res)
            # mask_loss,mask_losses = self.smoothL1_compute(img_masks, mask_res)
            ious = torch.zeros(1, device=self.device)
            # print(img_masks.shape,mask_res.shape)
            # iou = self.mask_iou(img_masks, mask_res)

            iou = self.mask_miou(img_masks, mask_res)
            # iou = self.iou_mean(mask_res, img_masks)
            ious+=iou

            # print("mask_loss",iou)
            # print("miou",self.iou_mean(mask_res, img_masks))

            sobel_loss, sobel_losses = self.Sobel_Loss_multi(img_masks, mask_res)



            if epoch is not None and epoch>=100:
                mask_loss+=sobel_loss

                
        #     # mask_losses = torch.zeros(len(mask_res), device=device, requires_grad=True)
        #     mask_losses = []
        #
        #     for i in range(len(mask_res)):
        #         c = mask_res[i].size()[0]
        #         h = mask_res[i].size()[1]
        #         w = mask_res[i].size()[2]
        #         # print("mask_res shape!!",h,w,c)
        #         # continue
        #
        #         mask_img = img_masks[i].mean(dim=0, keepdim=True)
        #         # print("mask_img.shape@@@@@@@@",mask_img.shape)
        #         _, h_mask_img, w_mask_img = mask_img.shape
        #
        #         mask = mask_res[i].unsqueeze(0)
        #         num, mask_c, mask_h, mask_w = mask.shape
        #         torch_resize = Resize([h_mask_img, w_mask_img])
        #         mask = torch_resize(mask)
        #         # print(mask.shape)
        #
        #         mask = mask.squeeze(0)
        #         # print(mask.shape)
        #         # mask = np.reshape(mask, (h_mask_img, w_mask_img, mask_c))
        #         # print("mask_res[i].size() before: ", mask_res[i].size())
        #         # print("mask_res[i]:",mask_res[i])
        #         # print("mask_res[i].size() after: ",name, mask_img.size(), mask.size())
        #
        #         # mask_loss += torch.tensor([self.BCELoss(mask, mask_img)], device=device)
        #         mask_losses.append(self.BCEWithLogitsLoss(mask, mask_img))
        #     # mask_loss = mask_loss/len(paths)
        #     #print("………………………………………………"*20,lbox)
        #     mask_loss = torch.tensor(mask_losses, device=device, requires_grad=True)
        #     mask_losses = torch.tensor(mask_losses, device=device).mean(0,keepdim=True)
        #     #print(mask_losses)
        #     #print(lbox, lobj, lcls)
        #     #print(torch.cat((lbox, lobj, lcls)).detach())
        #     #return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls, mask_losses)).detach(), mask_loss.mean()
        # #return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()
            return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls, mask_losses, ious, sobel_losses)).detach(), mask_loss
        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()


    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            # print("i", i, "len(p)", len(p))
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
