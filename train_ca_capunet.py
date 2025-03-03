import os, math
from datetime import datetime
import torch
from models.capunet import CAPUNet
from utils.Transform_self import SpatialTransform2D
from utils.dataloader import Dataset_CA_CAPUNet as TrainDataset
from utils.dataloader import Dataset_SPARC, get_det_label
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from utils.losses import tdl_ca, tdl_bce, DetLoss
from utils.utils import AverageMeter, LogWriter
import numpy as np
# from utils.sparc import get_sparc_path

def crt_file(path):
    os.makedirs(path, exist_ok=True)

class CAPUNetSeg(object):
    def __init__(self, args=None):
        super(CAPUNetSeg, self).__init__()

        self.fold = args.fold
        self.epoches = args.num_epoch
        self.save_epoch = args.save_epoch
        self.test_epoch_idx = args.test_epoch

        self.model_name = args.model_name
        self.det_w = args.det_w

        self.lr = args.lr
        self.bs = args.batch_size
        self.n_classes = 2
        
        """paths for training and testing image sequences with fixed length should be provided below"""
        # train_kSPARC = get_sparc_path('MS-CA-DET-LSTM', 'k3_tr_cv' + str(args.fold), 8, 4)        
        # test_kSPARC = get_sparc_path('MS-CA-LSTM', 'k3_val_cv' + str(args.fold), "all", 4)
        train_kSPARC = [
            [['slice_i', '...', 'slice_i+n'], ['...'], ['slice_i', '...', 'slice_i+n']],
            [['label_i', '...', 'label_i+n'], ['...'], ['label_i', '...', 'slabel_i+n']]
            ]
        test_kSPARC = [
            [['slice_i', '...', 'slice_i+n'], ['...'], ['slice_i', '...', 'slice_i+n']],
            [['label_i', '...', 'label_i+n'], ['...'], ['label_i', '...', 'slabel_i+n']]
            ]

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        self.checkpoint_dir = os.path.join(args.checkpoint_root, self.model_name+ '_' +timestamp)
        crt_file(self.checkpoint_dir)
        self.checkpoint_ = self.checkpoint_dir + "/fold_" + str(args.fold)
        crt_file(self.checkpoint_)

        # Data augmentation
        self.spatial_aug = SpatialTransform2D(
            in_sequence=True,
            do_rotation=True, angle_z=(-np.pi / 12, np.pi / 12), 
            do_scale=True, scale_x=(0.85, 1.15), scale_y=(0.85, 1.15),
            do_translate=False, do_shear=False, do_elastic_deform=False)

        # initialize model
        self.enc = CAPUNet(ch=8).cuda()
        self.enc.anchors = self.enc.anchors.cuda()
        
        self.opt_enc = torch.optim.Adam(self.enc.parameters(), lr=self.lr)
        self.scheduler_enc = StepLR(self.opt_enc, step_size=self.epoches//2, gamma=0.1)

        train_srs = train_kSPARC

        # initialize the dataloader
        trainsrs_dataset = TrainDataset(train_srs)
        self.dataloader_srstrain = DataLoader(trainsrs_dataset, batch_size=self.bs, shuffle=True, num_workers=4)
        self.iters = len(self.dataloader_srstrain)

        test_sparc_dataset = Dataset_SPARC(test_kSPARC)
        self.test_len_sparc = len(test_sparc_dataset)
        self.dataloader_test_sparc = DataLoader(test_sparc_dataset, batch_size=1, shuffle=False)

        # define loss
        self.L_seg = tdl_bce
        self.L_det = DetLoss(next(self.enc.parameters()).device, anchors=self.enc.anchors, strides=[2**(len(self.enc.ch_mult)-1)], a_box=0.5, a_obj=1)

        # define loss log
        self.L_seg_log = AverageMeter(name='L_Seg')
        self.L_det_log = AverageMeter(name='L_Det')
        # define test log
        self.L_seg_log_test_sparc = AverageMeter(name='L_Seg_test_sparc')
        self.dice_log_test_sparc = AverageMeter(name='DSC_test_sparc')
        self.L_seg_log_test_cain = AverageMeter(name='L_Seg_test_cain')
        self.dice_log_test_cain = AverageMeter(name='DSC_test_cain')

    def forward_enc(self):
        self.pred, self.pred_clm, self.pred_det, self.pred_det_clm = self.enc(self.srs_img)

    def compute_seg_loss(self):       
        ##seg loss
        self.seg_loss = (self.L_seg(self.pred, self.srs_label) + self.L_seg(self.pred_clm, self.srs_label)) / 2

    def compute_det_loss(self):
        self.det_loss = (self.L_det(self.pred_det, self.srs_label_det) + self.L_det(self.pred_det_clm, self.srs_label_det)) / 2
        self.det_loss *= self.det_w
 
    def train_iterator(self, img1, img1_label, img1_label_det):

        self.srs_img = img1
        self.srs_label = img1_label
        self.srs_label_det = img1_label_det
        ###Seg forward
        self.forward_enc()
        # self.forward_seg()
        self.compute_det_loss()
        self.compute_seg_loss()
        

        self.opt_enc.zero_grad()
        self.sen_loss = self.seg_loss + self.det_loss
        self.sen_loss.backward()
        self.opt_enc.step()

        self.L_seg_log.update(self.seg_loss.data, img1.size(0))
        self.L_det_log.update(self.det_loss[0].data, img1.size(0))

    def train_epoch(self, epoch):
        self.enc.train()
        srs_dataloader = iter(self.dataloader_srstrain)
        for i in range(self.iters):
            srsimg, srslabel = next(srs_dataloader)

            if torch.cuda.is_available():
                srsimg = srsimg.cuda()
                srslabel = srslabel.cuda()

            # Augment the source image and target image
            mat, code_spa = self.spatial_aug.rand_coords(srsimg.shape[-2:])

            srsimg = self.spatial_aug.augment_spatial(srsimg, mat, code_spa)
            srslabel = self.spatial_aug.augment_spatial(srslabel, mat, code_spa, mode="nearest").int()

            srs_label_det = get_det_label(srslabel)
            srslabel = srslabel[:, :, :2]
            self.srs_mask = (srslabel.transpose(0, 2)[1] > 0).detach()

            self.train_iterator(srsimg, srslabel, srs_label_det)

            res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, self.epoches),
                             'Iter: [%d/%d]' % (i + 1, self.iters),
                             self.L_seg_log.__str__(),
                             self.L_det_log.__str__()])
            print(res)
    
    def test_epoch(self, epoch):
        self.enc.eval()
        log_info = [epoch]
        # test sparc
        sparc_dataloader = iter(self.dataloader_test_sparc)
        for i in range(self.test_len_sparc):
            tarimg, tarlabel = next(sparc_dataloader)

            if torch.cuda.is_available():
                tarimg = tarimg.cuda()
                tarlabel = tarlabel.cuda()
            tarlabel = tarlabel[:, :, :2]

            ###验证损失
            with torch.no_grad():
                pred_mask_b, pred_mask_b_clm, pred_det, pred_det_clm = self.enc(tarimg)
                loss_seg = self.L_seg(pred_mask_b, tarlabel)
                loss_seg_clm = self.L_seg(pred_mask_b_clm, tarlabel)
                
            tarlab = torch.where(tarlabel > 0.5, 1, 0)
            tarseg = torch.where(pred_mask_b > 0.5, 1, 0)
            tarseg_clm = torch.where(pred_mask_b_clm > 0.5, 1, 0)

            tardice_all = (1 - tdl_ca(tarseg, tarlab)+1 - tdl_ca(tarseg_clm, tarlab)) / 2
            self.L_seg_log_test_sparc.update((loss_seg.data + loss_seg_clm.data)/2., 1)
            self.dice_log_test_sparc.update(tardice_all.data, 1)
        res = '\t'.join(['Test-SPARC Epoch: [%d/%d]' % (epoch + 1, self.epoches),
                            self.L_seg_log_test_sparc.__str__(),
                            self.dice_log_test_sparc.__str__()])
        print(res)
        log_info += [self.L_seg_log_test_sparc.avg.item(), self.dice_log_test_sparc.avg.item()]
              
        self.testwriter.writeLog(log_info)

    def checkpoint(self, epoch):
        torch.save(self.enc.state_dict(), '{0}/enc_epoch_{1}.pth'.format(self.checkpoint_, epoch))

    def load_model(self, path, epoch):
        print("loading model epoch ", str(epoch))
        self.seg.load_state_dict(torch.load('{0}/enc_epoch_{1}.pth'.format(path, epoch)),strict=True)

    def train(self):
        self.trainwriter = LogWriter(name=self.checkpoint_ + "/train_" + self.model_name,
                                     head=["epoch", 'loss_seg', 'loss_det']) 
        self.testwriter = LogWriter(name=self.checkpoint_ + "/test_" + self.model_name,
                                    head=["epoch", 'loss_seg', "dsc"]) 

        for epoch in range(self.epoches):
            self.L_seg_log.reset()
            self.L_det_log.reset()

            self.epoch = epoch
            self.train_epoch(epoch)
            self.scheduler_enc.step()
            log_info = [epoch, self.L_seg_log.avg.item(), self.L_det_log.avg.item()]
            
            self.trainwriter.writeLog(log_info)

            if (epoch+1) % self.save_epoch == 0:
                self.checkpoint(epoch)
                
            if (epoch+1) % self.test_epoch_idx == 0:
                self.L_seg_log_test_sparc.reset()
                self.dice_log_test_sparc.reset()
                self.test_epoch(epoch+1)

        self.checkpoint(self.epoches)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='UDA seg Training Function')

    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--save_epoch', type=int, default=5)
    parser.add_argument('--test_epoch', type=int, default=5)
    
    parser.add_argument('--model_name', default="CAPUNet_fold1")
    parser.add_argument('--det_w', type=float, default=0.1)

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=4)

    parser.add_argument('--checkpoint_root', default="./checkpoint/CAPUnet") 
    
    args = parser.parse_args()

    trainer = CAPUNetSeg(args = args)
    trainer.train()

