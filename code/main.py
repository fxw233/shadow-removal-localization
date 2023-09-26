import os.path as osp
from eval import Eval_thread
from dataloader import EvalDataset



pred_dir = 'pred/'
output_dir = './'
gt_dir = './Data/'

pred_dir= osp.join(pred_dir, 'ISTD_new')

gt_dir = osp.join(osp.join(gt_dir, 'ISTD/test/test_B/'))
# gt_dir = osp.join(osp.join(gt_dir, 'DSC2/train_B/'))

loader = EvalDataset(pred_dir, gt_dir)
thread = Eval_thread(loader, output_dir, cuda=True)
print(thread.run())
