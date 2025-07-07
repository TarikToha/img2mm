import glob
import os

import lpips
import numpy as np
import pandas as pd
import torch
from PIL import Image
from cleanfid.features import build_feature_extractor
from cleanfid.fid import get_folder_features, fid_from_feats
from piq import MultiScaleSSIMLoss, DISTS
from torch.nn import KLDivLoss
from torch.nn.functional import mse_loss
from torchvision import transforms
from tqdm import tqdm


def fn_transform(x):
    x_pil = Image.fromarray(x)
    out_pil = transforms.Resize(
        256, interpolation=transforms.InterpolationMode.LANCZOS
    )(x_pil)
    return np.array(out_pil)


assert torch.cuda.is_available(), 'cuda is not available'
soa_name = 'pix2pix'
root_dir = '/nas/longleaf/home/ttoha12/work/im2mm/'
gt_folder = 'output/azi_fft_test/no_weapon/'
output_dir = f'output/{soa_name}_test/no_weapon/'
ref_file = 'ref_stats.npy'

if soa_name == 'gen_rgbd':
    suffix = 'rgbd_azi_gen_256'

elif soa_name == 'gen_color':
    suffix = 'color_azi_gen_256'

elif soa_name == 'pix2pix':
    suffix = 'color_azi_gen_256'

elif soa_name == 'cvae':
    suffix = 'color_azi_gen_64'

else:
    print(f'{soa_name} is not defined')
    exit()

print(output_dir, suffix)

net_lpips = lpips.LPIPS(net='vgg').cuda()
ms_ssim_loss = MultiScaleSSIMLoss(data_range=1, reduction='none').cuda()
kl_loss = KLDivLoss(reduction='batchmean')
dists_loss = DISTS(reduction='none')

net_lpips.requires_grad_(False)

feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)

if not os.path.exists(ref_file):
    ref_stats = get_folder_features(
        gt_folder, model=feat_model, num_workers=0, num=None,
        shuffle=False, seed=0, batch_size=32, device=torch.device("cuda"),
        mode="clean", custom_image_tranform=fn_transform, description="", verbose=True
    )
    np.save(ref_file, ref_stats)
else:
    ref_stats = np.load(ref_file)
    print(f'{ref_file} loaded')

curr_stats = get_folder_features(
    output_dir, model=feat_model, num_workers=0, num=None,
    shuffle=False, seed=0, batch_size=32, device=torch.device("cuda"),
    mode="clean", custom_image_tranform=fn_transform, description="", verbose=True
)

fid_score = fid_from_feats(ref_stats, curr_stats)
print('fid', fid_score)

image_transform = transforms.Compose([
    transforms.Resize(
        (256, 256), interpolation=transforms.InterpolationMode.LANCZOS
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

gt_files = glob.glob(f'{root_dir}{gt_folder}*.jpg')
total_files = len(gt_files)
print('found {:,} images'.format(total_files))

v_l2, v_ms_ssim, v_kl_div = [], [], []
v_lpips, v_dists, v_clipsim = [], [], []
for idx, gt in tqdm(enumerate(gt_files), total=total_files, desc='metrics'):
    x_tgt = Image.open(gt)
    x_tgt = image_transform(x_tgt).unsqueeze(dim=0).cuda()

    file_name = os.path.basename(gt)
    file_name = file_name.split('_')
    pred = f'{output_dir}{file_name[0]}_{suffix}_{file_name[-2]}_{file_name[-1]}'
    assert os.path.exists(pred), 'prediction file does not exist'

    x_tgt_pred = Image.open(pred)
    x_tgt_pred = image_transform(x_tgt_pred).unsqueeze(dim=0).cuda()

    # compute the reconstruction losses
    val_l2 = mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean")
    val_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean()

    # [-1, 1] => [0, 1]
    x_tgt_pred_renorm = x_tgt_pred * 0.5 + 0.5
    x_tgt_renorm = x_tgt * 0.5 + 0.5
    val_ms_ssim = ms_ssim_loss(x_tgt_pred_renorm, x_tgt_renorm)
    val_dists = dists_loss(x_tgt_pred_renorm, x_tgt_renorm)

    x_tgt_pred_renorm = x_tgt_pred_renorm / x_tgt_pred_renorm.sum()
    x_tgt_renorm = x_tgt_renorm / x_tgt_renorm.sum()
    x_tgt_pred_renorm += 1e-10
    x_tgt_renorm += 1e-10
    val_kl_div = kl_loss(torch.log(x_tgt_pred_renorm), x_tgt_renorm)

    v_l2.append(val_l2.item())
    v_lpips.append(val_lpips.item())
    v_ms_ssim.append(val_ms_ssim.item())
    v_kl_div.append(val_kl_div.item())
    v_dists.append(val_dists.item())

results = {
    'soa_name': [soa_name],
    'l2': np.mean(v_l2).tolist(),
    'lpips': np.mean(v_lpips).tolist(),
    'ms_ssim': np.mean(v_ms_ssim).tolist(),
    'kl_div': np.mean(v_kl_div).tolist(),
    'dists': np.mean(v_dists).tolist(),
    'fid': [fid_score],
}
results = pd.DataFrame(results)
results.to_csv(f'{soa_name}_results.csv', index=False)
print(results)
