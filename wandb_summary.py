import pandas as pd
import wandb

name = 'wall_kld.rgbd_azi.v7.1'

metrics = ['val/kl_div', 'val/ms_ssim', 'val/clean_fid', 'val/lpips']
api = wandb.Api()
run = api.runs(path='toha/pix2mm_turbo', filters={'config.output_dir': name})[0]
print(run.name)

history = run.scan_history(keys=['_step', *metrics])
history = pd.DataFrame(history)
for idx, metric in enumerate(metrics):
    row = history[history[f'{metric}'] == history[f'{metric}'].min()].to_numpy()[0]
    print(metric, '=>', row[idx + 1], '\t', int(row[0]))
