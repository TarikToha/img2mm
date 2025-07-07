import argparse
import os
import time

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from matplotlib import pyplot as plt

from pix2pix_turbo import Pix2Pix_Turbo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=True, help='path to the input image')
    parser.add_argument('--prompt', type=str, required=True, help='the prompt to be used')
    parser.add_argument('--model_name', type=str, default='', help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default='', help='path to a model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    args = parser.parse_args()

    # only one of model_name and model_path should be provided
    if args.model_name == '' != args.model_path == '':
        raise ValueError('Either model_name or model_path should be provided')

    os.makedirs(args.output_dir, exist_ok=True)

    # initialize the model
    model = Pix2Pix_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path)
    model.set_eval()

    # make sure that the input image is a multiple of 8
    input_image = Image.open(args.input_image)
    orig_size = input_image.size
    input_image = input_image.resize((256, 256), Image.LANCZOS)
    bname = os.path.basename(args.input_image)
    out_path = os.path.join(args.output_dir, bname)

    # translate the image
    with torch.no_grad():
        c_t = F.to_tensor(input_image).unsqueeze(0).cuda()

        start_time = time.time()
        output_image = model(c_t, args.prompt)
        print('done', time.time() - start_time)

        output_image = output_image[0].cpu() * 0.5 + 0.5
        output_image = output_image.numpy()
        output_image = np.transpose(output_image, (1, 2, 0))
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2GRAY)

    # save the output image
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.axis('off')
    ax.imshow(output_image)
    fig.savefig(f'{out_path}', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(out_path, output_image.shape)
