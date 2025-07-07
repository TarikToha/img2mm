import argparse
import os
from threading import Thread

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from param import generate_wall_prompt
from pix2pix_turbo import Pix2Pix_Turbo
from weapon_gen_batch import save_thread


def run_wall_model(input1, input2):
    input1 = torch.cat(input1)
    with torch.no_grad():
        output = model(input1, input2)
        output = output * 0.5 + 0.5
        output *= 255
        output = torch.permute(output, (0, 2, 3, 1))
        output = output.cpu().numpy().astype(dtype=np.uint8)
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='path to a model state dict to be used')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='dir to the input image')
    parser.add_argument('--label_file', type=str, required=True,
                        help='path to the labels file')
    parser.add_argument('--image_type', type=str, required=True,
                        help='image type')
    parser.add_argument('--resolution', type=int, required=True,
                        help='resolution of the input image')
    args = parser.parse_args()

    output_dir = args.model_path.split('/')[0]
    output_dir = f'output/{output_dir}/'
    os.makedirs(output_dir, exist_ok=True)

    # initialize the model
    model = Pix2Pix_Turbo(pretrained_name='', pretrained_path=args.model_path)
    model.set_eval()

    image_transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution),
                          interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor()
    ])

    dataset = pd.read_csv(args.label_file)
    total_cases = dataset.shape[0]

    input_images, input_prompts, out_paths = [], [], []
    for case_idx, row in tqdm(dataset.iterrows(), total=total_cases, desc='Inference'):
        capture_id, target_dir = row['Capture_ID'], row['Target_Dir']
        wall_material = row['Dataset']
        start_frame, end_frame = row['Start_Frame'], row['End_Frame']

        prefix = f'{args.image_type}_{capture_id}'
        prompt = generate_wall_prompt(wall_material)

        for frame_id in range(start_frame, end_frame):
            image_in_file = f'{args.input_dir}{prefix}_{frame_id}.jpg'
            assert os.path.exists(image_in_file), f'{image_in_file} not found'

            input_image = Image.open(image_in_file)
            input_image = image_transform(input_image).unsqueeze(dim=0).cuda()
            input_images.append(input_image)

            input_prompts.append(prompt)
            out_paths.append(image_in_file)

        if len(input_prompts) > 32 or case_idx == total_cases - 1:
            output_images = run_wall_model(input_images, input_prompts)

            thread_list = []
            for idx, output_image in enumerate(output_images):
                file_name = os.path.basename(out_paths[idx])
                file_name = file_name.replace(args.image_type, 'azi_fft')
                out_path = f'{output_dir}{file_name}'
                # save_output_image(output_image, out_path)
                t = Thread(target=save_thread, args=(output_image, out_path))
                t.start()
                thread_list.append(t)

            for t in thread_list:
                t.join()

            input_images.clear()
            input_prompts.clear()
            out_paths.clear()
