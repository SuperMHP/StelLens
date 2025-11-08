import cv2
import os
import yaml
import astropy.io.fits as fits
import numpy as np
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import numbers
from tqdm import tqdm

import pdb
import cv2
from torch.autograd import Variable
from scipy.signal import stft
from torch.utils.data import DataLoader

from cores.model import PromptIR_CA
from cores.dataset import Deep_Space
from cores.vis import pred_vis

if __name__ == "__main__":
    cfg = yaml.load(open('config.yaml', 'r'), Loader=yaml.Loader)

    dataset = Deep_Space(
        cfg['input_dir'], 
        cfg['metadata'], 
        fast_reproject=bool(cfg['fast_reproject'])
        )
    
    infer_loader = DataLoader(
        dataset,   # Dataset对象
        batch_size=1,   
        shuffle=False,  
        num_workers=0  
    )
    print('3. Model initializing')
    model = PromptIR_CA(inp_channels = 10,
                out_channels = 2,
                dim = 48,
                num_blocks = [4,6,6,8],
                num_refinement_blocks = 4,
                heads = [1,2,4,8],
                ffn_expansion_factor = 2.66,
                bias = False,
                LayerNorm_type = 'WithBias',
                prompt= True,
                dual_pixel_task = True)

    model.load_state_dict(torch.load('./pretrained/last.pth', map_location=torch.device('cpu'))['model'])
    model.cuda()
    model.eval()

    print('   Model initialized')
    # 0. patch inference and stitch
    hann_1d = np.hanning(dataset.patch_size)
    window_2d = np.outer(hann_1d, hann_1d)

    all_results = np.zeros(dataset.project_shape)
    all_weights = np.zeros(dataset.project_shape)
    all_uncertaintys = np.zeros(dataset.project_shape)

    print('4. Restoring')
    with torch.no_grad():
        for i, item in tqdm(enumerate(infer_loader)):
            item = {k: v.to('cuda') for k, v in item.items()}

            results = model(item['input'], item)

            coordinate = item['coordinate'].squeeze(0)
            all_results[coordinate[0][0]:coordinate[1][0], coordinate[0][1]:coordinate[1][1]] += results['pred_img'].cpu().numpy().squeeze() * window_2d
            all_uncertaintys[coordinate[0][0]:coordinate[1][0], coordinate[0][1]:coordinate[1][1]] += torch.exp(results['pred_img_un']).cpu().numpy().squeeze() * window_2d
            all_weights[coordinate[0][0]:coordinate[1][0], coordinate[0][1]:coordinate[1][1]] += window_2d
    
    all_weights[all_weights == 0] = np.nan

    final_output = all_results / all_weights
    final_output_un = all_uncertaintys / all_weights
    
    final_output, final_output_un = dataset.decode_pred_array(final_output, final_output_un)
    print('   Restoration finished')
    print('5. Saving')
    # 1. Construction main image HDU
    hdu_primary = fits.PrimaryHDU(data=final_output, header=dataset.new_wcs.to_header())

    # Add some metadata to header
    hdu_primary.header['FILTER'] = (cfg['metadata']['filter'], 'Observation filter')
    hdu_primary.header['INSTRUME'] = (cfg['metadata']['instrument_name'], 'Observation instrument')
    hdu_primary.header['EXPTIME'] = (cfg['metadata']['exptime'], 'Exposure time (s)')
    hdu_primary.header['TELESCOP'] = ('StelLens', 'Telescope')
    hdu_primary.header['BUNIT'] = ('counts/sec', 'Data units (electrons per second)')

    # 2. Construction error map HDU
    hdu_error = fits.ImageHDU(data=final_output_un, name='ERROR')

    # 3. Combine and save
    hdul = fits.HDUList([hdu_primary, hdu_error])
    os.makedirs(cfg['save_dir'])
    hdul.writeto(os.path.join(cfg['save_dir'],'/output_StelLens.fits'), overwrite=True)
    print('   Saved')
    # 4. visualize
    pred_vis(final_output, savefir=cfg['save_dir'], mode=cfg['vis']['mode'])