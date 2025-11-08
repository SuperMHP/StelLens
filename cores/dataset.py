import os
from tqdm import tqdm
import pdb
import cv2
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import torch.utils.data as data
import random
import json
import io
import tos
import astropy.io.fits as fits
from astropy.wcs import WCS
import time
from math import sqrt
import torch
from scipy.signal import stft
from scipy.interpolate import interp2d
from scipy.interpolate import interp1d
from reproject import reproject_exact, reproject_interp
from matplotlib.colors import Normalize
import math
import shutil

def get_starburst_templete(kenerl_size=34, feature_size=128):
    # kernel img
    kernel = np.zeros((kenerl_size, kenerl_size), dtype=np.float32)
    # left 
    kernel[kenerl_size//2-1:kenerl_size//2+1, :kenerl_size//2] = 1.0  
    # right
    kernel[kenerl_size//2-1:kenerl_size//2+1, kenerl_size//2:] = -1.0  
    # up
    kernel[:kenerl_size//2, kenerl_size//2-2:kenerl_size//2+2] = 1.0  
    # down
    kernel[kenerl_size//2:, kenerl_size//2-2:kenerl_size//2+2] = -1.0  
    
    kernel_img = np.zeros((feature_size, feature_size), dtype=np.float32)
    kernel_img[feature_size//2-kenerl_size//2:feature_size//2+kenerl_size-kenerl_size//2,
               feature_size//2-kenerl_size//2:feature_size//2+kenerl_size-kenerl_size//2] = kernel
    return kernel_img

def modify_homography(homography, feature_size, scale_factor=1.0):
    scale_matrix = scale_factor*np.eye(2)
    homography = scale_matrix@homography
    homography = np.concatenate([homography,
                                 np.array([[(1-homography[0].sum())*(feature_size//2),
                                            (1-homography[1].sum())*(feature_size//2)]]).T],-1)
    homography = np.concatenate([homography,np.array([[0,0,1]])])
    return homography

def projected_kernel(homography,starburst_kernel):
    return cv2.warpPerspective(starburst_kernel, 
                        homography, 
                        (starburst_kernel.shape[1],  starburst_kernel.shape[0]), 
                        flags=cv2.INTER_LANCZOS4, 
                        borderValue=0.0)

def read_filter(filter_info_files):
    all_filter_infos = []
    x_min_list, x_max_list = [], []
    for filter_info_file in filter_info_files:

        filter_infos = np.loadtxt(filter_info_file)

        x_min_list.append(filter_infos[:, 0].min()), x_max_list.append(filter_infos[:, 0].max())
        
        all_filter_infos.append(filter_infos)

    min_x, max_x = min(x_min_list), max(x_max_list)

    return all_filter_infos, min_x, max_x

def padding(filter_infos, min_x, max_x):
    
    # before

    start_index = filter_infos[:, 0].min()
    end_index = filter_infos[:, 0].max()
    before_indices = math.ceil(start_index - min_x + 1)
    before_list = []
    for i in range(before_indices):
        before_list.append(np.array([min_x + i, 0.0]))
    
    before_array = np.array(before_list)
    filter_infos = np.concatenate((before_array, filter_infos))
    
    # after
    after_indices = math.ceil(max_x - end_index + 1)
    after_list = []
    for i in range(after_indices):
        after_list.append(np.array([end_index + i + 1, 0.0]))
    
    after_array = np.array(after_list)
    try:
        filter_infos = np.concatenate((filter_infos, after_array))
    except:
        
        import pdb; pdb.set_trace()

    return filter_infos

def normalize_row(arr):
    arr_norm = (arr - arr.min(axis=1, keepdims=True)) / (arr.max(axis=1, keepdims=True) - arr.min(axis=1, keepdims=True))
    return arr_norm

def normalize_column(arr):
    arr_norm = (arr - arr.min(axis=0, keepdims=True)) / (arr.max(axis=0, keepdims=True) - arr.min(axis=0, keepdims=True))
    return arr_norm

def get_dark_variance_gain(camcol, band, run):
        """
        根据 camcol, band 和 run 获取 dark variance 值
        :param camcol: CCD 的 camcol (1-6)
        :param band: 光学波段 ('u', 'g', 'r', 'i', 'z')
        :param run: 运行编号
        :return: dark variance 值
        """   
        dark_variance_tab = {
            1: {'u': 9.61, 'g': 15.6025, 'r': 1.8225, 'i': 7.84, 'z': 0.81},
            2: {'u': 12.6025, 'g': 1.44, 'r': 1.00,
                'i': {'1500-': 5.76, '1500+': 6.25},
                'z': 1.0},
            3: {'u': 8.7025, 'g': 1.3225, 'r': 1.3225, 'i': 4.6225, 'z': 1.0},
            4: {'u': 12.6025, 'g': 1.96, 'r': 1.3225,
                'i': {'1500-': 6.25, '1500+': 7.5625},
                'z': {'1500-': 9.61, '1500+': 12.6025}},
            5: {'u': 9.3025, 'g': 1.1025, 'r': 0.81, 'i': 7.84,
                'z': {'1500-': 1.8225, '1500+': 2.1025}},
            6: {'u': 7.0225, 'g': 1.8225, 'r': 0.9025, 'i': 5.0625, 'z': 1.21}
        }

        gain_tab = {
            1: {'u': 1.62, 'g': 3.32, 'r': 4.71, 'i': 5.165, 'z': 4.745},
            2: {'u': {'1100-': 1.595, '1100+': 1.825}, 
                'g': 3.855, 'r': 4.6, 'i': 6.565, 'z': 5.155},
            3: {'u': 1.59, 'g': 3.845, 'r': 4.72, 'i': 4.86, 'z': 4.885},
            4: {'u': 1.6, 'g': 3.995, 'r': 4.76, 'i': 4.885, 'z': 4.775},
            5: {'u': 1.47, 'g': 4.05, 'r': 4.725, 'i': 4.64, 'z': 3.48},
            6: {'u': 2.17, 'g': 4.035, 'r': 4.895, 'i': 4.76, 'z': 4.69},
        }

        dark_variance = dark_variance_tab[camcol][band]
        if isinstance(dark_variance, dict):  # 针对有条件的值
            if run <= 1500:
                dark_variance = dark_variance.get('1500-')
            else:
                dark_variance =  dark_variance.get('1500+')

        gain = gain_tab[camcol][band]
        if isinstance(gain, dict):
            if run <= 1100:
                gain = gain.get('1100-')
            else:
                gain =  gain.get('1100+')
        return dark_variance, gain  

def get_err_img(hdulist):
    darkVariance, gain = get_dark_variance_gain(hdulist[0].header['CAMCOL'], 
                                                hdulist[0].header['FILTER'],
                                                hdulist[0].header['RUN'])

    img = hdulist[0].data.copy()
    img[img < 0.] = 0.
    allsky = hdulist[2].data['ALLSKY'][0]
    xinterp = hdulist[2].data['XINTERP'][0]
    yinterp = hdulist[2].data['YINTERP'][0]

    xx = np.linspace(0, allsky.shape[1] - 1, allsky.shape[1])
    yy = np.linspace(0, allsky.shape[0] - 1, allsky.shape[0])
    f = interp2d(xx, yy, allsky, kind='linear')

    # 使用插值函数计算新坐标点的值
    simg = f(xinterp, yinterp)

    column_vector  = hdulist[1].data[:, np.newaxis]
    cimg = np.repeat(column_vector, img.shape[0], axis=1).T

    dn = img/cimg+simg
    dn_err = np.sqrt(dn/gain+darkVariance)
    img_err = dn_err*cimg
    return img_err


class Deep_Space(data.Dataset):
    def __init__(self, 
                 root_dir,
                 metadata,
                 filter_dir='./metadata/filter_infos',
                 patch_save_dir='./patchify_data', 
                 patch_size = 256,
                 data_type='float32',
                 overlap = 0.5,
                 fast_reproject=True,
                 **kargs):
        '''
            root dir: the directory of 5-band input files
            metadata: expected restoration setting
            filter_dir: the directory of filter transmission
            patch_save_dir: a temporary directory to save patchify data
            patch_size: patch size
            overlap: cross-patch overlap ratio
            fast_reproject: True--reproject_interp, False--reproject_exact
        '''
        print('1. Data Loading......')
        self.root_dir = root_dir
        self.data_type = data_type
        self.patch_size = patch_size
        self.overlap = overlap
        self.metadata = metadata
        
        # load filter infos
        self.load_filter_infos(filter_dir)

        # get 5 band file names, sorted with ugriz
        sdss_files = os.listdir(self.root_dir)
        band_order = {'u': 0, 'g': 1, 'r': 2, 'i': 3, 'z': 4}
        sdss_files = sorted(sdss_files, key=lambda s: band_order[self._get_band(s)])
        
        # load telescope metadatas (imaging parameters)
        self.source_mata = self.get_input_telescope_metadata(sdss_files)
        self.target_mata = self.get_target_telescope_metadata(self.source_mata)
        print('   Data Loaded')
        print('2. Data Reproject......')
        # reproject data (traditional upsampling)
        source_img, source_mask, source_err = self.reproject_inputs(sdss_files, self.target_mata, fast_reproject)
        
        ## patchify data and save in tmp dir
        # create patch directory
        self.patch_save_dir = patch_save_dir
        if os.path.exists(self.patch_save_dir):
            shutil.rmtree(self.patch_save_dir)
        os.makedirs(self.patch_save_dir)
        # patchify
        self.patchify_reproject(source_img, source_mask, source_err)
        print('   Data Reprojected')
        
        self.data_list = os.listdir(self.patch_save_dir)

    def __getitem__(self, item):
        st_file = self.data_list[item]
        
        st_data = np.load(os.path.join(self.patch_save_dir, st_file), allow_pickle=True).tolist()

        # noramlize
        source_data = (st_data['source'] - np.array([0.00476682, 0.00850402, 0.01348931, 0.02098092, 0.03629631]))/np.array([0.44788926, 0.45171679, 0.60070421, 0.85422988, 2.69837011])
        source_data = np.transpose(source_data,(2,0,1))
        source_data = np.nan_to_num(source_data, 0.0)

        source_err = np.transpose(st_data['source_err'] / np.array([0.44788926, 0.45171679, 0.60070421, 0.85422988, 2.69837011]), (2, 0, 1))

        source_err = np.nan_to_num(source_err, 0.0)
        mask = np.prod((source_data !=0), 0) * np.prod((source_err !=0), 0)
        

        return dict(input = source_data.astype(self.data_type), 
                    mask = mask,
                    source_err = source_err.astype(self.data_type),
                    exp_times = np.array((self.source_mata['exptime'], self.target_mata['exptime'])).astype(self.data_type),
                    source_filter_infos = self.source_mata['filter_infos'].astype(self.data_type),
                    target_filter_infos = self.target_mata['filter_infos'].astype(self.data_type),
                    flc2drc_condition = self.target_mata['anisotropy'],
                    sdss2drc_condition = self.source_mata['anisotropy'],
                    coordinate=torch.tensor(st_data['coordinate']),
                    sdss_ratio_feat_map = self.source_mata['scale_matrix'].astype(self.data_type),
                    hst_ratio_feat_map = self.target_mata['scale_matrix'].astype(self.data_type),
                    item = item)


    def _get_band(self,s):
        for b in 'ugriz':
            if f"-{b}-" in s:
                return b
        raise ValueError(f"No band found in {s}")

    def load_filter_infos(self,filter_dir):
        # read filter information
        hst_filter_info_files = [os.path.join(filter_dir, 'hst_filter', 'WFC3-UVIS2', 'HST_WFC3_UVIS2.F814W.dat'), 
                                    os.path.join(filter_dir, 'hst_filter', 'WFC3-UVIS2', 'HST_WFC3_UVIS2.F555W.dat'),
                                    os.path.join(filter_dir, 'hst_filter', 'WFC3-UVIS2', 'HST_WFC3_UVIS2.F438W.dat'),
                                    os.path.join(filter_dir, 'hst_filter', 'WFC3-UVIS2', 'HST_WFC3_UVIS2.F775W.dat'),
                                    os.path.join(filter_dir, 'hst_filter', 'WFC3-UVIS2', 'HST_WFC3_UVIS2.F275W.dat'),
                                    os.path.join(filter_dir, 'hst_filter', 'WFC3-UVIS2', 'HST_WFC3_UVIS2.F625W.dat'),
                                    os.path.join(filter_dir, 'hst_filter', 'WFC3-UVIS2', 'HST_WFC3_UVIS2.F390W.dat'),
                                    os.path.join(filter_dir, 'hst_filter', 'WFC3-UVIS2', 'HST_WFC3_UVIS2.F475W.dat'),
                                    os.path.join(filter_dir, 'hst_filter', 'WFC3-UVIS2', 'HST_WFC3_UVIS2.F606W.dat'),
                                    os.path.join(filter_dir, 'hst_filter', 'ACS_WFC', 'HST_ACS_WFC.F625W.dat'),
                                    os.path.join(filter_dir, 'hst_filter', 'ACS_WFC', 'HST_ACS_WFC.F555W.dat'),
                                    os.path.join(filter_dir, 'hst_filter', 'ACS_WFC', 'HST_ACS_WFC.F435W.dat'),
                                    os.path.join(filter_dir, 'hst_filter', 'ACS_WFC', 'HST_ACS_WFC.F775W.dat'),
                                    os.path.join(filter_dir, 'hst_filter', 'ACS_WFC', 'HST_ACS_WFC.F475W.dat'),
                                    os.path.join(filter_dir, 'hst_filter', 'ACS_WFC', 'HST_ACS_WFC.F814W.dat'),
                                    os.path.join(filter_dir, 'hst_filter', 'ACS_WFC', 'HST_ACS_WFC.F606W.dat')]

        sdss_filter_info_files = [os.path.join(filter_dir, 'sdss_filter', 'SLOAN_SDSS', 'SLOAN_SDSS.{}.dat'.format(item)) for item in ['g', 'i', 'r', 'u', 'z']]
        sdss_filter_infos, sdss_min_x, sdss_max_x = read_filter(sdss_filter_info_files)
        hst_filter_infos, hst_min_x, hst_max_x = read_filter(hst_filter_info_files)
        min_x, max_x = min(sdss_min_x, hst_min_x), max(sdss_max_x, hst_max_x)

        # padding sdss fillters
        padding_sdss_filter_infos = []
        for item in sdss_filter_infos:
            padding_item = padding(item, min_x, max_x)
            padding_sdss_filter_infos.append(padding_item)
        
        # sampling and stft
        new_sdss_filter_infos = []
        for item in padding_sdss_filter_infos:
            interp_func = interp1d(item[:, 0], item[:, 1], fill_value=0.0)
            xnew = np.arange(min_x, max_x, 1)    # interval 1
            ynew = interp_func(xnew)

            f, t, Zxx = stft(ynew, fs=1)

            magnitude = np.abs(Zxx)
            phase = np.angle(Zxx)

            new_sdss_filter_infos.append(np.stack([magnitude, phase], axis=0))

        
        new_sdss_filter_infos = np.stack(new_sdss_filter_infos)   # [5, 2, f, t]

        coord_map_xx, coord_map_yy = np.meshgrid(f, t, indexing='ij')
        norm_coord_map_xx = normalize_column(coord_map_xx)
        norm_coord_map_yy = normalize_row(coord_map_yy)

        coord_map = np.stack((norm_coord_map_xx, norm_coord_map_yy), axis=0)  # [2, f, t]
        
        coord_map = coord_map[np.newaxis, :, :]  # [1, 2, f, t]
        sdss_coord_map = np.tile(coord_map, (5, 1, 1, 1))    # [5, 2, f, t]
        new_sdss_filter_infos = np.concatenate([new_sdss_filter_infos, sdss_coord_map], axis=1)  # [5, 4, f, t]
        
        # padding hst fillters
        padding_hst_filter_infos = []
        for item in hst_filter_infos:
            padding_item = padding(item, min_x, max_x)
            padding_hst_filter_infos.append(padding_item)
        
        # sampling and stft
        new_hst_filter_infos = dict()
        for item, file_ in zip(padding_hst_filter_infos, hst_filter_info_files):
            interp_func = interp1d(item[:, 0], item[:, 1], fill_value=0.0)
            xnew = np.arange(min_x, max_x, 1)    # interval 1
            ynew = interp_func(xnew)

            f, t, Zxx = stft(ynew, fs=1)

            magnitude = np.abs(Zxx)
            phase = np.angle(Zxx)

            hst_filter_infos = np.stack([magnitude, phase], axis=0)   # [2, f, t]
            hst_filter_infos = hst_filter_infos[np.newaxis, :, :]   # [1, 2, f, t]

            filter_name = file_.split('/')[-1].split('.')[0].split('_')[1] + '_' + file_.split('/')[-1].split('.')[1]                      # HST_WFC3_UVIS2.F814W.dat

            hst_filter_infos = np.concatenate([hst_filter_infos, coord_map], axis=1)  # [1, 4, f, t]
            new_hst_filter_infos[filter_name] = np.squeeze(hst_filter_infos)
        self.sdss_filters, self.hst_filters = new_sdss_filter_infos, new_hst_filter_infos
    
    def get_input_telescope_metadata(self,sdss_files):
        ## load any one band fits to extract metadata
        source_hdu = fits.open(os.path.join(self.root_dir, sdss_files[0]))
        ## calculating pixel scale (ratio_k upsampling)
        source_wcs = WCS(source_hdu[0].header, fobj=source_hdu, naxis=2)
        source_pix_scale = source_wcs.pixel_scale_matrix
        source_scale = np.sqrt((source_pix_scale**2).sum(axis=0)) * 3600    
        sdss2drc_condition = self._meta_anisotropy(np.eye(2), scale_factor=0.5) 
        sdss_ratio_feat_map = self._meta_scalematrix(source_scale)
        
        return dict(
            wcs = source_wcs,
            scale = source_scale,
            exptime = source_hdu[0].header['EXPTIME'],
            anisotropy = sdss2drc_condition,
            scale_matrix = sdss_ratio_feat_map,
            filter_infos = self.sdss_filters,
        )
        
    def get_target_telescope_metadata(self,source_meta):
        new_wcs = source_meta['wcs'].deepcopy()
        new_wcs.wcs.crpix *= self.metadata['ratio_k'] 
        new_wcs.wcs.cd /= self.metadata['ratio_k'] 
        target_scale = source_meta['scale'] / (self.metadata['ratio_k'] * 2) #ratio need to x2. because during training, 
                        #the training target is half HST resolution, but the target scale is original HST one (x2)
        flc2drc_condition = self._meta_anisotropy(np.eye(2), scale_factor=5.0) 
        hst_ratio_feat_map = self._meta_scalematrix(target_scale)
        if 'ACS' in self.metadata['instrument_name']:
            filter_infos = self.hst_filters['ACS_{}'.format(self.metadata['filter'])]
        else:
            filter_infos = self.hst_filters['WFC3_{}'.format(self.metadata['filter'])]
        # modify final WCS
        self.new_wcs = new_wcs.deepcopy()
        self.new_wcs.pixel_shape = tuple(np.array(self.new_wcs.pixel_shape)*self.metadata['ratio_k']) 
        return dict(
            wcs = new_wcs,
            scale = target_scale,
            exptime = self.metadata['exptime'],
            anisotropy = flc2drc_condition,
            scale_matrix = hst_ratio_feat_map,
            filter_infos = filter_infos,
        )
        

    def decode_pred_array(self, pred_patch, pred_patch_un):
        if self.metadata['PHOTOPLAM']=='None' and self.metadata['PHOTOFLAM']=='None':
            print("   \033[31mDo not have accurate FLAM and PLAM for unit converse (nanomaggy to counts/sec).\033[0m")
            print("   \033[43mUse the average FLAM and PLAM of the corresponding device and filter\033[0m")
            print("   \033[43mThe original nanomaggy is more accurate\033[0m")
            photopara = np.load('../metadata/FLAM_PLAM.npy', allow_pickle=True).tolist()
            self.metadata['PHOTOFLAM'], self.metadata['PHOTOPLAM'] = photopara[self.metadata['instrument_name'].replace('/','_')][self.metadata['filter']]
        ## decode value
        # denormlize value: (v_hst*c2n*s2t_ratio-mean_hst)/std_hst-->v_hst*c2n 
        pred_patch = (pred_patch*0.85422988 + 0.02098092)/(self.metadata['ratio_k'])**2
        #sigma denormalize (same to value, sigma_hst*c2n)
        pred_patch_un = pred_patch_un*0.85422988/(self.metadata['ratio_k'])**2

        # * compute possion err
        c2n = self.metadata['PHOTOFLAM']*self.metadata['PHOTOPLAM']**2*10**((-21.10+18.6921-8.9)/(-2.5))/3.631e-6
        # convert nanomaggy to ori counts 
        # v_hst*c2n-->v_hst
        pred_patch /= c2n
        # c2n*drizzle_sigma_hst-->drizzle_sigma_hst
        pred_patch_un /= c2n
        signal_counts = pred_patch * self.metadata['exptime']  # 转换为电子数
        sigma_poisson = np.sqrt(np.abs(signal_counts)) / self.metadata['exptime']  # 再转换为 counts/sec 的σ
        pred_patch_un = np.sqrt(pred_patch_un**2 + sigma_poisson**2)
        return pred_patch, pred_patch_un
        
        
    def patchify_reproject(self, source_img, source_mask, source_err):
        patch_size = self.patch_size
        overlap = self.overlap
        stride = int(patch_size * (1 - overlap))
        # patchify the input data
        for x_idx in range(source_img.shape[0]//(stride)):
            for y_idx in range(source_img.shape[1]//(stride)):
                pathify_mask = source_mask[x_idx*(stride):x_idx*(stride)+patch_size,
                                        y_idx*(stride):y_idx*(stride)+patch_size]
                if pathify_mask.mean()<=0.7 or \
                    pathify_mask.shape[0]!=patch_size or pathify_mask.shape[1]!=patch_size:
                    continue
                patchify_source = source_img[x_idx*(stride):x_idx*(stride)+patch_size,
                                                    y_idx*(stride):y_idx*(stride)+patch_size]
                patchify_source_err = source_err[x_idx*(stride):x_idx*(stride)+patch_size,
                                                y_idx*(stride):y_idx*(stride)+patch_size]

                save_data = dict(source = patchify_source, 
                                mask = pathify_mask,
                                source_err = patchify_source_err,
                                coordinate=[[x_idx*(stride), y_idx*(stride)],\
                                                [x_idx*(stride)+patch_size, y_idx*(stride)+patch_size]])
                np.save(os.path.join(self.patch_save_dir,'%d-%d.npy'%(x_idx,y_idx)),save_data)
        
    def reproject_inputs(self, sdss_files, target_mata, fast_reproject):
        self.project_shape = np.array(target_mata['wcs'].array_shape)*self.metadata['ratio_k']  #array([5956, 8192])
        print('resolution change from {} to {}'.format(self.target_mata['wcs'].pixel_shape, tuple(self.project_shape)))
        one_batch_source = [] #sdss: 5 u,g,r,i,z
        one_batch_mask = []
        one_batch_err = []

        for item in sdss_files:
            sdss_file = os.path.join(self.root_dir, item)
            sdss_hdu = fits.open(sdss_file)
            sdss_err = get_err_img(sdss_hdu)

            img_mask = ~np.isnan(sdss_hdu[0].data)
            if fast_reproject:
                source_data_reprojected, source_footprint = reproject_interp(fits.PrimaryHDU(data=sdss_hdu[0].data, header=sdss_hdu[0].header), \
                                                                        target_mata['wcs'], shape_out=self.project_shape)
            else:
                source_data_reprojected, source_footprint = reproject_exact(fits.PrimaryHDU(data=sdss_hdu[0].data, header=sdss_hdu[0].header), \
                                                                        target_mata['wcs'], shape_out=self.project_shape)
            reproject_mask = (source_footprint!=0)*(cv2.resize((1-img_mask).astype(np.float32), (self.project_shape[1],self.project_shape[0]))==0)
            source_err_resize = cv2.resize(sdss_err, (self.project_shape[1],self.project_shape[0]))

            one_batch_source.append(source_data_reprojected)
            one_batch_mask.append(reproject_mask)
            one_batch_err.append(source_err_resize)
        
        source_img = np.stack(one_batch_source,-1) #ugriz
        source_mask = np.prod(np.stack(one_batch_mask,-1),-1).astype(bool)
        source_err = np.stack(one_batch_err, -1) 
        return source_img, source_mask, source_err
        
    def _meta_anisotropy(self,homography,scale_factor=1.0):
        starburst_kernel = get_starburst_templete(feature_size=self.patch_size)
        homography = modify_homography(homography, feature_size=self.patch_size, scale_factor=scale_factor)
        anisotropy = projected_kernel(homography,starburst_kernel)
        anisotropy = np.expand_dims(anisotropy, axis=0)
        return anisotropy
    def _meta_scalematrix(self,scale):
        x_coords, y_coords = np.arange(256) * scale[0], np.arange(256) * scale[1]
        ratio_feat_map = np.stack(np.meshgrid(x_coords, y_coords)) / 100
        return ratio_feat_map        

    def __len__(self):
        return len(self.data_list)
