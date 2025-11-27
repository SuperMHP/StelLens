## StelLens

This is the official repository for the StelLens paper:

<Imaging physics-driven artificial intelligence makes ground-based telescope resolve deep field universe><https://www.researchsquare.com/article/rs-8068579/v1>

Resources include a pre-trained model and inference code are released here.

## Installation

The downloaded files shall be organized as the following hierarchy:

```plain
├── root
│   ├── cores
│   ├── input_data
│   │   ├── 301_1458_4_700
│   │   │   ├── frame-g-001458-4-0700.fits.bz2
│   │   │   ├── frame-i-001458-4-0700.fits.bz2
│   │   │   ├── ...
│   ├── metadata
│   │   ├── filter_infos
│   │   │   ├── hst_filter
│   │   │   ├── sdss_filter
│   │   ├── others
│   │   │   ├── train_ra_dec.txt
│   │   │   ├── test_ra_dec.txt
│   │   ├── FLAM_PLAM.npy
│   ├── pretrained
│   │   ├── last.pth
│   ├── config.yaml
│   ├── requirements.txt
│   ├── inference.py
```

#### Filter information preparation

The filter transmission curves of ground- and space-based telescopes are required for inference. 
Please download the filter transmission files from [Google Drive](https://drive.google.com/drive/folders/1JDytApf61C6yURcwgo2OpYRZDkoQsVuP?usp=sharing) and place them under the `filter_infos` folder.

#### Download trained model

Please download the pre-trained model checkpoint file (~568MB) from [Google Drive](https://drive.google.com/file/d/12BOZVGndIsYVfDJlsZZKPHhl6nZ6dFgY/view?usp=sharing).

The model should be put at `./pretrained/last.pth`.

#### Build Environment

You need a GPU environment, and run:
```
pip install -r requirements.txt
```


## Deep field revelation (inference) using the trained model

#### Input data preparation

Please prepare the 5-band SDSS *ugriz*-style data in FITS format and place them in the `input_data/{data_name}` folder. 
The files should follow the naming convention that includes the band identifier (e.g., `-g-`, `-r-`, etc.), such as:
`frame-g-xxx.fits.bz2`, `frame-r-xxx.fits.bz2`, `frame-i-xxx.fits.bz2`, `frame-u-xxx.fits.bz2`, and `frame-z-xxx.fits.bz2`,
where `xxx` can be any custom identifier.

We provide an example of the input data, `input_data/301_1458_4_700`, which corresponds to a dense star field case. Please download them from [Google Drive](https://drive.google.com/drive/folders/1tzOlm2ycRS5f56kKV5Qf8m6OjC5yztRo?usp=sharing).

#### Define parameters in the configuration file

Please configure the parameters in `config.yaml`, which defines the parameters for data preprocessing, metadata settings, and visualization options used during the inference process. 

###### Input (`input_dir`)

Specifies the path to the folder containing the input ground-based data, such as `'./input_data/301_1458_4_700'`.

###### Metadata (`metadata`)

Defines the target observation parameters that guide the model’s inference process.

| Parameter             | Description                                                      | Example     |
| --------------------- | ---------------------------------------------------------------- | ----------- |
| **`ratio_k`**        | Desired image upsampling factor (typically between 2–5).                | `4`  
| **`exptime`**         | Target exposure time (in seconds) for the simulated observation. | `300.0`     |
| **`filter`**          | Target filter name used for inference. <br>Options include `'F390W'`- `'F814W'`     | `'F625W'`   |
| **`instrument_name`** | Target instrument configuration. <br>Options include `'ACS/WFC'` and `'WFC3/UVIS'`  | `'ACS/WFC'` |

Two special parameters `'PHOTOFLAM'` and `'PHOTOPLAM'` could be given for accurate unit transfer and uncertainty estimation (the model outputs are nanomaggy units). Their default values are `'None'`, which means using the average values of each device and filter. 

| Parameter             | Description                                                      | Example     |
| --------------------- | ---------------------------------------------------------------- | ----------- |
| **`PHOTOFLAM`**        | parameter for scaling image counts to physical flux density.                | `'None'`  
| **`PHOTOPLAM`**         | parameter representing the filter’s effective wavelength. | `'None'`     |

###### Visualization (`vis`)

Controls the visualization mode used for inspecting output results.

| Parameter  | Description                                                         | Example        |
| ---------- | ------------------------------------------------------------------- | -------------- |
| **`mode`** | Visualization normalization method. <br>Options include `'percentile'`, `'zscale'`, and `'hybrid'`. | `'percentile'` |

###### Tips (Conflict check)

In `./metadata/others`, there are two files showing the RA/DEC coordinates of training and test samples, respectively. Users can provide test samples freely, but it is recommended that they be at least 10 arcminutes away from any training sample to training avoid information leakage.


#### Inference

After the above steps are finished, please check `inference.py` for an example of near-space-quality image reconstruction from ground-based observations.

For example, running the following command, one can get a near-space-quality image with its per-pixel uncertainty map in FITS format in the `output_data` folder:

```
python inference.py 
```

## License

This project is licensed under the [MIT License](https://opensource.org/license/mit).

**The commercial use of the model is forbidden.**
