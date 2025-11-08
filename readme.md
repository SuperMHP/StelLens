## StelLens

This is the official repository for the StelLens papers.

'''paper link'''

Resources including pre-trained model and inference code are released here.

## Installation

The downloaded files shall be organized as the following hierarchy:

```plain
├── root
│   ├── patchify_data
│   │   ├── 301_1458_4_700
│   │   │   ├──0-0.npy
│   │   │   ├──0-1.npy
│   │   │   ├──...
│   ├── sdss_data
│   │   ├── 301_1458_4_700
│   │   │   ├──frame-g-001458-4-0700.fits.bz2
│   │   │   ├──frame-i-001458-4-0700.fits.bz2
│   │   │   ├──...
│   ├── filter_infos
│   │   ├── hst_file
│   │   ├── sdss_file
│   ├── output_data
│   ├── model.onnx
│   ├── config.yaml
│   ├── requirements.txt
│   ├── dataset.py
│   ├── inference.py
│   ├── vis.py
```

You need a GPU environment, and run:
```
pip install -r requirements.txt
```

## Deep field revelation (inference) using the trained model

#### Downloading trained model

Please download the pre-trained model (~?GB each) from Google drive or Baidu netdisk:

The model (stellens.onnx): [Google drive](link)/[Baidu netdisk](link)

The model is stored using the ONNX format, and thus can be used via different languages such as Python, C++, C#, Java, etc.

#### Input data preparation

Please prepare the 5-band SDSS data in FITS format and place them in the `sdss_data/{data_name}` folder. 
The files should follow the naming convention that includes the band identifier (e.g., `-g-`, `-r-`, etc.), such as:
`frame-g-xxx.fits.bz2`, `frame-r-xxx.fits.bz2`, `frame-i-xxx.fits.bz2`, `frame-u-xxx.fits.bz2`, and `frame-z-xxx.fits.bz2`,
where `xxx` can be any custom identifier.

We provide an example of the input data, `sdss_data/301_1458_4_700`, which corresponds to a dense star field case. Please download them from [Google drive](https://drive.google.com/drive/folders/1tzOlm2ycRS5f56kKV5Qf8m6OjC5yztRo?usp=sharing).

#### Filter information preparation

The filter transmission curves of ground- and space-based telescopes are required for inference. 
Please download the filter transmission files from [Google drive](https://drive.google.com/drive/folders/1JDytApf61C6yURcwgo2OpYRZDkoQsVuP?usp=sharing) and place them under the `filter_infos` folder.

#### Define parameters in the configuration file

Please configure the parameters in `config.yaml`, which defines the parameters for data preprocessing, metadata settings, and visualization options used during the inference process. 

###### Input (`input_dir`)

Specifies the path to the folder containing the input ground-based data, such as `'./sdss_data/301_1458_4_700'`.


###### Metadata (`metadata`)

Defines the target observation parameters that guide the model’s inference process.

| Parameter             | Description                                                      | Example     |
| --------------------- | ---------------------------------------------------------------- | ----------- |
| **`ratio_k`**        | Desired image upsampling factor (typically between 2–5).                | `4`  
| **`exptime`**         | Target exposure time (in seconds) for the simulated observation. | `300.0`     |
| **`filter`**          | Target filter name used for inference. <br>Options include `'F390W'`- `'F814W'`     | `'F625W'`   |
| **`instrument_name`** | Target instrument configuration. <br>Options include `'ACS/WFC'` and `'WFC3/UVIS'`  | `'ACS/WFC'` |


###### Visualization (`vis`)

Controls the visualization mode used for inspecting output results.

| Parameter  | Description                                                         | Example        |
| ---------- | ------------------------------------------------------------------- | -------------- |
| **`mode`** | Visualization normalization method. <br>Options include `'percentile'`, `'zscale'`, and `'hybrid'`. | `'percentile'` |


#### Inference

After the above steps are finished, please check `inference.py` for an example of near-space-quality image reconstruction from ground-based observations.

For example, running the following command, one can get a near-space-quality image with its per-pixel uncertainty map in FITS format in the `output_data` folder:

```
python inference.py 
```

## License
