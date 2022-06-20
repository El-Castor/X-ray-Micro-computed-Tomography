

# X-ray Micro Computed Tomography Data Analysis Documentation

# Table of Contents

   * [1. Analysis Objectives](#pipeline-objectives)
   * [2. Steps of the worflow](#teps-of-the-worflow)
      * [Quick installation using conda](#quick-installation-using-conda-linux64)
   * [3. Software and ressources environment](#software-and-ressources-environment)
   * [4. Data and configuration requirements](#data-and-configuration-requirements)
   * [5. Testing](#testing)

----------------
## Terms definition
working directory: repertory from which you launch the pipeline and within which results will be stored
remote repository: refers to any project hosted on internet or other network platform. XrayMCtomography pipeline and test dataset folder are hosted in GitHub.


## 1. Pipeline Objectives
High-resolution imaging techniques such as X-ray micro computed tomography (micro-CT) have recently become increasingly popular for phenotypic trait measurments in plant science. Micro-CT techniques allows 3D reconstruction of scanned objects (Dhondt et al., 2013). In the present pipeline, we propose a micro-CT downstream analysis to reconstruct the shape of respective male and female melon flowers in the context of nectar biology. 

## 2. Steps of the worflow
### 2.1- Reconstruction of 3D images processing
All resulting projection images are reconstruct using the NReacon software (v.1.7.3.0), Bruker-micro-CT, Kontich, Belgium) with post-alignement, beam hardening correction (41%) and ring artifact reduction (8). 3D images were cropped, rotated and registered using the Data viwer program (v1.5.6.2 64 bit) in order to optimize and facilitate the measurments on the different data sets.

### 2.2- Data set analysis

## 3. Data and ressources configuration requirements
### 3.1- Mandatory requirements

all dependencies requiered to run the analysis:

```bash
dependencies:
  - _libgcc_mutex=0.1=main
  - _openmp_mutex=5.1=1_gnu
  - brotli=1.0.9=h166bdaf_7
  - brotli-bin=1.0.9=h166bdaf_7
  - ca-certificates=2022.6.15=ha878542_0
  - certifi=2022.6.15=py37h89c1867_0
  - cloudpickle=2.0.0=pyhd3eb1b0_0
  - colorama=0.4.5=pyhd8ed1ab_0
  - cycler=0.11.0=pyhd8ed1ab_0
  - cytoolz=0.11.0=py37h7b6447c_0
  - dask-core=1.1.4=py37_1
  - dbus=1.13.6=he372182_0
  - decorator=5.1.1=pyhd3eb1b0_0
  - docopt=0.6.2=py_1
  - expat=2.4.8=h27087fc_0
  - fontconfig=2.14.0=h8e229c2_0
  - fonttools=4.33.3=py37h540881e_0
  - freetype=2.10.4=h0708190_1
  - glib=2.69.1=h4ff587b_1
  - gst-plugins-base=1.14.0=hbbd80ab_1
  - gstreamer=1.14.0=h28cd5cc_2
  - icu=58.2=hf484d3e_1000
  - imageio=2.9.0=pyhd3eb1b0_0
  - jpeg=9e=h166bdaf_1
  - kiwisolver=1.4.2=py37h295c915_0
  - ld_impl_linux-64=2.38=h1181459_1
  - libblas=3.9.0=15_linux64_openblas
  - libbrotlicommon=1.0.9=h166bdaf_7
  - libbrotlidec=1.0.9=h166bdaf_7
  - libbrotlienc=1.0.9=h166bdaf_7
  - libcblas=3.9.0=15_linux64_openblas
  - libffi=3.3=he6710b0_2
  - libgcc-ng=11.2.0=h1234567_1
  - libgfortran-ng=12.1.0=h69a702a_16
  - libgfortran5=12.1.0=hdcd56e2_16
  - libgomp=11.2.0=h1234567_1
  - liblapack=3.9.0=15_linux64_openblas
  - libopenblas=0.3.20=pthreads_h78a6416_0
  - libpng=1.6.37=h21135ba_2
  - libstdcxx-ng=11.2.0=h1234567_1
  - libtiff=4.0.10=hc3755c2_1005
  - libuuid=2.32.1=h7f98852_1000
  - libxcb=1.13=h7f98852_1004
  - libxml2=2.9.14=h74e7548_0
  - lz4-c=1.9.3=h9c3ff4c_1
  - matplotlib=3.5.2=py37h89c1867_0
  - matplotlib-base=3.5.2=py37hc347a89_0
  - munkres=1.1.4=pyh9f0ad1d_0
  - ncurses=6.3=h7f8727e_2
  - networkx=2.2=py37_1
  - numpy=1.21.6=py37h976b520_0
  - olefile=0.46=pyh9f0ad1d_1
  - openssl=1.1.1o=h166bdaf_0
  - packaging=21.3=pyhd8ed1ab_0
  - pcre=8.45=h9c3ff4c_0
  - pillow=6.2.1=py37h6b7be26_0
  - pip=21.2.2=py37h06a4308_0
  - pthread-stubs=0.4=h36c2ea0_1001
  - pyparsing=3.0.9=pyhd8ed1ab_0
  - pyqt=5.9.2=py37hcca6a23_4
  - python=3.7.13=h12debd9_0
  - python-dateutil=2.8.2=pyhd8ed1ab_0
  - python_abi=3.7=2_cp37m
  - pywavelets=1.3.0=py37h7f8727e_0
  - qt=5.9.7=h5867ecd_1
  - readline=8.1.2=h7f8727e_1
  - scikit-image=0.19.2=py37h51133e4_0
  - scipy=1.7.3=py37hf2a6cf1_0
  - setuptools=61.2.0=py37h06a4308_0
  - sip=4.19.8=py37hf484d3e_0
  - six=1.16.0=pyh6c4a22f_0
  - sqlite=3.38.3=hc218d9a_0
  - tifffile=2020.10.1=py37hdd07704_2
  - tk=8.6.12=h1ccaba5_0
  - toolz=0.11.2=pyhd3eb1b0_0
  - tornado=6.1=py37h540881e_3
  - tqdm=4.64.0=pyhd8ed1ab_0
  - typing_extensions=4.2.0=pyha770c72_1
  - unicodedata2=14.0.0=py37h540881e_1
  - wheel=0.37.1=pyhd3eb1b0_0
  - xlrd=2.0.1=pyhd8ed1ab_3
  - xorg-libxau=1.0.9=h7f98852_0
  - xorg-libxdmcp=1.1.3=h7f98852_0
  - xz=5.2.5=h7f8727e_1
  - zlib=1.2.12=h7f8727e_2
  - zstd=1.4.9=ha95c52a_0
```

#### 3.1.1- Repository configuration folder 

Strictly keep the same filenames due to scripts dependencies explained below.

.
├── X-ray-Micro-computed-Tomography                                       
│   ├── README.md
│   ├── **codes**
│   |     **├── Read_flowers_Excel_file.py**
│   |     **├── Segmentation_nectary.py**
│   ├── data_infos
│   │     **├── infos_flowers.xlsx
│   ├── conda-env
│   │     **├── XrayMicroCTomography.yml
│   ├── **documentation**
│   |     **├── Documentation-XrayMicroCTomography.md**
└── test_Dataset   


1)                        Read_flowers_Excel_file.py

Read_flowers_Excel_file.py is the submission scipt to use to read data refered in the infos_flowers.xlsx file.

main_command

```python
Usage:
    Read_flowers_Excel_file.py
    Read_flowers_Excel_file.py (-h | --help)
    Read_flowers_Excel_file.py <file> [--typef=<typef>] [--num_lot=<numlot>] \
    [--vol_nectar=<voln>]

Options:  
    -h, --help              for help on syntax
    -t, --typef TYPEF       to take the infos only for one type of flower. The\
    different possibilities are: male/female/hermaphrodite/all [default: 'all']
    -l, --num_lot NUMLOT    to take only one batch: 1 à 8 / all [default: 'all']
    -v, --vol_nectar VOLN   to select the flower where the volume of nectar\
    collected is known: yes/no/all [defaut: 'all']
```

2)                        Segmentation_nectary.py  

Read_flowers_Excel_file.py is the python scipt to use to process the data analysis of files reads using previous script `Read_flowers_Excel_file.py`.

```python

Usage:
    Segmentation_nectary.py
    Segmentation_nectary.py (-h | --help)
    Segmentation_nectary.py <name_directory> [--seuil=<seuil>]
    Segmentation_nectary.py [--path=<path>] [--type=<type>] [--hide | --show]\
    [--quiet | --verbose] [--seuil=<seuil>]

Options:  
    -h, --help          show this
    -s, --seuil SEUIL   threshold parameter for nectary detection [default: 13_995]
    --hide              hide the intermediate images (default)
    --show              visualization of intermediate images
    --path PATH         path of the directory containing images to analyse [default: ./datasets/]
    --type TYPE         dictionary of "types" of flowers to process, as writen
                        in the names of images. ex: {'Ma':'Male', 'Fe':'Female',...}
    --quiet             print less text (default)
    --verbose           print more text

```


## 5. Full example of execution with Test dataset

### 5.1- Prepare cluster configuration 

Clone sapticon repository into your usual common directory readable for all lab users so that they can execute it: 
```bash 
git clone https://github.com/El-Castor/X-ray-Micro-computed-Tomography.git 
``` 

Mandatory: `conda-env` folder must contain its conda environment script `XrayMicroCTomography.yml` to install all dependencies required without conflict.
Optionaly: `main` folder may contain README file `README.md` to give specific instructions for the users. README file is systematically imported into working directory at the first execution.


### 5.2- Prepare your data

Clone test dataset repository within [test dataset repo](link.test.dataset) with : 
```bash 
git clone git/clou test dataset
``` 

Create **downstream analysis directory** directory; for example:
```bash
mkdir analysis-name_2021-30-11
``` 

#### 5.3- First preparatory execution

Execute the data read submission script the first time to prepare analysis directory:
```bash
python3 ./Read_flowers_Excel_file.py 
```

This command will import mandatory files for execution as  `config.yaml` file needed to set parameters and  `units.tsv`, `samples.tsv` sheets needed sequencing data processing.

#### 5.3.1- Excel file configuration
                          

Open `infos_flowers.xlsx` file to edit it according to test diffrent parameters as those already set for test dataset.

#### 5.3.2- Run downstream analysis pipeline

Execute the submission script for the second time to run the pipeline:
```bash
python3 ./Segmentation_nectary.py 
```

