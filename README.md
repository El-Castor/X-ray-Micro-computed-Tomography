# X-ray-Micro-computed-Tomography : X-ray micro computed Tomography data analysis.

To install all dependancy you should install miniconda3, see below the documentation :
[conda_documentation](https://docs.conda.io/en/latest/)


## Authors

* Myriam Oger (myriam.oger@def.gouv.fr)
* Abdelhafid Bendahmane (Abdelhafid.Bendahmane@universite-paris-saclay.fr)
* Filip Slavkovic (Filip.slavkovic@universite-paris-saclay.fr)

## Maintainers

* Clement Pichot (clementpch@gmail.fr)
* Myriam Oger (myriam.oger@def.gouv.fr)

## QUICK START


### Clone project

```bash
git clone https://github.com/El-Castor/X-ray-Micro-computed-Tomography.git
```


### Clone test dataSet

```bash
cd ./X-ray-Micro-computed-Tomography
TODO : when a way to put data test in cloud is available
```


### Create a work directory

~~~
mkdir  my_test_project
cd my_test_project
~~~


#### Documentation

See `./documentation/Documentation-XrayMicroCTomography.md` file for details about configuration options.


#### Configure dependency environment

- install conda environment :

~~~
cd ./conda-env
conda env create -f XrayMicroCTomography.yml
~~~

- Activate conda environment :

~~~
conda activate XrayMicroCTomography.yml
~~~

### Execute script

**Note** : Before launch script, be sure to fill excels file, please refer to the documentation.

~~~
python3 ./codes/Segmentation_nectary.py
~~~



## LICENSE  
TODO
To determine with Myriam

## Reference
TODO, when protocol is publish add citation
