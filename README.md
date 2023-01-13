# DeepAstroUDA
We develop a Universal Domain Adaptation method DeepAstroUDA, capable of performing semi-supervised domain alignment that can be applied to datasets with different types of class overlap. Extra classes can be present in any of the two datasets, and the method can even be used in the presence of unknown classes. DeepAstroUDA can be used for classification, regression and anomaly detection. 

<div align="center">
<img align="center" width="700" src="images/DA_horizontal.png">
  
<sub>Types of DA problems. Source domain is represented with solid line ellipse, while target domain uses dashed lines. Classes in the source domain are represented with filed shapes and target domain classes by empty shapes. In our work we focus on Open DA problems, but the code we develop is capable of handling all four types of DA problems.</sub>
</div>

### Intro and Experiments
In the era of big astronomical surveys, our ability to leverage artificial intelligence algorithms simultaneously for multiple datasets will open new avenues for scientific discovery. Unfortunately, simply training a deep neural network on images from one data domain often leads to very poor performance on any other dataset.
For the first time, we demonstrate the successful use of domain adaptation on two very different observational datasets (from SDSS and DECaLS). We show that our method is capable of bridging the gap between two astronomical surveys, and also performs well for anomaly detection and clustering of unknown data in the unlabeled dataset. We apply our model to two examples of galaxy morphology classification tasks with anomaly detection: 1) classifying spiral and elliptical galaxies with detection of merging galaxies (three classes including one unknown anomaly class); 2) a more granular problem where the classes describe more detailed morphological properties of galaxies, with the detection of gravitational lenses (ten classes including one unknown anomaly class).

### Architecture
Our experiments were performed using a ResNet50 architecture, trained with early stopping that monitors the change in accuracy and stops the training when there is no improvement in 12 epochs. Domain-specific batch normalization is used to eliminates domain style information leakage. The model is trained using stochastic gradient descent with Nesterov momentum and an initial learning rate of 0.001. We train our models on 4 NVIDIA RTX A6000 GPUs (available from Google Colab and LambdaLabs), and on average the training converges in approximately 5 hours. 


<div align="center">
<img align="left" width="300" src="images/UDA1.png">
<br>
<br>
<br>
<sub>DeepAstroUDA method and the effects of different loss functions in an OpenDA problem. Cross-entropy loss (filled red arrows) clusters the labeled source domain data (filled circles and squares). Adaptive clustering loss (empty red arrows) pushes unlabeled target domain data (empty circles and squares), towards data it shares most similar features with (both source and target data, which are stored in the bank). Finally entropy loss (empty violet arrows) uses entropy to push away unknown classes further away from the known ones. In this Open DA example the unknown class is present in the target domain (empty squares.)</sub>
</div>
<br>
<br>
<br>
<br>

### Requirements
This code was developed using Pytorch. The poetry.lock file lists all Python libraries that your notebooks will need, and they can be installed using the following command:
```
python3 -m poetry shell
```

### Pieces of the Package

#### Input

DeepAstroUDA includes access to the template files and sample datasets needed to run the system with varying amounts of input. Any sample datasets are automatically downloaded.

The available sample datasets (and their original use cases for this project) used are as follows:

| Dataset | Experimental Use |
|:---|:---|
| [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html) | fast bench-marking and initial hyperparameter testing |
| [Astro-NN](https://astronn.readthedocs.io/en/latest/galaxy10.html) | subset of GalaxyZoo2 data (DECaLS data with cleaner class distinctions and 10 galaxy morphology classes) |
| [GalaxyZoo2](https://data.galaxyzoo.org) | GalazyZoo2 10-class data (SDSS wide field and Stripe 82 coadded-depth deep field)|
| [DeepAdversaries](https://zenodo.org/record/5514180#.Y2FkZC-B2_w) | Simulated LSST mocks after 1 and 10 years of observing (3 classes: spiral, elliptical, merging galaxy)|

Along with the datasets above, an available configuration file template is available for new datasets. If a configuration file is not included at run-time, a configuration file of the same format will be output at the end of training.

#### Output<a name="output"></a>

The following are the metric documents output by the system:

- Loss CSV and plot
- Accuracy CSV and plots (total, closed, and per class accuracy)
- t-SNE plot and t-SNE visualization-tool file

## Using DeepAstroUDA

Below are the list of commands, along with some possible approaches to using DeepAstroUDA.

### DeepAstroUDA Available Commands and Optionals <a name="deep"></a>

| Command | Function |
|:---|:---|
| `deep_dance -h `| Displays a help prompt that gives a comprehensive overview of possible commands. |
| `deep_dance demo `| Automatically runs the DANCE pipeline on the Astro-NN dataset. | 
| `deep_dance demo -h` | Displays a help prompt that gives a cromprehensive overview of possible arguments and optionals. |
| `deep_dance demo --dataset={'office', 'astro-nn', 'gz2', 'deep-adv', 'data'}` | Runs a chosen example dataset with default training configuration. Use the **'data'** option if you're running deep_dance on your own dataset. |
| `deep_dance demo --config-path={path\to\referenced\file}` | Not a required value. Uses the provided path to get training configuration information from yaml. If file path not provided, a default training configuration yaml is created at the DEFAULT_PATH="./files/config_env/{dataset_name}-{num_classes}-train-config_{domain_type}. |
| `deep_dance demo --unknowns-supplied={Boolean}` | Not a required value. Informs program whether to use auto-clustering or conform to number of clusters specified in config file. If the variable is not provided, then the training will default to supplied based on data directory structure. |
| `deep_dance demo --image-path-text={path\to\referenced\file}` | Not a required value. Informs program whether an image directory to file path text file has been created. If optional is not supplied, a utility function will be called to create one. |
| `deep_dance demo --output-directory={path\to\referenced\directory}` | Not a required value. Informs program whether a desired output directory exists for training and testing output. If optional is not supplied, an output directory structure will be created. |
| `deep_dance demo --data-type={'jpg', 'png', 'numpy'}` | Not a required value. Informs program whether your personal dataset is in the form of jpeg, png, or .npy images/arrays. If optional is not supplied, program will (1) warn the user, and (2) infer the data type by peaking at the first data file available. |
| `deep_dance demo --domain-type={'open', 'open-partial', 'closed'}` | Allows user to choose what type of domain adaptation is being used ('open-partial' covers both partial and open-partial). If not supplied, **open** is the default. For comparison model trained without domain adaptation is also trained in parallel with any domain adaptation training.|




### Possible Experimental Setups

To begin any example run of the DeepAstroUDA code, pip install the package using the following code:
`pip install deepastrouda`. Any required data will be automatically downloaded by the package.


|:---||
| [Run a Simple Example ](#example) |
| [Using Input Data + Default Training Configuration](#data-default) |
| [Using Input Data + Input Training Configuration](#data-config) |


<!-- Checking links in paper [DeepAstroUDA Available Commands](#deep)
-- feature table with hyperlink to different approaches to using DeepAstroUDA -->

#### Approach 1: Run Minimal Example <a name="example"></a>

Users should then configure their desired output paths and project environment path in their `config.json` file as follows:

```
{
  "path_to_ouput": "path/to/output/folder",
  "environment_path": "path/to/venv/in/use",
}
```

Lastly, begin the demo run using the command-line snippet below. All [output](#output) from both training and testing will be dumped in the output folder.

` deep_astro demo `

If you'd like to use another example dataset listed above, you can use the command-line optional below to specify which dataset to use.

For more help, run the following command-line code snippet:

` deep_astro demo -h`

#### Approach 2: Input Data + Default Training Configuration <a name="data-default"></a> 



<!-- ### Approach 3: Input Data + Input Training Configuration File <a name="data-config"></a> -->


### Authors
- Aleksandra Ćiprijanović
- Ashia Lewis

### References
If you use this code, please cite our paper: [arXiv:2211.00677](https://arxiv.org/abs/2211.00677)
