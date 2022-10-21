# DeepAstroUDA
Universal Domain Adaptation for cross-survey classification, regression and anomaly detection. Code can be applied to any kind of domain adaptation problems i.e. closed, open, partial or open-partial. 

<div align="center">
<img align="center" width="800" src="images/DA_horizontal.png">
  
<sub>Types of DA problems. Source domain is represented with solid line ellipse, while target domain uses dashed lines. Classes in the source domain are represented with filed shapes and target domain classes by empty shapes. In our work we focus on Open DA problems, but the code we develop is capable of handling all four types of DA problems.</sub>
</div>

### About
In the era of big astronomical surveys, our ability to leverage artificial intelligence algorithms simultaneously for multiple datasets will open new avenues for scientific discovery. Unfortunately, simply training a deep neural network on images from one data domain often leads to very poor performance on any other dataset. Here we develop a Universal Domain Adaptation method \textit{DeepAstroUDA}, capable of performing semi-supervised domain alignment that can be applied to datasets with different types of class overlap. Extra classes can be present in any of the two datasets, and the method can even be used in the presence of unknown classes. For the first time, we demonstrate the successful use of domain adaptation on two very different observational datasets (from SDSS and DECaLS). We show that our method is capable of bridging the gap between two astronomical surveys, and also performs well for anomaly detection and clustering of unknown data in the unlabeled dataset. We apply our model to two examples of galaxy morphology classification tasks with anomaly detection: 1) classifying spiral and elliptical galaxies with detection of merging galaxies (three classes including one unknown anomaly class); 2) a more granular problem where the classes describe more detailed morphological properties of galaxies, with the detection of gravitational lenses (ten classes including one unknown anomaly class).

### Architecture
Our experiments were performed using a ResNet50 architecture, trained with early stopping that monitors the change in accuracy and stops the training when there is no improvement in 12 epochs. Domain-specific batch normalization is used to eliminates domain style information leakage. The model is trained using stochastic gradient descent with Nesterov momentum and an initial learning rate of $0.001$. We train our models on 4 NVIDIA RTX A6000 GPUs (available from Google Colab and LambdaLabs), and on average the training converges in ${\approx}5$ hours. 

### Datasets
We use two GalaxyZoo datasets: SDSS (source domain) and DECaLS (target domain). The images used can be found at [Zenodo](https://....). We apply our method to a3-class and 10-class galaxy morphology classification problem. Furthermore, our unlabeled targed domain contains one uknown anomally class (strong gravitational lenses), that the model needs to also classify i.e. detect, cluster and separate from other known classes.


<div align="center">
<img align="left" width="300" src="images/UDA1.png">
<br>
<br>
<br>
<sub>DeepAstroUDA method and the effects of different loss functions in an OpenDA problem. Cross-entropy loss (filled red arrows) clusters the labeled source domain data (filled circles and squares). Adaptive clustering loss (empty red arrows) pushes unlabeled target domain data (empty circles and squares), towards data it shares most similar features with (both source and target data, which are stored in the bank). Finally entropy loss (empty violet arrows) uses entropy to push away unknown classes further away from the known ones. In this \textit{Open DA} example the unknown class is present in the target domain (empty squares.)</sub>
</div>
<br>
<br>
<br>
<br>

### Training
Explanations of how to run different versions of training, evaluation as well as plotting are given in this example notebook: 
```
.......ipynb 
```

### Requirements
This code was developed using Pytorch .... The requirements.txt file lists all Python libraries that your notebooks will need, and they can be installed using:
```
pip install -r requirements.txt
```

### Authors
- Aleksandra Ćiprijanović
- Ashia Lewis

### References
If you use this code, please cite our paper: [arXiv:....](.....)
