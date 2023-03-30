![Ciencias_Logo_Azul-01](https://user-images.githubusercontent.com/106987072/228209396-a8737601-f28f-486e-8566-918709663369.png)


# Advanced Machine Learning
This repository encompasses the projects developed for the Advanced Machine Learning course, which aimed to deepen the knowledge of more Machine Learning models and dwell into the world of Deep Learning and Reinforcement Learning.


#### Team:
- André Dias
- Tiago Rodrigues

#### Professors: 
- Luís Correia
- Helena Aidos


## First Assignment - Neural Networks
This assignment focused on understanding the backpropagation algorithm and the implementation of a Neural Network with distinct activation functions and momentum values.

### Models used:
- Neural Networks

### Problems:
The problems were adapted from [Neural Networks and Learning Machines](https://books.google.pt/books/about/Neural_Networks_and_Learning_Machines.html?id=KCwWOAAACAAJ&redir_esc=y). The problems proposed were adapted from problem 4.1 and 4.2 of the book.

### Assignment Objectives:
- First exercise:
  - Determine separation planes of the proposed Neural Network
  - Obtain truth table
  - Consider a sigmoid activation function and calculate 1 iteration of the error backpropagation algorithm
- Second exercise:
  - Implement the Neural Network proposed and test different activation functions and momentum values
  - Implement the Neural Network of the first exercise and compare results


## Second Assignment - Support Vector Machines
This assignment focused on understanding the effect of support vectors, the optimization of the SVMs parameters and data transformation to obtain linear separation of data. 

### Models used:
- Support Vector Machines (Kernels: linear, poly, sigmoid and rbf)

### Problems:
The problems were adapted from [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/). The first problem focused on studying the effect of support vectors on the decision boundary; the second focused on tunning the SVM parameters and the third on data transformation to obtain linear separation of data.

### Data:
The data used was the well known [Iris Dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html).

### Assignment Objectives:
- First exercise:
  - Train a poly SVM on the dataset
  - Remove some support vectors and evaluate how it affects the decision boundary
- Second exercise:
  - Optimize the hyparameters of the SVM
- Third exercise:
  - Find a data transformation that leads to a linear separable problem
  - Visualize data in 3D using the extra feature from the data transformation
  - Use different kernels and evaluate which one leads to better performance on the original dataset
  
  
## Third Assignment - Mixture Models
This assignment focused on understanding the effect of using several simple models to obtain a better overall performance (Ensemble).

### Models used:
- Decision Tree Classifiers
- K-Nearest Neighbours
- Naïve Bayes
- Logistic Regression


### Problems:
The problem focused on the implementation of a Weighted Average Ensemble System in Python, which combines the output of several experts with a linear combination, whose weights are the accuracy scores of the experts on the dataset. This output is then rounded to the nearest integer.

### Data:
The data used was the well known [Breast Cancer dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).

### Assignment Objectives:
- Implement the Weighted Average Ensemble System and test distinct combinations of models for the ensemble



## Fourth Assignment - Hidden Markov Models
This assignment focused on understanding the impact of the amount of data on the quality of the model when some of the information is not supplied (emission and transmission matrixes, number of states, etc).

### Models used:
- Hidden Markov Models

### Problems:
The problem focused on a dishonest casino, similar to the one seen [here](https://hmmlearn.readthedocs.io/en/latest/auto_examples/plot_casino.html#sphx-glr-auto-examples-plot-casino-py). From here, a truth model was built (with all parameters known) and later other models that estimated some of the parameters.

### Data:
The data used changed in each question. For the first and second, a small sequence of the rolls and dice states was provided. For the third, the dice.txt and rolls.txt was used. For the last, data was generated using the model in the first exercise.

### Assignment Objectives:
- First exercise:
  - Train a HMM with all parameters known
- Second exercise:
  - Train a HMM assuming the transition matrix is unknown
- Third exercise:
  - Train a HMM assuming the transition and emission matrixes are unknown
  - Train a HMM assuming all parameters are unknown
- Fourth exercise:
  - Sample the model in the first exercise
  - Redo the third exercise with this data
  
  
## Fifth Assignment - Bayesian Networks
This assignment focused on modelling a Bayesian Network and comparing Naïve Bayes with Random Forests.

### Models used:
- Bayesian Networks
- Naïve Bayes
- Random Forest Classifiers

### Problems:
The problem was adapted from [IDA/AIICS Course on Artificial Intelligence and Lisp](https://www.ida.liu.se/ext/caisor/TDDC65/dectree-exercise/page-100930.html), which focused on the development of a Bayesian Network and the computation of conditional probabilities. The second part focused on the comparison of the performance of Naïve Bayes and Random Forests.

### Data:
For the Bayesian Network, the data utilized was the one given in the mentioned problem. For the second part, the well known [Breast Cancer dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) was used.

### Assignment Objectives:
- First exercise:
  - Create a Bayesian Network
  - Calculate conditional probabilities
- Second exercise:
  - Train a Naïve Bayes and Random Forest Classifier, optimize their parameters and compare results
