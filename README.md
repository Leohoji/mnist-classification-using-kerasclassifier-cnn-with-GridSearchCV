<h1>MNIST classification using CNN applying gridsearch</h1>

**Notebooks Here**  --> [CNN-keras-mnist-gridsearch.ipynb](https://github.com/Leohoji/mnist-classification-using-kerasclassifier-cnn-with-GridSearchCV/blob/main/CNN-keras-mnist-gridsearch.ipynb)

Use GridSearchCV with KerasClassifier to search best hyperparameters in simple CNN model trained on MNIST database.

**Techniques**: Computer Vision, Classification, GridSearchCV, KerasClassifier

<h2>Prerequisites</h2>

- python 3.8

- tensorflow 2.5.0+ 【 If you have GPU to accelerate, you could install tensorflow-gpu 2.5.0 with CUDA 11.2 and cuDNN 8.1 】

- Others in `requirements.txt`

💻 The GPU I use is **GeForce GTX 1050 Ti**

<h2>How Do I Complete This Project</h2>

### Summary of experimental process
<p align='left'>
  <img alt="process of project" src="https://github.com/Leohoji/projects_for_hyperparameters_searching/blob/main/introduction_images/process_for_cnn_gridsearch.png?raw=true" width=700 height=400>
</p>

### Hyperparameters to be searched
| Hyperparameters | Values to search |
| -- | -- |
| Activation functions | Sigmoid, ReLU, LeakyReLU, PReLU, tanh |
| Loss functions | MSE, Cross-Entropy |
| Batch size | 8, 32, 128 |

Epoch is fixed as **5** and Optimizer as **Adam** for convenience of experimentation.

### Process of experiment
1. Create helper functions and import necessary libraries.
2. Load MNIST database and preprocess it, including create training and testing data.
3. Use KerasClassifier wrapper to pacckage CNN model.
4. Pass the wrapped model and hyperparameters expected to searched into GridSearchCV API.
5. Analize the results and conclude the experiment.

<h2> Results </h2>

Finally find some conditions that can make the model peform better, following conditions are the hyperparameters options that make the accuracy better than 90% within this project.

| Hyperparameters      | Better Performance   |
| -------------------- | -------------------- |
| Batch Size           | 8, 32                |
| Activation Functions | ReLU or Tanh         |
| Loss functions       | MSE or Cross-Entropy |

> [Back to outlines](#main-title)
