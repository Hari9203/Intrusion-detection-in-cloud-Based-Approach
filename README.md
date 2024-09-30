# Intrusion-detection-in-cloud-Based-Approach

## Convolutional Neural Network (CNN) with Stochastic Gradient Descent (SGD)

### Dataset
The dataset used for this project can be found at the following link: https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv.
This dataset contains various features related to network traffic, along with labels indicating whether the traffic is normal or represents an intrusion attempt.

### Key Features
Traffic Features: Includes metrics such as packet sizes, protocol types, and connection status.
Attack Labels: The dataset categorizes traffic into different attack types, allowing for targeted analysis and model training.


### Model Overview
This project employs a Convolutional Neural Network (CNN) architecture to detect intrusions in network traffic data. The CNN model is well-suited for this task due to its ability to learn hierarchical feature representations from raw input data automatically.

Stochastic Gradient Descent (SGD)
We utilize the Stochastic Gradient Descent (SGD) optimization algorithm to optimise the CNN model. SGD is a popular optimization technique for training deep learning models, as it updates model weights iteratively based on a small batch of data, allowing for faster convergence.

### Benefits of Using CNN with SGD

Efficiency: SGD processes only a subset of data at each iteration, which reduces computation time and memory usage.

Generalization: The stochastic nature of the updates helps prevent overfitting, enabling the model to generalize better to unseen data.

Adaptability: SGD allows for dynamic learning rates, which can be adjusted during training to improve performance.

#### Training the Model
The model is trained using the dataset referenced above, and the following steps are followed:

Data Preprocessing: Clean and prepare the dataset for input into the CNN model.

Model Architecture: Define the CNN architecture with convolutional, pooling, and fully connected layers.

Compilation: Compile the model using SGD as the optimizer, specifying a suitable loss function and metrics.

Training: Fit the model to the training data while monitoring validation performance to ensure proper convergence.

#### Results
The performance of the CNN model with SGD will be evaluated based on metrics such as accuracy, precision, recall, and F1-score, with results visualized in graphs for better understanding.
