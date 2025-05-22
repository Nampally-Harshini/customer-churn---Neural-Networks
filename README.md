Customer Churn Prediction
Project Overview
This project predicts customer churn for a bank using a range of machine learning models, including neural networks. 
The dataset contains customer demographic and account information, and the goal is to identify customers likely to exit the bank based on their profile and activity.

Models and Evaluation
The following models are implemented and compared:
Decision Tree
Random Forest
AdaBoost
XGBoost
Voting Classifier
Artificial Neural Networks (ANNs)

Performance metrics include Train Score, Test Score, F1 Score, Recall, and Precision.
The Voting Classifier achieved the highest test accuracy among the traditional models, with a Test Score of 0.8205

Artificial Neural Networks (ANNs) are implemented to predict customer churn, leveraging their ability to model complex nonlinear relationships in the data.
The ANN model is built using a sequential architecture with input, hidden, and output layers, typically using ReLU activation for hidden layers and sigmoid activation for the output layer. The model is trained using optimizers like Adam and evaluated with metrics such as accuracy, precision, recall, and F1-score.
In comparative studies, ANNs have shown higher accuracy than traditional models like logistic regression for bank churn prediction, achieving accuracy rates around 86%.
Advanced neural network architectures, such as Convolutional Neural Networks (CNNs) and hybrid models (e.g., BiLSTM-CNN), have been explored in recent research and have demonstrated even higher accuracy, especially when extracting complex patterns from customer data.
