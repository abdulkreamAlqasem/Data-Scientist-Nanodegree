
# Installations
* Python 3.9.16 
* Pandas 1.5.3
* matplotlib 3.7.1
* scikit-learn 1.2.2
* lightgbm 3.3.5
# Project Motivation
The project talks about how to predict the job of anyone who do **stack overflow annual developer survey** based on his characteristics. The project will help me build a machine learning model with real data and solve the problem with it. I am interested in this project because it will make me comfortable building any machine learning model.

# File Descriptions
I divided the Stack Overflow Annual Developer Survey Model file into nine parts that will help you go to any part you want:
## Parts 
1. Import important libraries: import the libraries necessary for the project.
2. Read data: read the required data.
3. Exploraity Data Analysis: EDA for Post in Blog on Medium.com.
4. Choose Features for Model: choose which feature for model.
5. Handle Missing Value: Handle the missing value in final features.
6. Split the dataset to train and test the model.
7. Feature Encoding: Feature encoding for categorical features
8. Feature Scaling: Feature Scaling for Numeric Features
9. Model: build a model and see the performance of the model.
# How to Interact with Your Project
In this project, i go through the journey of a machine learning model, read data and handle missing values EDA, clean the data, do feature encoding and scaling, and build a machine learning model.

For handling missing values, I use the mean for numeric missing values and the mode for categorical missing values. Also, I face two features that must be numeric instead of object features, so I change them.

In feature encoding, I do two things: Label encoding and One-Hot Encoding. Label encoding for features with two values and for target feature .
One-hot encoding for features with more than two values. I use these two approaches to make the number of features smaller than when using only One-hot encoding.

I use MinMaxScaler scaling to make all values between 0 and 1.

I chose the LGBMClassifier model from lightgbm for this project as the baseline, and the result was %84. I face another problem, which is that the data is imbalanced in the target feature.
# Acknowledgements
I wolud to thank Udacity for help me build knowleg in Data science feild by do this project on Data Scientist Nanodegree Certifacte.  
