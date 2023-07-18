# Data-Science-Capstone-Project

# Installations

* Python 3.9.15
* Pandas 1.5.3
* scikit-learn 1.1.3
* Flask 2.3.2
* numpy 1.23.4
* seaborn 0.12.2
* matplotlib 3.7.1
* pickle 0.7.4



# Project Motivation 

The Titanic dataset is a well-known dataset that provides information about the passengers onboard the Titanic ship. It is a popular dataset in the machine learning community and can be used as a beginner-level project to learn about data analysis, data preprocessing, and model building techniques. The goal of this project is to build a classification machine learning model that can predict the survival of passengers based on their characteristics. The project aims to provide insights into the factors that contributed to the survival of passengers, which can be useful in future disaster planning.

I do it to be refrence for anyone want to learn who to build machine learning project.


# File Descriptions
```
- app
| - template
| |- index.html # main page of web app
| |- predict.html  # page that show the result
| |- Titanic_model.pkl  # saved model 
|- app.py  # Flask file that runs app

- data
|- train.csv  # train data to train the model
|- test.csv  # test data to test the model 

- Titanic project (baseline) # notebook of the project

- README.md
```



# The Journey
The journey of this project on Titanic dataset starts with exploratory data analysis (EDA). EDA involves exploring the dataset to understand its structure and characteristics. The data preprocessing step involves handling missing data, outlier rows, and transforming skewed columns to improve the performance of the model. Feature engineering and encoding are used to create new features and convert categorical variables to numerical variables. The model selection step involves choosing an appropriate algorithm and tuning its hyperparameters to improve its performance. The evaluation step involves testing the model to know if its good or no (i get 75% accuracy). The project can be useful for anyone interested in learning about the journey of building a classification machine learning model and can provide insights into the importance of data preprocessing, feature engineering, and model selection.



# How to run web app
```
# run the web app 
python run.py

# then open http://127.0.0.1:5000/

```


# Acknowledgements
I would like to thank Udacity for help me build knowledge in Data science field by do this project on Data Scientist Nano-degree Certificate and also i would to thank Kaggle to provide this dataset.

