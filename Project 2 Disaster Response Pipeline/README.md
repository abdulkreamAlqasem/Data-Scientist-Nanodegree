# Installations

* Python 3.6.3
* Pandas 0.23.3
* scikit-learn 0.19.1
* Flask 0.12.5
* numpy 1.12.1
* plotly 2.0.15
* SQLAlchemy 1.2.19
* pickle 0.7.4



# Project Motivation 
The project talks about predict disaster  from real messages that were sent during disaster event. the dataset given from [Appen](https://appen.com) ( formally Figure 8 ). i will create a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

The project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data that can help the interested people.


the project will help me to know how to build a machine learning pipeline and build ETL pipeline.


# File Descriptions
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|-DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
```



# The Journey
In this project, i go through the journey of build a machine learning pipeline, ETL pipeline (Extract,Transform,Load) and a small web app with flask where an emergency worker can input a new message and get classification results in several categories, and some visulization help the interested people.

The project will save the dataset on dataset after do some cleaning step for message data, then the cleaning data will go to machine learning pipeline to train and do optmization approach using grid search way, the accuracy of model was 63%, I face a  problem, which is that the data is imbalanced in the target feature.
After try the model we will save it in pickle file.

# How to run 
```
# step to run the project 

# to run ETL pipeline
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db 

# to run Machine learning pipeline 
python train_classifier.py ../data/DisasterResponse.db classifier.pkl

# run the web app 
python run.py

# then open http://0.0.0.0:3001/

```



# Acknowledgements
I would to thank Udacity for help me build knowledge in Data science field by do this project on Data Scientist Nano-degree Certificate.

