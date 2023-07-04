# import libraries

import nltk
import sys
nltk.download('punkt')
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import RidgeClassifier
import pickle




def load_data(database_filepath):
    """
    Load the dataframe from database

    INPUT:
    database_filepath: database file path

    OUTPUT:
    X: messages to train the model
    y: target categories of each message
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table("Clean_Data",engine)
    X = df["message"].values
    y = df.iloc[:,2:].values
    category_names = df.iloc[:,2:].columns
    return X,y,category_names


def tokenize(text):
    """
    Tokenize each message to spreate words and do some cleaning (lemmatizion, lower case , remove spaces)

    INPUT:
    text: message to be tokenize it

    OUTPUT:
    list of clean words
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens




def build_model():

        """
        Build model using Pipeline to make it simple,
        and grid search to find best parameters in one function.

        INPUT:
        No input

        OUTPUT:
        Final model.
        """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('nlp_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
        ])),
        ("rcf",MultiOutputClassifier(estimator=RidgeClassifier(solver="lsqr")))
        ])

    # Define the parameter grid to search over
    parameters = {
        'rcf__estimator__alpha': [0.001,0.01,0.1, 1.0 ],
        'rcf__estimator__solver': ['auto', 'svd', 'lsqr', 'sag', 'saga']
   }

    cv = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1,verbose=4)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model by print Classification Report

    INPUT:
    model: The model to evaluate
    X_test: Message to test the model
    Y_test: Target of message to test the model
    category_names: Categories name.

    OUTPUT:
    No return only print the result of the model.
    """
    result_model = model.predict(X_test)
    report = classification_report(Y_test, result_model, target_names=category_names)
    print("Classification Report:\n", report)


def save_model(model, model_filepath):
    """
    Save the model as pickle file

    INPUT:
    model: The model to evaluate
    model_filepath: file path of model

    OUTPUT:
    No return only save the model to the model_filepath argument.
    """
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
