# import libraries
import pandas as pd
import re
import sys
import string
from sqlalchemy import create_engine
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# read messages and categories datasets and combine them
def load_data(messages_filepath, categories_filepath):
    """
    Load data and combine it to be one dataframe

    INPUT:
    messages_filepath: Path of messages.csv
    categories_filepath: Path of categories.csv

    OUTPUT:
    New dataframe of the two sub dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    categories.drop("id",axis=1,inplace=True)
    return pd.concat([messages,categories],axis=1,join="inner")

def clean_data(df):
    """
    Clean the data that include build new dataframe,
     remove dublicaties, remove inneeded columns,
     clean the message which include:
     -normlize text
     -removing punctuations
     -removing repeating characters
     -remove stop words
     -remove number
     -remove outliers

    INPUT:
    df: Orginal dataframe

    OUTPUT:
    clean and ready dataframe.
    """
    row = df["categories"][0]
    categories = df.categories.str.split(";",expand=True)
    categories.columns = [value for value in row.split(";")]

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : int(x[-1]))

    categories.columns = [value[:-2] for value in list(categories.columns)]

    df.drop("categories",axis=1,inplace=True)
    categories = categories.applymap(encode)

    df = pd.concat([df,categories],axis=1,join="inner")

    # remove dublicate and reset index
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True)
    df.drop("index",axis=1,inplace=True)

    # remove inneeded columns
    df.drop(["id","original"],axis=1,inplace=True)

    # normlize text
    df["message"] = df["message"].str.lower()

    #removing punctuations
    df["message"] = df["message"].apply(lambda x: cleaning_punctuations(x))

    #removing repeating characters
    df["message"] = df["message"].apply(lambda x: cleaning_repeating_char(x))

    # remove stop words
    df["message"] = df["message"].apply(lambda text: remove_stopwords(text))

    df["len_message"] = df["message"].apply(lambda x : len(x))

    df = tukey_rule(df, "len_message")
    df = df.drop("len_message",axis=1)

    return df

def tukey_rule(data_frame, column_name):
    """
    Load data and combine it to be one dataframe

    INPUT:
    data_frame: dataframe
    column_name: column name to do tukey rule

    OUTPUT:
    New dataframe with remove outliers
    """
    
    Q1 = data_frame[column_name].quantile(0.25)
    Q3 = data_frame[column_name].quantile(0.75)
    IQR = Q3 - Q1
    max_value = Q3 + 1.5 * IQR
    min_value = Q1 - 1.5 * IQR

    return data_frame[(data_frame[column_name] >= min_value) & (data_frame[column_name] <= max_value)]


def encode(row):
    """
    Remove number above 1 in final dataframe to make it ready for machine learning algorithm.

    INPUT:
    row: row of the dataframe

    OUTPUT:
    new value of row with number equal to 1 or 0 only.
    """

    res=0
    if row > 0 :
        res = 1
    return res


def remove_stopwords(text):
    """
    Custom function to remove the stopwords

    INPUT:
    text: text of message to remove stop word of it

    OUTPUT:
    new text with no stopwords.
    """
    STOPWORDS = set(stopwords.words('english'))
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


def cleaning_punctuations(text):
    """
    Custom function to remove the punctuations

    INPUT:
    text: text of message to remove punctuations of it

    OUTPUT:
    new text with no punctuations.
    """
    english_punctuations = string.punctuation
    punctuations_list = english_punctuations
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)

def cleaning_repeating_char(text):
    """
    custom function to remove the repeating character

    INPUT:
    text: text of message to remove repeating character of it

    OUTPUT:
    new text with no repeating character.

    """
    return re.sub(r'(.)1+', r'1', text)

def save_data(df, database_filename):
    """
    save the dataframe to database

    INPUT:
    df: dataframe
    database_filename: path of database store

    OUTPUT:
    save dataframe to database
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Clean_Data', engine, index=False,if_exists="replace")

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
