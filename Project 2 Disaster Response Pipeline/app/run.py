import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Clean_Data', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # second chart

    # count how many label in message
    df['count_ones_'] = df.drop(["message","genre"],axis=1).sum(axis=1)
    count_name = df['count_ones_'].value_counts().sort_values(ascending=False).keys()
    count_value = df['count_ones_'].value_counts().sort_values(ascending=False).values

    # thrid chart
    # length of each tweet
    df["len_message"] = df["message"].apply(lambda x : len(x))
    count_len_name = df['len_message'].value_counts().sort_values(ascending=False).keys()
    count_len_value = df['len_message'].value_counts().sort_values(ascending=False).values

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=count_name,
                    y=count_value
                )
            ],

            'layout': {
                'title': 'Distribution of Labels in each Message',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "# Labels"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=count_len_name,
                    y=count_len_value
                )
            ],

            'layout': {
                'title': 'Distribution of Message Length',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Length of Message"
                }
            }
        }

    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
