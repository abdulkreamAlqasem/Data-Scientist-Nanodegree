from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('templates/Titanic_model.pkl')

# Define the homepage route
@app.route('/')
def index():
    return render_template('index.html')

# Define the route that handles form submission and displays the prediction results
@app.route('/predict', methods=['POST'])
def predict():


    ticket_class = int(request.form['ticket_class'])
    gender = 0 if request.form['gender'] == 'male' else 1

    embarked = request.form['Embarked']
    if embarked == "3":
        embarked = [1,0]
    elif embarked == "1":
        embarked = [0,1]
    else:
        embarked = [0,0]

    age = float(request.form['age'])

    fare = float(request.form['fare'])

    is_alone = 1 if request.form['is_alone'] == "yes" else 0

    title_name = request.form['Title_Name']
    if title_name == "1":
        title_name = [0,1,0,0]
    elif title_name == "2":
        title_name = [1,0,0,0]
    elif title_name == "3" :
        title_name = [0,0,1,0]
    elif title_name == "4" :
        title_name = [0,0,0,1]
    else:
        title_name = [0,0,0,0]

    data = [[ticket_class, gender, age, fare,is_alone, title_name, embarked ]]

    # loop over each element of the output_list and modify it in-place
    for i in range(len(data)):

        # extract the sublists at the end of the element
        sublist1, sublist2 = data[i][-2:]

        # flatten the sublists and insert the flattened values into the element
        data[i] = data[i][:5] + sublist1 + sublist2

    prediction = model.predict(data)[0]

    prediction_result = 'Sorry, you did not survive the Titanic disaster.' if prediction == 0 else 'Congratulations, you survived the Titanic disaster!'

    # Render the predict.html template with the prediction result
    return render_template('predict.html', prediction_result=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)
