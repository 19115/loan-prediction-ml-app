from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import json
import pandas as pd

app = Flask(__name__)

app.secret_key = "some_secret_key"

##load model
model = pickle.load(open('loan_model.pkl', 'rb'))
le = pickle.load(open('le_grade.pkl', 'rb'))

with open('feature_columns.json', 'r') as f:
    feature_columns = json.load(f)

numeric_columns = [
    "age",
    "annual_income",
    "monthly_income",
    "debt_to_income_ratio",
    "credit_score",
    "loan_amount",
    "interest_rate",
    "loan_term",
    "installment"
]

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    grades = le.classes_
    if request.method == "POST":
        input_data = request.form.to_dict()
        df = pd.DataFrame([input_data])

        # Convert numeric columns
        for col in numeric_columns:
         if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[numeric_columns] = df[numeric_columns].fillna(0)


        ##encode grade_subgrade 
        df['grade_subgrade'] = le.transform(df['grade_subgrade'])

        ##one hot encoding
        df = pd.get_dummies(df)

        ##allignment with training data
        df = df.reindex(columns=feature_columns, fill_value=0)

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        confidence = round(probability * 100, 2)
        if prediction == 1:
         result_text = "Loan will be repaid"
         result_class = "success"
        else:
         result_text = "Loan may default"
         result_class = "warning"

        # store result in session temporarily
        session["result"] = result_text
        session["result_class"] = result_class
        session["confidence"] = confidence

        return redirect(url_for("index"))
    
    ##GET request (after redirect)
    result = session.pop("result", None)
    result_class = session.pop("result_class", None)
    confidence = session.pop("confidence", None)

    return render_template(
        "index.html",
        result=result,
        result_class=result_class,
        confidence=confidence,
        grades=grades
    )


if __name__  == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

