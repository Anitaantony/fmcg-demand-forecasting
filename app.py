#app.py

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)

from flask import jsonify

@app.route('/sales-data')
def sales_data():
    return jsonify({
        "dates": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
        "quantities": [120, 135, 200, 180]
    })
