from flask import Flask, request, jsonify
import csv

app = Flask(__name__)

@app.route("/portfolio", methods=['POST'])
def portfolio():
    re = request.get_json()
    if re['portfolio_id'] == 'balance':
        tagName = 'Balanced'
    if re['portfolio_id'] == 'aggressive':
        tagName = 'Defensive'
    amount = 0.0
    if re['amount']:
        try:
            amount = float(re['amount'])
        except ValueError:
            amount = 0.0

    if 'tagName' not in locals():
        return {
            'error': 'portfolio_id name error'
        }, 400
    coin_list = list()
    with open('portfolio_weight.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            coin_list.append({
                'symbol': row['symbol'],
                'percentage': round(float(row[tagName]), 6),
                'amount': amount * round(float(row[tagName]), 6)
            })
    return {
        'data': {
            'portfolio_id': re['portfolio_id'],
            'coin_list': coin_list
        }
    }

if __name__ == '__main__':
    app.run()