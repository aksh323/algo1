from flask import Flask, jsonify, request
from threading import Thread
import time
from datetime import datetime, timedelta
import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Initialize live data for multiple intervals and tickers
live_data = {
    '5': {'current': 'data will be available', 'predicted': 'data will be available'},
    '15': {'current': 'data will be available', 'predicted': 'data will be available'},
    '30': {'current': 'data will be available', 'predicted': 'data will be available'},
    '60': {'current': 'data will be available', 'predicted': 'data will be available'}
}

def fetch_and_predict(ticker, interval):
    headers = {'Content-Type': 'application/json'}
    end_string = str(datetime.now()).split()[0]
    end_object = datetime.strptime(end_string, "%Y-%m-%d")
    current_date = end_object - timedelta(days=60)
    start_string = current_date.strftime("%Y-%m-%d")
    
    request_url = f"https://api.tiingo.com/iex/{ticker}/prices?startDate={start_string}&resampleFreq={interval}min&columns=open,high,low,close,volume&token=7d4174a2c32587ce3224af1365332f411d56e6bf"
    request_response = requests.get(request_url, headers=headers)
    elevations = request_response.json()
    df = pd.DataFrame(elevations)
    data_uni = df[['close']]
    target = data_uni.values
    scaler = MinMaxScaler()
    target_scaled = scaler.fit_transform(target.reshape(-1, 1))

    total_number = len(target_scaled)
    last_values = target_scaled[total_number - 101:]

    base_path = r"Algorithm/"
    new_model = tf.keras.models.load_model(base_path + f'models/{ticker}/{ticker}_{interval}.h5')

    all_temp_value = []
    temp_value = last_values.tolist()
    for i in temp_value:
        all_temp_value.append(i[0])

    lst_output = []
    n_steps = 100
    i = 0
    while i < 15:
        if len(all_temp_value) >= 100:
            x_input = np.array(all_temp_value[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = new_model.predict(x_input, verbose=0)
            all_temp_value.extend(yhat[0].tolist())
            all_temp_value = all_temp_value[1:]
            lst_output.extend(yhat.tolist())
            i += 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = new_model.predict(x_input, verbose=0)
            all_temp_value.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i += 1

    predict_values = scaler.inverse_transform(lst_output)
    predict_values = [i[0] for i in predict_values.tolist()]
    predict_value = round(predict_values[0], 2)
    live_data[interval]['predicted'] = predict_value

    now_date = str(datetime.now()).split()[0]
    current = requests.get(
        f"https://api.tiingo.com/iex/{ticker}/prices?startDate={now_date}&resampleFreq={interval}min&columns=open,high,low,close,volume&token=7d4174a2c32587ce3224af1365332f411d56e6bf", headers=headers)
    try:
        live_data[interval]['current'] = current.json()[-1]['close']
    except:
        live_data[interval]['current'] = request_response.json()[-1]['close']

@app.route('/api/data1', methods=['GET'])
def get_data():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400
    
    intervals = ['5', '15', '30', '60']
    threads = []
    for interval in intervals:
        thread = Thread(target=fetch_and_predict, args=(ticker, interval))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    return jsonify(live_data)

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0", port=64007)
