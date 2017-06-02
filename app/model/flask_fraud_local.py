from flask import Flask, request
from sqlalchemy import create_engine
import psycopg2
import pandas as pd
import numpy as np
import datetime
import json
import random
import cPickle as pickle
import sys
import os
sys.path.insert(0, './model')
from predict import PredictFraud
from RandomForest_fraud_detection import RFmodel
from pandas.io import sql
import urllib2
# pd.set_option('display.max_columns', None)
app = Flask(__name__)


def create_table(cur):
    cur.execute(
        '''CREATE TABLE fraud (
            id serial PRIMARY KEY,
            approx_payout_date integer,
            body_length integer,
            channels integer,
            country text,
            currency text,
            delivery_method double precision,
            description text,
            email_domain text,
            event_created integer,
            event_end integer,
            event_published integer,
            event_start integer,
            fb_published integer,
            gts double precision,
            has_analytics integer,
            has_header double precision,
            has_logo integer,
            listed text,
            name text,
            name_length integer,
            num_order integer,
            num_payouts integer,
            object_id integer,
            org_desc text,
            org_facebook double precision,
            org_name text,
            org_twitter double precision,
            payee_name text,
            payout_type text,
            sale_duration double precision,
            sale_duration2 double precision,
            show_map integer,
            user_age integer,
            user_created integer,
            user_type integer,
            venue_address text,
            venue_country text,
            venue_latitude double precision,
            venue_longitude double precision,
            venue_name text,
            venue_state text,
            ticket_types text,
            previous_payouts text,
            fraud_probability double precision);
        ''')
    conn.commit()
    conn.close()


def insert_db(df, engine, table='fraud'):
    df.to_sql(table, engine, if_exists='append', index=0)


def format_data(staging=False):
    example_path = 'ex2.csv'
    url_path = 'http://galvanize-case-study-on-fraud.herokuapp.com/data_point'
    model_path = '../../data/model.pkl'
    if staging:
        # For local testing
        md = RFmodel()
        df_full = pd.read_csv(example_path)
        test = pd.read_csv(example_path)
        X_prep = md.prepare_data(test, y_name=False)
        df = md.df
    else:
        md = RFmodel()
        Pred = PredictFraud(model_path, example_path, url_path)
        df_full = Pred.read_entry()
        X_prep = md.prepare_data(df_full, y_name=False)
        df = md.df
    return df_full, df, X_prep


def make_prediction(df, X_prep):
    '''
    Make a prediction and save it in the psql db
    '''
    engine = create_engine(
        'postgresql://aymericflaisler:1323@localhost:5432/fraud_prediction')
    # do the prediction
    model_path = '../../data/model.pkl'
    model = pickle.load(open(model_path, 'rb'))
    y_pred = model.predict_proba(X_prep)[0, 1]
    df['fraud_probability'] = y_pred
    # insert_db(df, engine, table='fraud')
    # If prediction < 0.17: low
    # If prediction < 0.50: medium
    if y_pred > .5:
        risk_band = "High"
    elif (y_pred > .17) and (y_pred < .5):
        risk_band = "Medium"
    else:
        risk_band = "Low"
    return df, X_prep, y_pred, risk_band

# Flask can respond differently to various HTTP methods
# By default only GET allowed, but you can change that using the methods
# argument


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        json_dict = request.get_json()
        event_start = json_dict['event_start']
        # json.load(open(example_path))['description']
        data = {'event_start': event_start}
        return "from json: " + str(data)
    else:
        # response = urllib2.urlopen(
        #     'http://galvanize-case-study-on-fraud.herokuapp.com/data_point')
        # raw_json =  json.load(response)
        i = 0
        while i < 200:
            df_full, df_, X_prep = format_data(staging=False)
            # print X_prep
            df, X, y_pred, risk_band = make_prediction(
                df_full, X_prep)
            print y_pred, risk_band
            i += 1

        return "Event Name: " + df.name.to_string(index=0) + "<br>" + "Venue Name: " + df.venue_name.to_string(index=0) + "<br>" + " Prediction: " + \
            str(y) + "<br>" + "Risk band: " + risk_band + "<br>" + df.to_html()


if __name__ == "__main__":
    # AWS
    # app.run(host='0.0.0.0', debug=True)
    # local server
    port = 8000

    # Open a web browser pointing at the app.
    os.system('open http://localhost:{0}'.format(port))

    # Set up the development server on port 8000.
    app.debug = True
    app.run(port=port)