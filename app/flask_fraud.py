from flask import Flask, request, render_template
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
# from predict import PredictFraud
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


def prepare(url="http://galvanize-case-study-on-fraud.herokuapp.com/data_point"):
    '''
    Read single entry from http://galvanize-case-study-on-fraud.herokuapp.com/data_point
    '''
    response = urllib2.urlopen(url)
    d = json.load(response)
    df = pd.DataFrame()
    df_ = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.iteritems() if (
        k != 'ticket_types') and (k != 'previous_payouts')]))
    df_['ticket_types'] = str(d['ticket_types'])
    df_['previous_payouts'] = str(d['previous_payouts'])
    df = df.append(df_)
    df.reset_index(drop=1, inplace=True)
    df.fillna(0, inplace=True)
    return df


def insert_db(df, engine, table='fraud'):
    df.to_sql(table, engine, if_exists='append', index=0)


def format_data(staging=False):
    example_path = 'ex2.csv'
    url_path = 'http://galvanize-case-study-on-fraud.herokuapp.com/data_point'
    model_path = '../data/model.pkl'
    if staging:
        # For local testing
        md = RFmodel()
        df_full = pd.read_csv(example_path)
        test = pd.read_csv(example_path)
        X_prep = md.prepare_data(test, y_name=False)
        df = md.df
    else:
        md = RFmodel()
        df_full = prepare(url_path)
        X_prep = md.prepare_data(df_full, y_name=False)
        df = md.df
    return df_full, df, X_prep


def risk_band_(y_pred):
    # If prediction < 0.17: low
    # If prediction < 0.50: medium
    if y_pred > .5:
        risk_band = "High"
    elif (y_pred > .17) and (y_pred <= .5):
        risk_band = "Medium"
    else:
        risk_band = "Low"
    return risk_band


def make_prediction(df, X_prep):
    '''
    Make a prediction and save it in the psql db
    '''
    engine = create_engine(
        'postgresql://aymericflaisler:1323@localhost:5432/fraud_prediction')
    # do the prediction
    model_path = '../data/model.pkl'
    model = pickle.load(open(model_path, 'rb'))
    y = model.predict_proba(X_prep)
    y_pred = model.predict_proba(X_prep)[0, 1]
    df['fraud_probability'] = y_pred
    insert_db(df, engine, table='fraud')
    risk_band = risk_band_(y_pred)
    return df, X_prep, y_pred, risk_band, y


# Flask can respond differently to various HTTP methods
# By default only GET allowed, but you can change that using the methods
# argument


@app.route("/")
def my_form():
    return render_template("intro.html") + "<br>" + "<h2>Live data:</h2>" \
        + df_.to_html() + "Event Name: " + df.name.to_string(index=0) + "<br>" \
        + "Venue Name: " + df.venue_name.to_string(index=0) + "<br>" \
        + "Risk Band Prediction: " + "<font color='FF0000'>" + risk_band + "</font>" + "<br>" \
        + "Probability of Fraud: " + "<font color='FF0000'>" + str(round(y[0][1], 3) * 100) + "%" + "</font>"\
        + "<br>" + render_template("my-form.html", df_=df_)


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        df_.ach = request.form['ach']
        df_.check = request.form['check']
        df_.missing_payment = request.form['miss']
        df_.has_logo = request.form['logo']
        df_.has_analytics = request.form['analytics']
        X_prep = df_.values
        model_path = '../data/model.pkl'
        model = pickle.load(open(model_path, 'rb'))
        y2 = model.predict_proba(X_prep)
        risk_band2 = risk_band_(round(y2[0][1], 3))
        # my_form()
        return render_template("intro.html") + "<br>" + "<h2>Live data:</h2>" + df_.to_html() + "Event Name: " + df.name.to_string(index=0) + "<br>" \
            + "Venue Name: " + df.venue_name.to_string(index=0) + "<br>" \
            + "Risk Band Prediction: " + "<font color='FF0000'>" + risk_band2 + "</font>" + "<br>" \
            + "Probability of Fraud: " + "<font color='FF0000'>" + str(round(y2[0][1], 3) * 100) + "%" + "</font>"\
            + "<br>" + render_template("my-form.html", df_=df_)


if __name__ == "__main__":
    md = RFmodel()
    df_full, df_, X_prep = format_data(staging=False)
    print X_prep
    df, X, y_pred, risk_band, y = make_prediction(
        df_full, X_prep)
    print y_pred, risk_band, y
    app.run(host='0.0.0.0', debug=True)
