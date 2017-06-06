import pandas as pd
import numpy as np
import datetime
import json
import random
import cPickle as pickle
from RandomForest_fraud_detection import RFmodel
import urllib2
# pd.set_option('display.max_columns', None)


class PredictFraud(object):
    '''
    Reads in a single example from test_script_examples, unpickles the model, predicts the
    label, and outputs the label probability
    '''

    # data_source=["local", "url", 'post']
    def __init__(self, model_path, example_path, url_path, data_source='local'):
        self.model_path = model_path
        self.example_path = example_path
        self.url_path = url_path
        self.data_source = 'url'

    def read_entry(self):
        '''
        Read single entry from http://galvanize-case-study-on-fraud.herokuapp.com/data_point
        '''
        if self.data_source == 'url':
            response = urllib2.urlopen(self.url_path)
            d = json.load(response)
        # elif self.data_source == 'post':
        #     d = data
        elif self.data_source == 'local':
            df_frauder = pd.read_csv(self.example_path, index_col=False)
            df_frauder = df_frauder[df_frauder.iloc[:, 0] != 'acct_type']
            df_frauder.reset_index(inplace=True, drop=1)
            d = dict(df_frauder.values)
        else:
            with open(self.example_path) as data_file:
                d = json.load(data_file)
        df = pd.DataFrame()
        df_ = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.iteritems() if (
            k != 'ticket_types') and (k != 'previous_payouts')]))
        df_['ticket_types'] = str(d['ticket_types'])
        df_['previous_payouts'] = str(d['previous_payouts'])
        df = df.append(df_)
        df.reset_index(drop=1, inplace=True)
        df.fillna(0, inplace=True)
        self.example = df
        return df

    def fit(self):
        '''
        Load model with cPickle
        '''
        self.read_entry()
        md = RFmodel()
        X_prep = md.prepare_data(self.example)
        self.df = md.df
        return X_prep

    def predict(self):
        return self.model.predict_proba(self.X_prep)


if __name__ == '__main__':

    # Data
    example_path = '../../data/test_script_example.json'
    example_path = '../../data/data.json'
    example_path = 'ex2.csv'
    url_path = 'http://galvanize-case-study-on-fraud.herokuapp.com/data_point'
    model_path = '../../data/model.pkl'

    # Test with one fraud
    test = pd.read_csv(example_path)
    md = RFmodel()
    X_prep = md.prepare_data(test, y_name=False)
    df = md.df
    model = pickle.load(open(model_path, 'rb'))
    model.predict_proba(X_prep)

    # Test with URL data_source=url (params are:'fraud', 'df', 'url')
    md = RFmodel()
    mdPred = PredictFraud(
        model_path, example_path, url_path, data_source='url')
    df = mdPred.read_entry()
    X_prep = md.prepare_data(df, y_name=False)
    model = pickle.load(open(model_path, 'rb'))
    model.predict_proba(X_prep)

    # X_prep = mdPred.fit()
    # mdPred.df
    # model = pickle.load(open(model_path, 'rb'))
    # model.predict_proba(X_prep)
    # y_pred = model.predict(X_prep)
