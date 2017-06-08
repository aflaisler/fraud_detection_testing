from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GDBC
from sklearn.ensemble import AdaBoostClassifier as ADR
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, recall_score, f1_score, precision_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import json
import random
import cPickle as pickle
pd.set_option('display.max_columns', None)


class RFmodel(object):

    def load_data(self, filepath):
        '''
        Load data from json file located in the data folder
        '''
        with open(filepath) as data_file:
            data = json.load(data_file)
        df = pd.DataFrame.from_dict(data)
        df['fraud'] = df.acct_type.map(lambda x: 1 if ((x == 'fraudster') | (
            x == 'fraudster_att') | (x == 'fraudster_event')) else 0)
        df.fillna(0, inplace=True)
        df.reset_index(drop=1, inplace=True)
        self.df = df.copy()
        return self.df

    def prepare_data(self, df, y_name=False):
        '''
        Take the dataframe with all data (X and y) and create additional dummy features
        from categorical data
        '''
        # df.drop('acct_type', inplace=True, axis=1)
        df['eur'] = df.currency.map(lambda x: 1 if (x == 'eur') else 0)
        df['gbp'] = df.currency.map(lambda x: 1 if (x == 'gbp') else 0)
        df['ach'] = df.payout_type.map(lambda x: 1 if x == 'ACH' else 0)
        df['check'] = df.payout_type.map(lambda x: 1 if x == 'CHECK' else 0)
        df['missing_payment'] = (1 - df.ach - df.check)
        df['dict_elements'] = df.previous_payouts.map(
            lambda x: 0 if '[]' else np.size(x))
        df.reset_index(drop=1, inplace=True)
        self.df_full = df.copy()
        if y_name:
            col_to_keep = ['eur', 'gbp', 'ach', 'check', 'missing_payment', 'dict_elements', 'gts',
                           'has_logo', 'user_type', 'delivery_method', 'org_facebook', 'org_twitter', 'has_analytics', 'fraud']
            df_ = df.loc[:, col_to_keep]
            df_.reset_index(drop=1, inplace=True)
            y = df_.pop(y_name).values
            X = df_.values
            X = X.astype(int)
            self.X, self.y = X, y
            self.df = df_.copy()
            return X, y
        else:
            col_to_keep = ['eur', 'gbp', 'ach', 'check', 'missing_payment', 'dict_elements', 'gts',
                           'has_logo', 'user_type', 'delivery_method', 'org_facebook', 'org_twitter', 'has_analytics']
            df_ = df.loc[:, col_to_keep]

            df_.reset_index(drop=1, inplace=True)
            X = df_.values
            X = X.astype(int)
            self.X = X
            self.df = df_.copy()
            return X

    def grid_search(self, model_type, params, X, y):
        '''
        Grid search for the data previously loaded
        '''
        model = GridSearchCV(model_type, params,
                             scoring=make_scorer(recall_score), cv=5, n_jobs=-1)
        model.fit(X, y)
        print model.best_params_
        # print model.best_score_
        self.best_est = model.best_estimator_
        self.best_params_ = model.best_params_
        self.best_score = model.best_score_

    def scores_for_best_model(self, best_est, X, y):
        '''
        Calculates f1, precision, recall, accuracy scores
        '''
        best_est.fit(X, y)
        predictions = best_est.predict(X)
        recall = recall_score(y, predictions)
        precision = precision_score(y, predictions)
        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions)
        print "F1 score: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, Accuracy: {:.2f}".format(f1, precision, recall, accuracy)
        self.scores = {'f1': f1, 'precision': precision,
                       'recall': recall, 'accuracy': accuracy}

    def plot_roc_curve(self, best_est, X, y):
        '''
        Plot ROC curve for fitted model
        '''
        best_est.fit(X, y)
        predictions = best_est.predict_proba(X)
        # find roc numbers
        fpr, tpr, thresholds = roc_curve(y, predictions[:, 1], pos_label=1)
        # plot roc curve
        x_line = np.linspace(0, 1, 70)
        y_line = x_line
        plt.plot(fpr, tpr, label='RF Classifier (best model)', color='b')
        plt.plot(x_line, y_line, color='r')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.legend()
        dt_ = datetime.datetime.now().strftime('%d%b%y_%H%M')
        plt.savefig("../../data/" + dt_ + ".png")
        plt.show()
        pass

    def fit(self, filepath, gridsearch=False):
        self.load_data(filepath)
        X, y = self.prepare_data(self.df_full, y_name='fraud')
        # grid search RF
        rand = np.random.randint(100)
        params_rf = {'n_jobs': [-1], 'n_estimators': [50, 100], 'max_depth': [
            1, 2, 5, 8], 'min_samples_split': [2], 'random_state': [rand]}
        # "criterion": ["gini", "entropy"],
        # 'sample_leaf_options': [1, 5, 10, 50],
        # conditional gridsearch
        if gridsearch == True:
            self.grid_search(RF(), params_rf, X, y)
        else:
            self.best_est = RF().fit(X, y)
        self.scores_for_best_model(self.best_est, X, y)
        self.plot_roc_curve(self.best_est, X, y)

    def predict(self, X_):
        model = self.best_est
        model.predict_proba(X_)


if __name__ == '__main__':
    filepath = '../../data/data.json'
    model = RFmodel()
    x = model.load_data('../../data/data.json')
    X, y = model.prepare_data(x, y_name='fraud')
    # pd.DataFrame(x.iloc[14336, :]).T.to_csv('ex2.csv')
    # Fit the data
    model.fit('../../data/data.json', gridsearch=1)

    # model.df
    # model.df_full
    # Test
    # pd.DataFrame(model.X[:200, :].reshape(200, 13))

    # Save model using cPickle
    pickle.dump(model.best_est, open('../../data/model.pkl', 'wb'))

    # Make predictions
    np.round_(model.best_est.predict_proba(
        model.X[:200, :].reshape(200, 13)), 3)

    # md = pickle.load(open('../../data/model.pkl', 'rb'))

    # md.predict(model.X[:200, :].reshape(200, 13))
    # md.predict(X[2])
    # md.predict_proba(model.X[:200, :].reshape(200, 13))
    # model.plot_roc_curve(model.best_est, X, y)
    #
    # pd.DataFrame(md.predict(model.X)).to_clipboard()
    # pd.DataFrame(md.predict_proba(model.X)).to_clipboard()
    # Overall model scores:
    #  F1 score: 0.87, Precision: 0.96, Recall: 0.79, Accuracy: 0.98
    # (0.86963906581740968,
    #  0.96421845574387943,
    #  0.79195668986852286,
    #  0.97858687312547954)
