import pandas as pd
import numpy as np
from mp_api.client import MPRester
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition
from src.processing import OxyData
from sklearn.model_selection import train_test_split
import xgboost
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score


class Classifier:
    def __init__(self, df):
        self.df = df

        self.df = self.df.sort_values(
            by='energy_above_hull')  # DROPS DUPLICATES (e.g. 4 forms of CeO2, where 1 is stable) - keeps the most stable data statistically
        self.df = self.df.drop_duplicates(subset=['formula_pretty'], keep='first')

        self.df['is_stable'] = (self.df['energy_above_hull'] <= 0.05).astype(int)

    def prep_data(self):
        # Dataframe with stable materials
        self.stable_df = pd.DataFrame(self.df)

        X = self.df.drop(
            columns=['formula_pretty', 'formation_energy_per_atom', 'structure', 'composition', 'energy_above_hull',
                     'is_stable'])
        Y = self.df['is_stable']

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    def model(self):
        self.model_class = xgboost.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=1000,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            early_stopping_rounds=25,
            random_state=42
        )

    def load_model(self):
        self.model_class = xgboost.XGBClassifier()
        self.model_class.load_model('models/classifier.json')

    def train(self):
        self.fit = self.model_class.fit(self.X_train, self.Y_train,
                                        eval_set=[(self.X_test, self.Y_test)], verbose=False)
        self.y_pred = self.model_class.predict(self.X_test)

    def results(self):
        print(f'Classification model performance:')
        print(f'{classification_report(self.Y_test, self.y_pred)}')
        print(f'')
        print(f'Confusion matrix:')
        print(pd.DataFrame(confusion_matrix(self.Y_test, self.y_pred), index=['Real 0', 'Real 1'],
                           columns=['Predicted 0', 'Predicted 1']))
        print(f"")
        xgboost.plot_importance(self.model_class, importance_type='gain', max_num_features=10)
        plt.show()

    def save_model(self):
        self.model_class.save_model('models/classifier.json')



class Regressor:
    def __init__(self, df):
        self.df = df

        self.df = self.df.sort_values(
            by='energy_above_hull')  # DROPS DUPLICATES (e.g. 4 forms of CeO2, where 1 is stable) - keeps the most stable data statistically
        self.df = self.df.drop_duplicates(subset=['formula_pretty'], keep='first')

        self.df['is_stable'] = (self.df['energy_above_hull'] <= 0.05).astype(int)

    def prep_data(self):

        # Dataframe with stable materials
        self.stable_df = pd.DataFrame(self.df)

        # CLEAR DATA

        unstable_idx = [idx for idx in self.stable_df.index if self.stable_df.loc[idx, 'is_stable'] == 0]
        self.stable_df = self.df.drop(unstable_idx)

        Xr = self.stable_df.drop(
            columns=['formula_pretty', 'formation_energy_per_atom', 'structure', 'composition', 'energy_above_hull',
                     'is_stable'])
        Yr = self.stable_df['formation_energy_per_atom']

        self.Xr_train, self.Xr_test, self.Yr_train, self.Yr_test = train_test_split(Xr, Yr, test_size=0.2, random_state=42)

    def model(self):
        self.model_reg = xgboost.XGBRegressor(
            max_depth=8,
            learning_rate=0.01,
            n_estimators=2000,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            early_stopping_rounds=25
        )

    def load_model(self):
        self.model_reg = xgboost.XGBRegressor()
        self.model_reg.load_model('models/regression.json')

    def train(self):
        self.fit = self.model_reg.fit(self.Xr_train, self.Yr_train,
                                      eval_set=[(self.Xr_test, self.Yr_test)], verbose=False)
        self.yr_pred = self.model_reg.predict(self.Xr_test)


    def results(self):
        print(f'')
        print(f'R2 = {r2_score(self.Yr_test, self.yr_pred)}')
        print(f'MAE = {mean_absolute_error(self.Yr_test, self.yr_pred)}')
        print(f"")
        plt.xlabel('true y')
        plt.ylabel('predicted y')
        lims = [min(self.Yr_test), max(self.Yr_test)]
        plt.plot(lims, lims, 'k--')
        plt.title(f'predicted y vs. true y')
        plt.plot(self.Yr_test, self.yr_pred, '.', alpha=0.4)
        plt.show()

    def save_model(self):
        self.model_reg.save_model('models/regression.json')

