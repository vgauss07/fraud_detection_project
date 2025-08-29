import joblib 
import mlflow
import mlflow.sklearn
import os
import pandas as pd

from scipy.stats import randint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression

from config.paths_config import *
from config.model_params import *
from src.custom_exception import CustomException
from src.logger import get_logger
from utils.common_functions import read_yaml, load_data


logger = get_logger(__name__)


class ModelTraining:
    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_dist = LOGISTREG_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def load_and_split(self):
        try:
            logger.info(f'Loading data from {self.train_path}')
            train_df = load_data(self.train_path)

            logger.info(f'Loading data from {self.test_path}')
            test_df = load_data(self.test_path)

            X_train = train_df.drop(columns=['isFraud'])
            y_train = train_df['isFraud']

            X_test = test_df.drop(columns=['isFraud'])
            y_test = test_df['isFraud']

            logger.info('Data split done successfully')

            return X_train, y_train, X_test, y_test

        except Exception as e:
            logger.error(f'Error while loading data {e}')
            raise CustomException('Failed to load data', e)

    def train_logistic_reg(self, X_train, y_train):
        try:
            logger.info('Initializing our model')

            logistic_reg_model = LogisticRegression(random_state=self.random_search_params['random_state'])

            logger.info('Starting Hyperparameter Tuning')

            random_search = RandomizedSearchCV(
                estimator=logistic_reg_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params['n_iter'],
                cv=self.random_search_params['cv'],
                verbose=self.random_search_params['random_state'],
                scoring=self.random_search_params['scoring']
            )

            random_search.fit(X_train, y_train)

            logger.info('Hyperparameter Tuning Completed')

            best_params = random_search.best_params_
            best_logistic_reg_model = random_search.best_estimator_

            logger.info(f'Best parameters are: {best_params}')

            return best_logistic_reg_model
        
        except Exception as e:
            logger.error(f'Error while training model {e}')
            raise CustomException('Failed to train model', e)

    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)

            logger.info('Saving the Model')
            joblib.dump(model, self.model_output_path)
            logger.info(f'Model saved to {self.model_output_path}')

        except Exception as e:
            logger.error(f'Error while saving model {e}')
            raise CustomException('Failed to save model', e)

    def run(self):
        try:
            with mlflow.start_run():
                logger.info('Starting Model Training Pipeline')

                logger.info('Start MLFlow Experimentation Tracking')

                logger.info('Logging the training and testing the dataset to MLFLOW')
                mlflow.log_artifact(self.train_path, artifact_path='datasets')
                mlflow.log_artifact(self.test_path, artifact_path='datasets')

                X_train, y_train, X_test, y_test = self.load_and_split()
                best_logistic_reg_model = self.train_logistic_reg(X_train,
                                                                  y_train)
                metrics = self.evaluate_model(best_logistic_reg_model, X_test,
                                              y_test)
                self.save_model(best_logistic_reg_model)

                logger.info('Logging Model into MLFLOW')
                mlflow.log_artifact(self.model_output_path)

                logger.info('Logging parameters and metrics to MLFLOW')
                mlflow.log_params(best_logistic_reg_model.get_params())
                mlflow.log_metrics(metrics)

                logger.info('Model Training Successfully Completed')

        except Exception as e:
            logger.error(f'Error in model training pipeline {e}')
            raise CustomException('Failed to train model', e)


if __name__ == '__main__':
    trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH)
    trainer.run()
