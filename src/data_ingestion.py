import os
import pandas as pd

from sklearn.model_selection import train_test_split

from config.paths_config import RAW_DIR, RAW_FILE_PATH, CONFIG_PATH
from config.paths_config import TRAIN_FILE_PATH, TEST_FILE_PATH
from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_functions import read_yaml


logger = get_logger(__name__)


class DataIngestion:
    def __init__(self, config):
        self.config = config['data_ingestion']
        self.train_test_ratio = self.config['train_ratio']

        os.makedirs(RAW_DIR, exist_ok=True)

        logger.info('Reading and Splitting the Data')

    def split_data(self):
        try:
            logger.info('Starting the Splitting Process')
            data = pd.read_csv(RAW_FILE_PATH)

            train_data, test_data = train_test_split(
                                        data,
                                        test_size=1 - self.train_test_ratio,
                                        random_state=32)
            train_data.to_csv(TRAIN_FILE_PATH)
            test_data.to_csv(TEST_FILE_PATH)

            logger.info(f'Train data saved to {TRAIN_FILE_PATH}')
            logger.info(f'Test data saved to {TEST_FILE_PATH}')

        except Exception as e:
            logger.error('Error splitting data')
            raise CustomException('Failed to split data', e)

    def run(self):
        try:
            logger.info('Starting data ingestion process')
            self.split_data()

            logger.info('Data ingestion completed successfully')

        except CustomException as ce:
            logger.error(f'CustomException: {str(ce)}')

        finally:
            logger.error('Data Ingestion Completed')


if __name__ == "__main__":

    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()
