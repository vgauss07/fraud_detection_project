import joblib
import streamlit as st

from config.paths_config import MODEL_OUTPUT_PATH
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

# Loading the trained model
trained_model = MODEL_OUTPUT_PATH

model = joblib.load(MODEL_OUTPUT_PATH)


@st.cache_data()
# make prediction based on data provided by the user
def prediction(OldBalance, Amount, NewBalance,
               TransactionType, TransactionHour):
    # Preprocess user input
    try:
        logger.info('Preprocessing user input')
        if TransactionType == 'Cash-In':
            TransactionType = 0
        elif TransactionType == 'Cash-Out':
            TransactionType = 1
        elif TransactionType == 'Debit':
            TransactionType = 2
        elif TransactionType == 'Payment':
            TransactionType = 3
        elif TransactionType == 'Transfer':
            TransactionType = 4
        else:
            print('INVALID TRANSACTION TYPE')

        prediction = model.predict([[OldBalance, Amount,
                                     NewBalance, TransactionType,
                                     TransactionHour
                                     ]])

        if prediction == 0:
            pred = 'Not Fraudulent'
        else:
            pred = 'Fraudulent'

        logger.info('Preprocessing and prediction done successfully')

        return pred

    except Exception as e:
        logger.error(f'Error processing and predicting user input {e}')
        raise CustomException('error processing and predicting use input', e)


# Function to Define Homepage of the Application
def main():

    try:
        logger.info('Building the Front End of the App')
        html_temp = """
        <div style ="background-color:green;padding:1px>
            <h1 style ="color:white;text-align:center;">
                Fraud Detection Application
            </h1>
        </div>
        """

        logger.info('Display the front end aspect')
        st.markdown(html_temp, unsafe_allow_html=True)

        logger.info('Create box field to get user data')
        OldBalance = st.number_input("Balance before Transaction")
        Amount = st.number_input("Amount Depositing")
        NewBalance = st.number_input("Balance after Transaction")
        TransactionType = st.selectbox("Type of Transaction", ("Cash-In",
                                                               "Cash-Out",
                                                               "Debit",
                                                               "Transfer",
                                                               "Payment"))
        TransactionHour = st.number_input("Time to Complete Transaction"
                                          "(in hours)")

        result = ""

        if st.button("Predict"):
            result = prediction(OldBalance, Amount,
                                NewBalance, TransactionType,
                                TransactionHour)
            st.success('This Transaction is {}'.format(result))

        logger.info('Successfully completed front end build')

    except Exception as e:
        logger.error(f'Error implementing front end build {e}')
        raise CustomException('error implementing front end build', e)


if __name__ == '__main__':
    main()
