import pickle
import numpy as np
import pytz
import logging

from datetime import timedelta, datetime

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

from lib.recommender import Recommender
from lib.utils import create_dir
from scipy.sparse import coo_matrix

FTP_CONN = 'teste_ftp'
HTTP_CONN = 'test_http'
MYSQL_CONN = 'place'

default_args = {
    'owner': 'lucas.silva',
    'depends_on_past': True,  # para esperar que os dias anteriores terminem
    'start_date': days_ago(10),
    'retries': 5,
    'retry_delay': timedelta(minutes=2)
}

time_step = 1
input_path = '/usr/local/airflow/s3/input_data/data.pickle'
output_path = '/usr/local/airflow/s3/output_data/output_train.pickle'
model_path = '/usr/local/airflow/s3/model/recommender_model.pickle'
dag = DAG('recommender_pipeline_dag', default_args=default_args, schedule_interval='@daily')


# Extract task
def extract_data(**context):
    logging.info('EXECUTION_DATE: %s' % context['task_instance'].execution_date)
    time_step = (context['task_instance'].execution_date - default_args['start_date']).days + 1
    logging.info('TIME_STEP: %d' % time_step)

    with open(input_path, mode='rb') as input_file:
        movielens = pickle.load(input_file)
        train = movielens['train'][:time_step]
        shape = movielens['shape']

        # TODO: OPTMIZE
        rows, cols, dta = np.concatenate([i[0] for i in train]), np.concatenate([i[1] for i in train]), np.concatenate(
            [i[2] for i in train])
        train_data = coo_matrix((dta, (rows, cols)), shape=shape)
        logging.info('NNZ: %d' % train_data.nnz)

        create_dir(output_path)
        with open(output_path, mode='wb') as output_file:
            pickle.dump(train_data, output_file)
            path = output_file.name

    # send the path for the next task
    context['task_instance'].xcom_push(key='time_step', value=time_step)
    return path


extract_task = PythonOperator(
    task_id='extract_task',
    python_callable=extract_data,
    provide_context=True,
    dag=dag
)


# Train task
def train_model(**context):
    epochs = 10
    path = context['task_instance'].xcom_pull(task_ids='extract_task')
    with open(path, mode='rb') as file:
        train_interactions = pickle.load(file)
    recommender = Recommender()
    recommender.fit(interactions=train_interactions, epochs=epochs)
    recommender.dump_model(model_path)


train_task = PythonOperator(
    task_id='train_task',
    python_callable=train_model,
    provide_context=True,
    dag=dag
)


# Publish task
def publish_model():
    with open(model_path, mode='rb') as file:
        model = pickle.load(file)
        with open(input_path, mode='rb') as input_file:
            data = pickle.load(input_file)
            test_data = data['test']
            auc = model.evaluate_auc(test_data)
            mak = model.evaluate_at_k(test_data, 10)

            logging.info('TEST AUC: %.2f:' % auc)
            logging.info('TEST PRECISION_AT_10: %.2f:' % mak)


publish_task = PythonOperator(
    task_id='publish_task',
    python_callable=publish_model,
    dag=dag,
)

# Define graph
train_task.set_upstream(extract_task)
publish_task.set_upstream(train_task)

if __name__ == "__main__":
    DEFAULT_DATE = datetime.now()
    DEFAULT_DATE = DEFAULT_DATE.replace(tzinfo=pytz.utc)
    logging.info(DEFAULT_DATE)
    logging.info('START')
    # dag.run()
    extract_task.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE)
    train_task.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE)
    publish_task.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE)
    logging.info('END')

