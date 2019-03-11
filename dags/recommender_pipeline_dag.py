import pickle
import numpy as np
import logging

from datetime import timedelta

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
    'depends_on_past': True,
    'start_date': days_ago(10),
    'retries': 5,
    'retry_delay': timedelta(minutes=2)
}

input_file_path = '/usr/local/airflow/s3/input_data/data.pickle'
output_path = '/usr/local/airflow/s3/output_data/'
model_file_path = '/usr/local/airflow/s3/model/recommender_model.pickle'
dag = DAG('recommender_pipeline_dag', default_args=default_args, schedule_interval='@daily')


# Extract task
def extract_data(**context):
    logging.info('EXECUTION_DATE: %s' % context['task_instance'].execution_date)
    time_step = (context['task_instance'].execution_date - default_args['start_date']).days
    logging.info('TIME_STEP: %d' % time_step)

    with open(input_file_path, mode='rb') as input_file:
        movielens = pickle.load(input_file)
        train = movielens['train'][time_step]
        shape = movielens['shape']

        # TODO: OPTMIZE
        rows, cols, dta = train[0], train[1], train[2]
        train_data = coo_matrix((dta, (rows, cols)), shape=shape)
        logging.info('NNZ: %d' % train_data.nnz)

        create_dir(output_path)
        with open(output_path+'output_train.pickle', mode='wb') as output_file:
            pickle.dump(train_data, output_file)
            path = output_file.name

    # send the path for the next task
    context['task_instance'].xcom_push(key='time_step', value=time_step)
    context['task_instance'].xcom_push(key='path', value=path)


extract_task = PythonOperator(
    task_id='extract_task',
    python_callable=extract_data,
    provide_context=True,
    dag=dag
)


# Train task
def train_model(**context):
    epochs = 50
    path = context['task_instance'].xcom_pull(key='path', task_ids='extract_task')
    with open(path, mode='rb') as file:
        train_interactions = pickle.load(file)
    with open(input_file_path, mode='rb') as input_file:
        test_interactions = pickle.load(input_file)['test']

    try:
        with open(model_file_path, mode='rb') as file:
            recommender = pickle.load(file)
    except FileNotFoundError:
        recommender = Recommender()
    max_epoch, max_auc = recommender.fit_until_decay(interactions=train_interactions, val_interactions=test_interactions, max_epochs=epochs, patience=1)
    logging.info('MAX EPOCH: {} \t MAX AUC: {}'.format(max_epoch, max_auc))
    recommender.dump_model(model_file_path)


train_task = PythonOperator(
    task_id='train_task',
    python_callable=train_model,
    provide_context=True,
    dag=dag
)


# Test task
def test_model(**context):
    with open(model_file_path, mode='rb') as file:
        model = pickle.load(file)
        with open(input_file_path, mode='rb') as input_file:
            data = pickle.load(input_file)
            test_data = data['test']
            auc = model.evaluate_auc(test_data)
            pre_at_10 = model.evaluate_at_k(test_data, 10)
            context['task_instance'].xcom_push(key='metrics', value=(auc, pre_at_10))


test_task = PythonOperator(
    task_id='test_task',
    python_callable=test_model,
    provide_context=True,
    dag=dag
)


# Publish task
def publish_model(**context):
    time_step = context['task_instance'].xcom_pull(key='time_step', task_ids='extract_task')
    auc, pre_at_10 = context['task_instance'].xcom_pull(key='metrics', task_ids='test_task')

    logging.info('TEST AUC: %.2f:' % auc)
    logging.info('TEST PRECISION_AT_10: %.2f:' % pre_at_10)
    with open(output_path + 'performance.csv', mode='a') as result_file:
        result_file.write('{time_step},{auc},{pre_at_10}\n'.format(time_step=time_step, auc=auc, pre_at_10=pre_at_10))


publish_task = PythonOperator(
    task_id='publish_task',
    python_callable=publish_model,
    provide_context=True,
    dag=dag,
)

# Define graph
train_task.set_upstream(extract_task)
test_task.set_upstream(train_task)
publish_task.set_upstream(test_task)
