# AIRFLOW RECOMMENDER PIPELINE

This repository consists of a simplified recommender pipeline using the Apache Airflow framework.
Although is not the focus of this work, the recommendation engine was based on the great LightFM library.
And the dataset used for the tests was the movielens dataset (the 100k ratings version) previously parsed into a sparse COO matrix format.

The dataset is composed of roughly 90k ratings for the training set, and 10k ratings for the test set.
The training set was randomized and evenly distributed in 10 sparse matrices.

To make the data transfer simpler, all the data transfer is file-based, and is stored on the s3 folder, to simulate 
the AWS S3 service, which is basically a cloud file system. The same results could be obtained using the boto3 library
in conjunction with the AWS S3 service, or any data source.

## Tasks structure
To simulate the passage of time, the start of the DAG was set by default to 10 days ago of the running date of the DAG. 
Each new day would represent the addition of a new split of the training data being added to the model. 

This way, we're trying to simulate the **addition of new data** and the trigger of a job to:
- Extract this new data;
- Retrain the current model with the new data;
- Test the model performance;
- Publish the results;

With the addition of new data we expect the performance of the model to increase, which can be seen, after all the tasks
ended on the file:
```
s3/output_data/performance.csv
```

## Setup
The easiest way to setup this project is using docker-compose. To do so, open this project directory and
run the docker-compose file based on the great [airflow image provided by puckel](https://github.com/puckel/docker-airflow).

For this, just run the script **run_airflow_container.sh**, which, in addition of setting the container up,
will also set the write permissions needed on the volumes.

Then after, the airflow webserver will start on [localhost:8080](http://localhost:8080/), you just need to turn the DAG on, 
and wait for all of the runs to be executed.

### Windows
If you are on windows, remember to set the PWD environment variable to the current directory before running 
"docker-compose up":
```
SET PWD=%cd%
docker-compose up
```

