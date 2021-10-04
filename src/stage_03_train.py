from src.utils.all_utils import read_yaml, create_directory, save_local_df
import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet 
#from sklearn.externals import joblib
import joblib

def train(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts_dir = config["artifacts"]['artifacts_dir']
    split_data_dir = config["artifacts"]["split_data_dir"]

    train_data_filename = config["artifacts"]["train"]
    test_data_filename = config["artifacts"]["test"]

    train_data_path = os.path.join(artifacts_dir,split_data_dir,train_data_filename)

    random_state = params["base"]["random_state"]
    alpha = params["model_params"]["ElasticNet"]["alpha"]
    l1_ratio = params["model_params"]["ElasticNet"]["l1_ratio"]

    train = pd.read_csv(train_data_path)
    print(train.head())
    train_y = train["quality"]
    train_x = train.drop("quality",axis=1)

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    model = lr.fit(train_x, train_y)

    model_dir_name = config["artifacts"]["model_dir"]
    model_dir = os.path.join(artifacts_dir,model_dir_name)
    create_directory([model_dir])
    model_filename = config["artifacts"]["model_filename"]
    model_path = os.path.join(model_dir, model_filename)
    #print(model_path)
    joblib.dump(lr, model_path)

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config\config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()

    train(config_path=parsed_args.config, params_path=parsed_args.params)
