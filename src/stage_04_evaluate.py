from src.utils.all_utils import read_yaml, create_directory, save_report
import argparse
import pandas as pd
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_metrices(actual_values, predicted_values):
    rmse = np.sqrt(mean_squared_error(actual_values,predicted_values))
    mae = mean_absolute_error(actual_values, predicted_values)
    r2 = r2_score(actual_values,predicted_values)

    return rmse, mae, r2

def evaluate(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts_dir = config["artifacts"]['artifacts_dir']
    split_data_dir = config["artifacts"]["split_data_dir"]

    test_data_filename = config["artifacts"]["test"]

    test_data_path = os.path.join(artifacts_dir,split_data_dir,test_data_filename)

    test = pd.read_csv(test_data_path)

    test_y = test["quality"]
    test_x = test.drop("quality",axis=1)

    model_dir_name = config["artifacts"]["model_dir"]
    model_dir = os.path.join(artifacts_dir,model_dir_name)

    model_filename = config["artifacts"]["model_filename"]
    model_path = os.path.join(model_dir, model_filename)
    model = joblib.load(model_path)

    predicted_values = model.predict(test_x)

    rmse, mae, r2 = evaluate_metrices(test_y, predicted_values)
    print(f"{rmse} {mae} {r2}")

    report_dirname = config["artifacts"]["report_dir"]
    report_dirpath = os.path.join(artifacts_dir, report_dirname)
    create_directory([report_dirpath])
    report_filename = config["artifacts"]["report_filename"]
    report_filepath = os.path.join(report_dirpath, report_filename)

    scores = {
        "rmse" : rmse,
        "mae" : mae,
        "r2" : r2 
    }

    save_report(scores, report_filepath)
    print(f"scores: {scores} added to the {report_filepath}")

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config\config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()

    evaluate(config_path=parsed_args.config, params_path=parsed_args.params)
