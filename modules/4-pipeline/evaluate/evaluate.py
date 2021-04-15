# Copyright (c) 2021 Microsoft
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import json
import os
from pathlib import Path
from typing import Any

import click
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from azureml.core import Experiment, Model, Run, Workspace
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Retrieve the run, experiment and workspace
run = Run.get_context()
experiment: Experiment = run.experiment
workspace: Workspace = experiment.workspace

METADATA_JSON = "metadata.json"
RECOMMENDATION_JSON = 'recommend.json'


def read_metadata(folder_path: str) -> dict:
    model_metadata_path = Path(folder_path)
    model_metadata_file = model_metadata_path / METADATA_JSON

    with open(model_metadata_file, 'r') as fo:
        metadata = json.load(fo)

    return metadata



def load_champion_model(model_name: str) -> Any:
    champ = Model(workspace, model_name)
    champ_model_path = champ.download(target_dir='./champion')

    return joblib.load(champ_model_path)


def load_challenger_model(metadata: dict) -> Any:
    train_run = Run.get(workspace, metadata['run_id'])
    challenger_path = "./challenger/model.pkl"

    train_run.download_file(
        name=metadata['model_path'],
        output_file_path=challenger_path
    )

    return joblib.load(challenger_path)


def read_data(data_path):
    diabetes_df = pd.read_csv(data_path)

    y = diabetes_df.pop('target').values
    X = diabetes_df.values

    return X, y


def run_assessment(model, y_true: np.array, X: np.array) -> float:
    y_pred: np.array = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse


def write_recommendation_file(output_path: str, recommend_register: bool) -> None:
    recommendation = {"register": recommend_register}
    
    output_path: Path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    recommend_file = output_path / RECOMMENDATION_JSON
    
    with open(recommend_file, 'w+') as fo:
        json.dump(recommendation, fo)


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option("--validation-data-path",
              type=click.Path(exists=True, file_okay=True, dir_okay=True),
              help='The path to the training data', required=True)
@click.option('--model-metadata-folder',
              type=click.Path(exists=True, file_okay=True, dir_okay=True),
              help="The folder to save model metadata")
@click.option("--existing-model-name",
              type=str,
              default='diabetes',
              help="The name of the currently registered model to compare to")
@click.option("--registration-decision-folder",
              type=click.Path(exists=False, file_okay=True, dir_okay=True),
              help='The path to save an indicator file to determine if the model should be registered', required=True)
def main(validation_data_path: str, model_metadata_folder: str, existing_model_name: str, registration_decision_folder: str):
    # Load the model metadata
    metadata = read_metadata(model_metadata_folder)

    # Download and rehydrate the champion and challenger models
    champ_model = load_champion_model(existing_model_name)
    chall_model = load_challenger_model(metadata)

    # Load the validation dataset
    X, y = read_data(validation_data_path)

    # Calculate the champion and challenger RMSEs
    champ_rmse = run_assessment(champ_model,
                                y_true=y,
                                X=X)

    chall_rmse = run_assessment(chall_model,
                                 y_true=y,
                                 X=X)

    run.log('champion_rmse', champ_rmse)
    run.log('challenger_rmse', chall_rmse)

    # If the challenger has a lower RMSE, then recommend model registration
    recommend_register = bool(chall_rmse < champ_rmse)

    run.log('recommend_register', recommend_register)

    # Write the model recommendation
    write_recommendation_file(registration_decision_folder, recommend_register)

if __name__ == '__main__':
    main()