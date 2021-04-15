# Copyright (c) 2021 Microsoft
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import os
import json
from pathlib import Path

import click
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from azureml.core import Run
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Get the Run context - this will allow us to log metrics against the run.
run = Run.get_context()

METADATA_JSON = "metadata.json"


def read_data(data_path):
    diabetes_df = pd.read_csv(data_path)

    y = diabetes_df.pop('target').values
    X = diabetes_df.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    print(
        f"Data contains {len(X_train)} training samples and {len(X_test)} test samples")

    return X_train, X_test, y_train, y_test


def plot_residuals_v_actuals(y, y_hat):
    """Residuals (y-axis) vs. Actuals (x-axis) - colored green"""
    resids = y - y_hat

    fig = plt.figure()
    sns.regplot(x=y, y=resids, color='g')

    plt.title('Residual vs. Actual')
    plt.xlabel("Actual Value")
    plt.ylabel("Residuals")

    plt.close(fig)
    return fig


def plot_predictions(y, y_hat):
    """Predictions (y-axis) vs. Actuals (x-axis)"""
    fig = plt.figure()

    sns.regplot(x=y, y=y_hat, color='b')

    plt.title("Prediction vs. Actual")
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")

    plt.close(fig)
    return fig


def plot_resid_histogram(y, y_hat):
    resids = y - y_hat

    fig = plt.figure()
    sns.histplot(resids, color='g', kde=True)

    plt.title("Residual Histogram")

    plt.close(fig)
    return fig


def write_metadata(output_path: str, run: Run) -> None:
    metadata = {
        "run_id": run.id,
        "model_path": "outputs/model.pkl"
    }

    output_path: Path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_file = output_path / METADATA_JSON

    with open(metadata_file, 'w+') as fo:
        json.dump(metadata, fo)


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option("--data-path",
              type=click.Path(exists=True, file_okay=True, dir_okay=True),
              help='The path to the training data', required=True)
@click.option('--model-metadata-folder', "model_metadata_folder",
              type=click.Path(exists=False, file_okay=True, dir_okay=True),
              help="The folder to save model metadata")
@click.option('--alpha',
              default=0.03,
              type=float)
def main(data_path: str, alpha: float, model_metadata_folder: str):

    X_train, X_test, y_train, y_test = read_data(data_path)

    # Create, fit, and test the scikit-learn Ridge regression model
    regression_model = Ridge(alpha=alpha)
    regression_model.fit(X_train, y_train)
    preds = regression_model.predict(X_test)

    # Output the Mean Squared Error to the notebook and to the run
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"RMSE:\t{np.round(rmse,4)}",
          f"MAE:\t{np.round(mae,4)}",
          f"R2:\t{np.round(r2,4)}",
          sep='\n')

    resid_fig = plot_residuals_v_actuals(y_test, preds)
    resid_hist = plot_resid_histogram(y_test, preds)
    pred_plt = plot_predictions(y_test, preds)

    # Log metrics to Azure ML
    # THIS IS THE ONLY AML SPECIFIC CODE HERE #
    run.log('alpha', alpha)
    run.log('model_type', 'Ridge')
    run.log('rmse', rmse)
    run.log('mae', mae)
    run.log('r2', r2)
    run.log_image(name='residuals-v-actuals', plot=resid_fig)
    run.log_image(name='residuals-histogram', plot=resid_hist)
    run.log_image(name='prediction-v-actual', plot=pred_plt)

    # Save the model to the outputs directory for capture
    # Anything saved to ./outputs/ folder will be sent to Azure ML
    # at the end of the run
    joblib.dump(value=regression_model, filename='outputs/model.pkl')

    # Save information about our run as a json file in model_metadata_folder
    write_metadata(output_path=model_metadata_folder, run=run)

if __name__ == "__main__":
    main()
