# Copyright (c) 2021 Microsoft
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import click
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from azureml.core import Run
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Get the Run context - this will allow us to log metrics against the run.
run = Run.get_context()


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


@click.command()
@click.option("--data-path", type=str, help='The path to the training data', required=True)
@click.option('--alpha', default=0.03, type=float)
@click.option('--learning-rate',
              default=0.1,
              type=float,
              help="Learning rate shrinks the contribution of each tree by learning_rate.")
@click.option('--n-estimators',
              default=100,
              type=int,
              help="The number of boosting stages to perform.")
@click.option('--max-depth',
              default=3,
              type=int,
              help="Maximum depth of the individual regression estimators.")
def main(data_path: str,
         alpha: float,
         learning_rate: float,
         n_estimators: int,
         max_depth: int):

    X_train, X_test, y_train, y_test = read_data(data_path)

    # Create, fit, and test the scikit-learn Ridge regression model
    regression_model = GradientBoostingRegressor(
        alpha=alpha,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth
        )

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
    run.log('learning-rate', learning_rate)
    run.log('max-depth', max_depth)
    run.log('n_estimators', n_estimators)
    run.log('model_type', 'Gradient Boosted Regressor')
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


if __name__ == "__main__":
    main()
