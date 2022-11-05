# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
from math import sqrt
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from admission_controller.app.config_generator.model_config_generator import (
    ModelConfigGenerator,
)
from admission_controller.app.model import ModelType


def plot_correlation_matrix(X):
    # Plot correlation matrix of features
    corr = X.corr(method="spearman")

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    fig1, ax = plt.subplots(figsize=(6, 5))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True, sep=100)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, linewidths=0.5
    )

    fig1.subtitle("Correlation matrix of features", fontsize=15)

    fig1.tight_layout()


def fit_model(
    X,
    y,
    transformer: Optional[ColumnTransformer],
    regressor,
    x_names,
    y_names,
    plot,
    inputs,
):
    regressor.fit(X, y)
    print(f"{regressor} score: {regressor.score(X,y)}")
    y_prediction = regressor.predict(X)
    print(f"Max Error: {max_error(y, y_prediction)}")
    print(f"MAE: {mean_absolute_error(y, y_prediction)}")
    print(f"MSE: {mean_squared_error(y, y_prediction)}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y, y_prediction))}")
    if type(regressor) == LinearRegression:
        print(f"Model coef: {regressor.coef_}")
        print(f"Model intercept: {regressor.intercept_}")

    if inputs:
        input_df = pd.DataFrame(
            {x_name: [inputs[i]] for i, x_name in enumerate(x_names)}
        )
        input_df = input_df.apply(pd.to_numeric, errors="coerce")
        input_df.fillna(0, inplace=True)
        print(input_df)

        if transformer:
            input_data = transformer.transform(input_df)
        else:
            input_data = input_df
        print(f"Transformed data: {input_data}")
        print(f"Prediction: {regressor.predict(input_data)}")

    if plot:
        y_pred = regressor.predict(X)

        i = 1
        plt.scatter(y, y_pred)


def fit_model_poly(X, y, x_names, y_names, plot):
    for i in range(1, 10):
        coeff, full_list = np.polynomial.polynomial.Polynomial.fit(
            np.squeeze(X), np.squeeze(y), i, full=True
        )
        poly_fn = np.poly1d(coeff)
        print(f"Polyfit degree {i} coeff: {coeff}")
        print(f"Polyfit degree {i} score: {sqrt(full_list[0])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--repo",
        type=str,
        required=True,
        help="directory of triton model repository",
    )
    parser.add_argument(
        "--platform",
        type=str,
        required=True,
        help="Platform of datafile",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="drop first with ohe",
    )
    parser.add_argument(
        "--isolate-categorical",
        type=str,
        default="",
        help="Just train model with one categorical var fixed",
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        required=True,
        help="List of models to be used during evaluation",
    )
    parser.add_argument(
        "-x",
        "--x-names",
        nargs="+",
        required=False,
        default=["Batch", "Concurrency", "CPU", "GPU"],
        help="List of x_names to be used in analysis",
    )
    parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        required=False,
        help="List of inputs to predict on",
    )
    parser.add_argument(
        "-y",
        "--y-names",
        nargs="+",
        required=False,
        default=[
            "Throughput (infer/sec)",
            "p99 Latency (ms)",
            "RAM Usage (MB)",
            "CPU Utilization (%)",
        ],
        help="List of y_names to be used in analysis",
    )
    parser.add_argument(
        "-s",
        "--scaling",
        action="store_true",
        help="Standardize training data",
    )
    parser.add_argument(
        "-d", "--dummy", action="store_true", help="Run dummy test"
    )
    parser.add_argument(
        "-p", "--plot", action="store_true", help="Output plot of data"
    )
    parser.add_argument(
        "--regressors",
        nargs="+",
        default=["linear"],
        choices=[
            "linear",
            "mo_linear",
            "mo_ridge",
            "mo_spline",
            "mo_rf",
            "numpy",
        ],
        help="Choose regression method",
    )
    args, unknown = parser.parse_known_args()

    if args.dummy:
        # create regression data
        x, y = make_regression(n_targets=5)
        # split into train and test data
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.30, random_state=42
        )
        # train the model
        clf = MultiOutputRegressor(
            RandomForestRegressor(max_depth=2, random_state=0)
        )
        lin = LinearRegression()
        lin_mo = MultiOutputRegressor(LinearRegression())
        ridge_mo = MultiOutputRegressor(Ridge(random_state=123))
        clf.fit(x_train, y_train)
        print(f"Random Forrest Reg score: {clf.score(x_train,y_train)}")
        lin.fit(x_train, y_train)
        print(f"Linear Reg score: {lin.score(x_train,y_train)}")
        lin_mo.fit(x_train, y_train)
        print(f"Linear Reg MO score: {lin_mo.score(x_train,y_train)}")
        ridge_mo.fit(x_train, y_train)
        print(f"Ridge Reg MO score: {ridge_mo.score(x_train,y_train)}")

        # predictions
        print(clf.predict([x_test[0]]))
        print(lin.predict([x_test[0]]))
        print(lin_mo.predict([x_test[0]]))
        print(ridge_mo.predict([x_test[0]]))

        sys.exit(0)

    for model in args.models:
        csv_file_path = (
            f"{args.repo}/{model}/metrics-model-inference-{args.platform}.csv"
        )
        model_type = ModelConfigGenerator.get_model_type(
            f"{args.repo}/{model}"
        )
        x_names, y_names, df = ModelConfigGenerator.model_analyzer_csv_to_df(
            model_type, csv_file_path
        )
        if args.isolate_categorical:
            df = df[df["cpu_accelerator"] == args.isolate_categorical]

        X = df.loc[:, args.x_names]
        y = df.loc[:, args.y_names]
        X = X.apply(pd.to_numeric, errors="coerce")
        X.fillna(0, inplace=True)
        print(X)
        print(y)

        for i, input in enumerate(args.inputs):
            try:
                args.inputs[i] = float(input)
            except:
                pass

        if (
            model_type == ModelType.tflite
            and "cpu_accelerator" in args.x_names
        ):
            t = ColumnTransformer(
                transformers=[
                    (
                        "onehot",
                        OneHotEncoder(
                            sparse=False, drop="first" if args.drop else None
                        ),
                        ["cpu_accelerator"],
                    ),
                ],
                remainder="passthrough",
            )
            X = t.fit_transform(X)
            print(X)

        else:
            t = None

        if "linear" in args.regressors:
            fit_model(
                X,
                y,
                t,
                LinearRegression(),
                args.x_names,
                args.y_names,
                args.plot,
                args.inputs,
            )
        if "mo_linear" in args.regressors:
            fit_model(
                X,
                y,
                t,
                MultiOutputRegressor(LinearRegression()),
                args.x_names,
                args.y_names,
                args.plot,
                args.inputs,
            )
        if "mo_ridge" in args.regressors:
            fit_model(
                X,
                y,
                t,
                MultiOutputRegressor(Ridge(random_state=123)),
                args.x_names,
                args.y_names,
                args.plot,
                args.inputs,
            )
        if "mo_rf" in args.regressors:
            fit_model(
                X,
                y,
                t,
                MultiOutputRegressor(
                    RandomForestRegressor(max_depth=2, random_state=0)
                ),
                args.x_names,
                args.y_names,
                args.plot,
                args.inputs,
            )
        if "numpy" in args.regressors:
            fit_model_poly(
                X,
                y,
                args.x_names,
                args.y_names,
                args.plot,
            )

        if args.plot:
            # plot_correlation_matrix(X)
            plt.show()
