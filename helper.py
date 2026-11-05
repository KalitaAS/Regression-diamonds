import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.base import clone
import plots as p
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.compose import TransformedTargetRegressor

def divide_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def winsorize_outliers(df, column_name, lower_bound=None, upper_bound=None):
    df = df.copy()

    if lower_bound is not None:
        df.loc[df[column_name] < lower_bound, column_name] = lower_bound
    if upper_bound is not None:
        df.loc[df[column_name] > upper_bound, column_name] = upper_bound

    return df


import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
def calculate_regression_metrics(y_test, y_pred):
    return {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'MAPE, %': mean_absolute_percentage_error(y_test, y_pred) * 100,
        'Predictions': {'y_true': y_test, 'y_pred': y_pred}
    }


def evaluate_regression(y_test, y_pred, model_name="Model", enable_plot=True):
    metrics = calculate_regression_metrics(y_test, y_pred)

    if enable_plot:
        p.plot_regression_results(metrics, model_name)
        p.print_regression_report(metrics, model_name)

    # Возвращаем только числовые метрики (без данных для графиков)
    return {k: v for k, v in metrics.items() if k != 'Predictions'}


def plot_regression_metrics_heatmap(metrics_df, title='Model Evaluation Metrics Comparison'):
    norm = (metrics_df - metrics_df.min()) / (metrics_df.max() - metrics_df.min())
    for col in ['MAE', 'RMSE', 'MAPE, %']:  # у этих метрик меньше = лучше
        norm[col] = 1 - norm[col]

    plt.figure(figsize=(8, 4))
    sns.heatmap(norm, cmap='RdBu', annot=metrics_df, fmt=".2f", cbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def train_evaluate_model_cv_regression(model, X, y, preprocessor=None, cv=5,
                                       seed=None, log_target=False):
    """Кросс-валидация регрессионной модели"""
    if seed is not None and hasattr(model, 'random_state'):
        model.set_params(random_state=seed)

    if log_target:
        model = TransformedTargetRegressor(
            regressor=model, func=np.log1p, inverse_func=np.expm1)

    if preprocessor is not None:
        pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    else:
        pipeline = model

    scoring = {
        'mae': 'neg_mean_absolute_error',
        'rmse': 'neg_root_mean_squared_error',
        'r2': 'r2',
        'mape': 'neg_mean_absolute_percentage_error',
    }
    cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)

    return {
        'MAE': -cv_results['test_mae'].mean(),
        'RMSE': -cv_results['test_rmse'].mean(),
        'R2': cv_results['test_r2'].mean(),
        'MAPE, %': -cv_results['test_mape'].mean() * 100,
    }


def train_evaluate_models_cv_regression(models: list, X, y, preprocessor=None,
                                        cv=5, seed=None, log_target=False,
                                        enable_plot=True):
    all_metrics = {}

    for model_name, model in models:
        # Работаем с копиями, чтобы не менять исходные объекты
        current_model = clone(model)
        current_preprocessor = clone(preprocessor) if preprocessor is not None else None

        all_metrics[model_name] = train_evaluate_model_cv_regression(
            current_model, X, y, current_preprocessor, cv, seed, log_target)

    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')

    if enable_plot:
        plot_regression_metrics_heatmap(metrics_df)  # из прошлого шага

    return metrics_df

def train_evaluate_models_regression(models: list, X_train, y_train, X_test, y_test,
                                     enable_plot=False):
    """
    Обучает модели на train и считает метрики регрессии на test.
    models: список (имя, модель или Pipeline).
    """
    all_metrics = {}
    for model_name, model in models:
        current_model = clone(model)
        current_model.fit(X_train, y_train)
        y_pred = current_model.predict(X_test)
        all_metrics[model_name] = evaluate_regression(
            y_test=y_test, y_pred=y_pred, model_name=model_name, enable_plot=False
        )

    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')

    if enable_plot:
        plot_regression_metrics_heatmap(metrics_df)

    return metrics_df