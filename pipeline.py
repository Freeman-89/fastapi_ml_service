from datetime import datetime
import dill
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def main():
    df = pd.read_csv('./data/data_drope_none_balanced.csv')
    print(df.head())

    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=['object'])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    transform = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    preprocessor = Pipeline(steps=[
        ('transform', transform)
    ])

    models = [
        LogisticRegression(solver='liblinear', class_weight='balanced'),
        RandomForestClassifier(class_weight='balanced'),
        SVC(C=1, kernel='rbf', probability=True, class_weight='balanced'),
        MLPClassifier(hidden_layer_sizes=(200, 100, 50), max_iter=1000, early_stopping=True, random_state=42)
    ]

    X = df.drop(['target'], axis=1)
    y = df['target']

    best_score = 0
    best_model = None

    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='roc_auc')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_model = pipe

    best_model.fit(X, y)

    print(f'Best Model: {type(best_model.named_steps["classifier"]).__name__}, Accuracy: {best_score:.4f}')
    with open('./model/model_pipe.pkl', 'wb') as file:
        dill.dump({
            'model': best_model,
            'metadata': {
                'name': 'target prediction model',
                'author': 'Gavrilov Artemiy',
                'version': 1,
                'date': datetime.now(),
                'type': type(best_model.named_steps["classifier"]).__name__,
                'accuracy': best_score
            }
        }, file)
    print('Best model saved to model.pkl')


if __name__ == '__main__':
    main()

    df = pd.read_csv('./data/data_drope_none_balanced.csv').drop('target', axis=1)

    types = {
        'int64': 'int',
        'float64': 'float'
    }
    for k, v in df.dtypes.items():
        print(f'{k}: {types.get(str(v), "str")}')

