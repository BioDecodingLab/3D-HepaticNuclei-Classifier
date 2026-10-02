"""Original classical hyperparameter grids, unchanged."""

PARAM_GRIDS = {
    "logreg": {
        "pca__n_components": [0.95, 0.5],
        "pca__whiten": [True, False],
        "clf__C": [0.001, 0.01, 0.1, 1, 10, 50],
        "clf__penalty": ["l2"],
        "clf__class_weight": [None],
    },
    "rf": {
        "pca__n_components": [0.95, 0.5],
        "pca__whiten": [False],
        "clf__n_estimators": [200, 300],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [5, 10],
        "clf__min_samples_leaf": [2, 4],
        "clf__max_features": ["sqrt"],
        "clf__class_weight": [None],
    },
    "svm": {
        "pca__n_components": [0.95, 0.5],
        "pca__whiten": [True, False],
        "clf__C": [0.001, 0.01, 0.1, 1, 10],
        "clf__kernel": ["rbf"],
        "clf__gamma": ["scale", 1e-3],
        "clf__class_weight": [None],
    },
    "mlp": {
        "pca__n_components": [0.95, 0.5],
        "pca__whiten": [False],
        "clf__hidden_layer_sizes": [(256, 128), (512, 256), (256, 128, 64)],
        "clf__alpha": [1e-4, 1e-3],
        "clf__learning_rate_init": [1e-3, 5e-4],
        "clf__batch_size": [64, 128],
    },
}
