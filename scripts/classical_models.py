"""Original estimators; explicit MLP validation and group-safe SVM calibration."""

import copy
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from pipeline_common import LABELS
from evaluation import metrics, score_tuple


def preprocessing(params, seed):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=params["pca__n_components"],
                    whiten=params["pca__whiten"],
                    svd_solver="full",
                    random_state=seed,
                ),
            ),
        ]
    )


def classifier(name, p, seed):
    kw = {k[5:]: v for k, v in p.items() if k.startswith("clf__")}
    if name == "logreg":
        # L2 is the default in both old and new sklearn; omit deprecated spelling.
        if kw.get("penalty") == "l2":
            kw.pop("penalty")
        return LogisticRegression(
            **kw, solver="lbfgs", max_iter=5000, random_state=seed
        )
    if name == "rf":
        return RandomForestClassifier(**kw, bootstrap=True, n_jobs=1, random_state=seed)
    if name == "svm":
        return SVC(
            **kw, probability=False, decision_function_shape="ovr", random_state=seed
        )
    if name == "mlp":
        return MLPClassifier(
            **kw,
            activation="relu",
            solver="adam",
            max_iter=1,
            early_stopping=False,
            shuffle=True,
            random_state=seed
        )
    raise ValueError(name)


class GroupCalibratedSVM:
    """OOF calibration on one original view per observed training identity.

    Each calibration fold excludes every augmented relative of its original
    evaluation nuclei. Scaler/PCA/SVM are refitted inside each calibration fold.
    Final SVM fits all supplied training rows. Validation/test never calibrate.
    """

    def __init__(self, pipeline, calibrator):
        self.pipeline, self.calibrator = pipeline, calibrator
        self.classes_ = np.asarray(LABELS)

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.calibrator.predict_proba(self.pipeline.decision_function(X))


def _fit_preprocessing(pre, X):
    """Cache only training-fitted preprocessing, never a classifier."""
    return pre, pre.fit_transform(X)


def train_candidate(
    name, p, X, y, Xv, yv, ids, Xo, yo, ido, seed=42, max_epochs=500, patience=15, memory=None
):
    history = []
    if name == "svm":
        observed = set(ids)
        take = np.array([s in observed for s in ido])
        Xbase, ybase, ibase = Xo[take], yo[take], ido[take]
        counts = np.array([np.sum(ybase == c) for c in LABELS])
        folds = min(5, int(counts.min()))
        if folds < 2:
            raise ValueError(
                "SVM calibration needs >=2 unique training nuclei per class"
            )
        oof = np.empty((len(ybase), 5))

        def make():
            return Pipeline(
                [("pre", preprocessing(p, seed)), ("clf", classifier(name, p, seed))],
                memory=memory,
            )

        for _, held in StratifiedKFold(folds, shuffle=True, random_state=seed).split(
            Xbase, ybase
        ):
            heldids = set(ibase[held])
            keep = np.array([s not in heldids for s in ids])
            if set(y[keep]) != set(LABELS):
                raise ValueError("Calibration training class missing")
            foldmodel = make().fit(X[keep], y[keep])
            oof[held] = foldmodel.decision_function(Xbase[held])
        cal = LogisticRegression(C=1.0, max_iter=5000, random_state=seed).fit(
            oof, ybase
        )
        model = GroupCalibratedSVM(make().fit(X, y), cal)
    elif name == "mlp":
        pre = preprocessing(p, seed)
        if memory is None:
            Xt = pre.fit_transform(X)
        else:
            pre, Xt = memory.cache(_fit_preprocessing)(pre, X)
        V = pre.transform(Xv)
        clf = classifier(name, p, seed)
        best = None
        bestscore = None
        stale = 0
        for epoch in range(1, max_epochs + 1):
            clf.partial_fit(Xt, y, classes=LABELS)
            m = metrics(yv, clf.predict(V), clf.predict_proba(V))
            score = score_tuple(m)
            history.append(dict(epoch=epoch, train_loss=float(clf.loss_), **m))
            if bestscore is None or score > bestscore:
                bestscore = score
                best = copy.deepcopy(clf)
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        model = Pipeline([("pre", pre), ("clf", best)])
    else:
        model = Pipeline(
            [("pre", preprocessing(p, seed)), ("clf", classifier(name, p, seed))],
            memory=memory,
        ).fit(X, y)
    fitted_pipeline = model.pipeline if name == "svm" else model
    fitted_pipeline.memory = None
    m = metrics(yv, model.predict(Xv), model.predict_proba(Xv))
    return model, m, history
