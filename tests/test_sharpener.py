import pytest
import numpy as np
from pyDMS.pyDMS import (
    DecisionTreeSharpener,
    DecisionTreeRegressorWithLinearLeafRegression,
)


# ----------------------------------------------------------------------
# DecisionTreeRegressorWithLinearLeafRegression
# ----------------------------------------------------------------------
def test_tree_fit_and_predict_basic():
    X = np.random.rand(10, 2)
    y = np.random.rand(10)

    model = DecisionTreeRegressorWithLinearLeafRegression()
    model.fit(X, y, sample_weight=None)
    y_pred = model.predict(X)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y.shape
    assert len(model.leafParameters) > 0


# ----------------------------------------------------------------------
# DecisionTreeSharpener
# ----------------------------------------------------------------------
def test_sharpener_init_basic():
    sharp = DecisionTreeSharpener(["h1.tif"], ["l1.tif"])
    assert sharp.movingWindowExtension == 0
    assert sharp.autoAdjustCvThreshold
    assert isinstance(sharp.regressorOpt, dict)


def test_sharpener_init_qualityfile_mismatch():
    with pytest.raises(IOError):
        DecisionTreeSharpener(
            ["h1.tif"], ["l1.tif"], lowResQualityFiles=["q1.tif", "q2.tif"]
        )
