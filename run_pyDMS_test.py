# -*- coding: utf-8 -*-
"""
@author: radoslaw guzinski
Copyright: (C) 2017, Radoslaw Guzinski
"""
import os
import time

from osgeo import gdal

import pyDMS.pyDMSUtils as utils
from pyDMS.pyDMS import DecisionTreeSharpener, NeuralNetworkSharpener
from pyDMS.pyDMS import REG_sklearn_ann

highResFilename = r"./example/S2_20180802T105621_REFL.tif"
lowResFilename = r"./example/S3_20180805T103824_LST.img"
outputFilename = r"./example/DMS_RF_20180805T103824_LST.tif"
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
##########################################################################################

if __name__ == "__main__":

    useDecisionTree = True

    commonOpts = {"highResFiles":               [highResFilename],
                  "lowResFiles":                [lowResFilename],
                  "cvHomogeneityThreshold":     0,
                  "movingWindowSize":           15,
                  "disaggregatingTemperature":  True}
    dtOpts =     {"method": "rf",
                  "perLeafLinearRegression":    True,
                  "linearRegressionExtrapolationRatio": 0.25,
                  "downsample": 5,
                  }
    ensembleOpts = {"n_jobs": 3,
                    "n_estimators": 10,
                    "bootstrap": False}
    sknnOpts =   {'hidden_layer_sizes':         (10,),
                  'activation':                 'tanh'}
    nnOpts =     {"regressionType":             REG_sklearn_ann,
                  "regressorOpt":               sknnOpts}
    gprOpts =    {"regressorOpt": {"kernel": 1.0 * RBF(length_scale=1.0,
                                      length_scale_bounds=(1e-3, 1e3))},
                  "chunk_size":                 20000}
    start_time = time.time()

    if useDecisionTree:
        opts = commonOpts.copy()
        opts.update(dtOpts)
        if dtOpts["method"] == "gpr":
            opts.update(gprOpts)
        elif dtOpts["method"] == "rf":
            opts["chunk_size"] = 20000
        disaggregator = DecisionTreeSharpener(**opts, ensembleOpt=ensembleOpts)
    else:
        opts = commonOpts.copy()
        opts.update(nnOpts)
        disaggregator = NeuralNetworkSharpener(**opts)

    print("Training regressor...")
    disaggregator.trainSharpener()
    print("Sharpening...")
    downscaledFile = disaggregator.applySharpener(highResFilename, lowResFilename)
    print("Residual analysis...")
    residualImage, correctedImage = disaggregator.residualAnalysis(downscaledFile, lowResFilename,
                                                                   doCorrection=True)
    print("Saving output...")
    highResFile = gdal.Open(highResFilename)
    if correctedImage is not None:
        outImage = correctedImage
    else:
        outImage = downscaledFile
    # outData = utils.binomialSmoother(outData)
    outFile = utils.saveImg(outImage.GetRasterBand(1).ReadAsArray(),
                            outImage.GetGeoTransform(),
                            outImage.GetProjection(),
                            outputFilename)
    residualFile = utils.saveImg(residualImage.GetRasterBand(1).ReadAsArray(),
                                 residualImage.GetGeoTransform(),
                                 residualImage.GetProjection(),
                                 os.path.splitext(outputFilename)[0] + "_residual" +
                                 os.path.splitext(outputFilename)[1])

    outFile = None
    residualFile = None
    downsaceldFile = None
    highResFile = None

    print(time.time() - start_time, "seconds")
