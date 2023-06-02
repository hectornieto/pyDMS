# -*- coding: utf-8 -*-
"""
@author: radoslaw guzinski
Copyright: (C) 2017, Radoslaw Guzinski
"""
import os
import time

from osgeo import gdal

import pyDMS.pyDMSUtils as utils
from pyDMS.pyDMS import DecisionTreeSharpener, NeuralNetworkSharpener, RandomForestSharpener

highResFilename = r"./example/S2_20180802T105621_REFL.tif"
lowResFilename = r"./example/S3_20180805T103824_LST.img"
lowResMaskFilename = r""
outputFilename = r"./example/test_20180805T103824_DMS.tif"
method = "rf"
##########################################################################################

if __name__ == "__main__":

    commonOpts = {"highResFiles":               [highResFilename],
                  "lowResFiles":                [lowResFilename],
                  "cvHomogeneityThreshold":     0,
                  "movingWindowSize":           15,
                  "disaggregatingTemperature":  True,
                  "perLeafLinearRegression":    False,
                  "linearRegressionExtrapolationRatio": 0.25,
                  }
    dtOpts = {"perLeafLinearRegression": True,
              "linearRegressionExtrapolationRatio": 0.25}

    ensembleOpts = {"n_jobs": 3,
                    "n_estimators": 10,
                    "bootstrap": False}

    nnOpts =     {'hidden_layer_sizes':         (10,),
                  'activation':                 'tanh',
                  "max_iter":                   1000,
                  "chunk_size":                 10000}

    rfOpts =     {"n_estimators": 100,
                  "max_samples": 0.8,
                  "max_features": 0.8,
                  "n_jobs": 3}


    start_time = time.time()

    opts = commonOpts.copy()

    if method == "rf":
        opts["regressorOpt"] = rfOpts.copy()
        disaggregator = RandomForestSharpener(**opts)

    elif method == "ann":
        opts["chunk_size"] = nnOpts.pop("chunk_size")	
        opts["regressorOpt"] = nnOpts.copy()
        disaggregator = NeuralNetworkSharpener(**opts)
    else:
        opts["regressorOpt"] = dtOpts.copy()
        disaggregator = DecisionTreeSharpener(**opts)


    print("Training regressor...")
    disaggregator.trainSharpener()
    print("Sharpening...")
    downscaledFile = disaggregator.applySharpener(highResFilename, lowResFilename)
    print("Residual analysis...")
    residualImage, correctedImage = disaggregator.residualAnalysis(downscaledFile, lowResFilename,
                                                                   lowResMaskFilename,
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
