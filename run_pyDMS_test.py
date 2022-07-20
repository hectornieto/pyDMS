# -*- coding: utf-8 -*-
"""
@author: radoslaw guzinski
Copyright: (C) 2017, Radoslaw Guzinski
"""
import os
import time

from osgeo import gdal

import pyDMS.pyDMSUtils as utils
from pyDMS.pyDMS import DecisionTreeSharpener, NeuralNetworkSharpener, SupportVectorSharpener
from sklearn.gaussian_process.kernels import RBF

highResFilename = r"./example/S2_20180802T105621_REFL.tif"
lowResFilename = r"./example/S3_20180805T103824_LST.img"
outputFilename = r"./example/DMS_ANN_20180805T103824_LST.tif"
##########################################################################################

if __name__ == "__main__":

    commonOpts = {"method": "ann", # choose between ann, rf, xbg, dt, svr o gpr
                  "highResFiles":               [highResFilename],
                  "lowResFiles":                [lowResFilename],
                  "cvHomogeneityThreshold":     0,
                  "movingWindowSize":           0,
                  "disaggregatingTemperature":  True,
                  "export_stats":               True,
                  "perLeafLinearRegression":    False,
                  "linearRegressionExtrapolationRatio": 0.25,
                  "downsample": None,
                  }

    ensembleOpts = {"n_jobs": 3,
                    "n_estimators": 10,
                    "bootstrap": False}

    nnOpts =     {'hidden_layer_sizes':         (10,),
                  'activation':                 'tanh',
                  "max_iter":                   1000,
                  "chunk_size":                 10000}

    gprOpts =    {"kernel": 1.0 * RBF(length_scale=1.0,
                                      length_scale_bounds=(1e-3, 1e3)),
                  "chunk_size":                 20000}

    svrOpts =    {"n_jobs":                     7,
                  "chunk_size":                 10000}

    start_time = time.time()

    opts = commonOpts.copy()

    if opts["method"] == "gpr":
        opts["chunk_size"] = gprOpts.pop("chunk_size")
        opts["regressorOpt"] = gprOpts.copy()
    elif opts["method"] == "svr":
        opts["chunk_size"] = svrOpts.pop("chunk_size")
        opts["regressorOpt"] = svrOpts.copy()
    elif opts["method"] == "ann":
        opts["chunk_size"] = nnOpts.pop("chunk_size")	
        opts["regressorOpt"] = nnOpts.copy()

    disaggregator = DecisionTreeSharpener(**opts, ensembleOpt=ensembleOpts)
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
