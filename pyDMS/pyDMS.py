# -*- coding: utf-8 -*-
"""
@author: Radoslaw Guzinski
Copyright: (C) 2017, Radoslaw Guzinski
"""

import math
import os

import numpy as np
from osgeo import gdal
from sklearn import tree, linear_model, ensemble, preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.cluster import KMeans
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
import pandas as pd

import pyDMS.pyDMSUtils as utils
from scipy import stats as st
import yaml

class RandomForestRegressorWLLR(RandomForestRegressor):
    def __init__(self,
                 n_estimators='warn',
                 criterion="squared_error",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=1.0,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 ):
        super().__init__(
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease


class GaussianProcessRegressorWLLR(GaussianProcessRegressor):
    def __init__(self,
                 kernel=1.0 * RBF(length_scale=1e1,
                                  length_scale_bounds=(1e-2, 1e4))\
                                   + WhiteKernel(noise_level=1,
                                   noise_level_bounds=(1e-5, 1e1)),
                 alpha=1e-10,
                 optimizer="fmin_l_bfgs_b",
                 n_restarts_optimizer=0,
                 normalize_y=True,
                 copy_X_train=True,
                 random_state=None,
                 ):
        super().__init__(
            kernel=kernel,
            alpha=alpha,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            copy_X_train=copy_X_train,
            random_state=random_state)


class DecisionTreeRegressorWithLinearLeafRegression(tree.DecisionTreeRegressor):
    ''' Decision tree regressor with added linear (bayesian ridge) regression
    for all the data points falling within each decision tree leaf node.

    Parameters
    ----------
    linearRegressionExtrapolationRatio: float (optional, default: 0.25)
        A limit on extrapolation allowed in the per-leaf linear regressions.
        The ratio is multiplied by the range of values present in each leaves'
        training dataset and added (substracted) to the maxiumum (minimum)
        value.

    decisionTreeRegressorOpt: dictionary (optional, default: {})
        Options to pass to DecisionTreeRegressor constructor. See
        http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
        for possibilities.

    Returns
    -------
    None
    '''
    def __init__(self,
                 criterion="mse",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 linearRegressionExtrapolationRatio=0.25):
        super(DecisionTreeRegressorWithLinearLeafRegression, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            )
        self.leafParameters = {}
        self.linearRegressionExtrapolationRatio = linearRegressionExtrapolationRatio

    def fit(self, X, y, sample_weight,  check_input=False, fitOpt={}):
        ''' Build a decision tree regressor from the training set (X, y).

        Parameters
        ----------
        X: array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            dtype=np.float32 and if a sparse matrix is provided to a sparse
            csc_matrix.

        y: array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (real numbers). Use dtype=np.float64 and
            order='C' for maximum efficiency.

        sample_weight: array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node.

        fitOpt: dictionary (optional, default: {})
            Options to pass to DecisionTreeRegressor fit function. See
            http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
            for possibilities.

        Returns
        -------
        Self
        '''

        # Fit a normal regression tree
        super(DecisionTreeRegressorWithLinearLeafRegression, self).fit(X, y,
                                                                       sample_weight,
                                                                       **fitOpt)

        # Create a linear regression for all input points which fall into
        # one output leaf
        predictedValues = super(DecisionTreeRegressorWithLinearLeafRegression, self).predict(X)
        leafValues = np.unique(predictedValues)
        for value in leafValues:
            ind = predictedValues == value
            leafLinearRegrsion = linear_model.BayesianRidge()
            leafLinearRegrsion.fit(X[ind, :], y[ind].ravel())
            self.leafParameters[value] = {"linearRegression": leafLinearRegrsion,
                                          "max": np.max(y[ind]),
                                          "min": np.min(y[ind])}

        return self

    def predict(self, X, predictOpt={}, check_input=False):
        ''' Predict class or regression value for X.

        Parameters
        ----------
        X: array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            dtype=np.float32 and if a sparse matrix is provided to a sparse
            csr_matrix.

        predictOpt: dictionary (optional, default: {})
            Options to pass to DecisionTreeRegressor predict function. See
            http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
            for possibilities.

        Returns
        -------
        y: array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        '''

        # Do normal regression tree prediction
        y = super(DecisionTreeRegressorWithLinearLeafRegression, self).predict(X, **predictOpt)

        # And also apply per-leaf linear regression
        for leafValue in self.leafParameters.keys():
            ind = y == leafValue
            if X[ind, :].size > 0:
                y[ind] = self.leafParameters[leafValue]["linearRegression"].predict(X[ind, :])
                # Limit extrapolation
                extrapolationRange = self.linearRegressionExtrapolationRatio * (
                                        self.leafParameters[leafValue]["max"] -
                                        self.leafParameters[leafValue]["min"])
                y[ind] = np.maximum(y[ind],
                                    self.leafParameters[leafValue]["min"] - extrapolationRange)
                y[ind] = np.minimum(y[ind],
                                    self.leafParameters[leafValue]["max"] + extrapolationRange)

        return y


class DecisionTreeSharpener(object):
    ''' Decision tree based sharpening (disaggregation) of low-resolution
    images using high-resolution images. The implementation is mostly based on [Gao2012].

    Decision tree based regressor is trained with high-resolution data resampled to
    low resolution and low-resolution data and then applied
    directly to high-resolution data to obtain high-resolution representation
    of the low-resolution data.

    The implementation includes selecting training data based on homogeneity
    statistics and using the homogeneity as weight factor ([Gao2012], section 2.2),
    performing linear regression with samples located within each regression
    tree leaf node ([Gao2012], section 2.1), using an ensemble of regression trees
    ([Gao2012], section 2.1), performing local (moving window) and global regression and
    combining them based on residuals ([Gao2012] section 2.3) and performing residual
    analysis and bias correction ([Gao2012], section 2.4)


    Parameters
    ----------
    highResFiles: list of strings
        A list of file paths to high-resolution images to be used during the
        training of the sharpener.

    lowResFiles: list of strings
        A list of file paths to low-resolution images to be used during the
        training of the sharpener. There must be one low-resolution image
        for each high-resolution image.

    lowResQualityFiles: list of strings (optional, default: [])
        A list of file paths to low-resolution quality images to be used to
        mask out low-quality low-resolution pixels during training. If provided
        there must be one quality image for each low-resolution image.

    lowResGoodQualityFlags: list of integers (optional, default: [])
        A list of values indicating which pixel values in the low-resolution
        quality images should be considered as good quality.

    cvHomogeneityThreshold: float (optional, default: 0)
        A threshold of coeficient of variation below which high-resolution
        pixels resampled to low-resolution are considered homogeneous and
        usable during the training of the disaggregator. If threshold is 0 or
        negative then it is set automatically such that 80% of pixels are below
        it.

    movingWindowSize: integer (optional, default: 0)
        The size of local regression moving window in low-resolution pixels. If
        set to 0 then only global regression is performed.

    disaggregatingTemperature: boolean (optional, default: False)
        Flag indicating whether the parameter to be disaggregated is
        temperature (e.g. land surface temperature). If that is the case then
        at some points it needs to be converted into radiance. This is becasue
        sensors measure energy, not temperature, plus radiance is the physical
        measurements it makes sense to average, while radiometric temperature
        behaviour is not linear.

    perLeafLinearRegression: boolean (optional, default: True)
        Flag indicating if linear regression should be performed on all data
        points falling within each regression tree leaf node.

    linearRegressionExtrapolationRatio: float (optional, default: 0.25)
        A limit on extrapolation allowed in the per-leaf linear regressions.
        The ratio is multiplied by the range of values present in each leaves'
        training dataset and added (substracted) to the maxiumum (minimum)
        value.

    regressorOpt: dictionary (optional, default: {})
        Options to pass to DecisionTreeRegressor constructor See
        http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
        for possibilities. Note that max_leaf_nodes and min_samples_leaf
        parameters will beoverwritten in the code.

    ensembleOpt: dictionary (optional, default: {})
        Options to pass to the selected ensemble constructor. See
        http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
        https://xgboost.readthedocs.io/en/latest/python/python_api.html
        for possibilities.

    Returns
    -------
    None


    References
    ----------
    .. [Gao2012] Gao, F., Kustas, W. P., & Anderson, M. C. (2012). A Data
       Mining Approach for Sharpening Thermal Satellite Imagery over Land.
       Remote Sensing, 4(11), 3287–3319. https://doi.org/10.3390/rs4113287
    '''
    def __init__(self,
                 highResFiles,
                 lowResFiles,
                 lowResQualityFiles=[],
                 lowResGoodQualityFlags=[],
                 cvHomogeneityThreshold=0,
                 movingWindowSize=0,
                 disaggregatingTemperature=False,
                 method="dt",
                 perLeafLinearRegression=True,
                 linearRegressionExtrapolationRatio=0.25,
                 regressorOpt={},
                 ensembleOpt={},
                 downsample=None,
                 chunk_size=1e9,
                 export_stats=False):

        self.highResFiles = highResFiles
        self.lowResFiles = lowResFiles
        self.lowResQualityFiles = lowResQualityFiles
        self.lowResGoodQualityFlags = lowResGoodQualityFlags

        if len(self.highResFiles) != len(self.lowResFiles):
            print("There must be a matching high resolution file for each low resolution file")
            raise IOError

        if len(self.lowResQualityFiles) == 0 or \
           (len(self.lowResQualityFiles) == 1 and self.lowResQualityFiles[0] == ""):
            self.useQuality_LR = False
        else:
            self.useQuality_LR = True

        if self.useQuality_LR and len(self.lowResQualityFiles) != len(self.lowResFiles):
            print("The number of quality files must be 0 or the same as number of low " +
                  "resolution files")
            raise IOError

        self.cvHomogeneityThreshold = cvHomogeneityThreshold
        # If threshold is 0 or negative then it is set automatically such that
        # 80% of pixels are below it.
        if self.cvHomogeneityThreshold <= 0:
            self.autoAdjustCvThreshold = True
            self.precentileThreshold = 80
        else:
            self.autoAdjustCvThreshold = False

        # Moving window size in low resolution pixels
        self.movingWindowSize = float(movingWindowSize)
        # The extension (on each side) by which sampling window size is larger
        # then prediction window size (see section 2.3 of Gao paper)
        self.movingWindowExtension = self.movingWindowSize * 0.25
        self.windowExtents = []

        self.disaggregatingTemperature = disaggregatingTemperature

        # Flag to determine whether a multivariate linear regression should be
        # constructed for samples in each leaf of the regression tree
        # (see section 2.1 of Gao paper)
        self.perLeafLinearRegression = perLeafLinearRegression
        self.linearRegressionExtrapolationRatio = linearRegressionExtrapolationRatio

        # Chosen machine learning method
        # Options: oneof ["dt", "rf", "xgb", "gpr"]
        self.method = method
        self.downsample = downsample
        self.regressorOpt = regressorOpt
        self.ensembleOpt = ensembleOpt
        self.chunk_size = np.int64(chunk_size)
        self.export_stats = export_stats

    def trainSharpener(self):
        ''' Train the sharpener using high- and low-resolution input files
        and settings specified in the constructor. Local (moving window) and
        global regression decision trees are trained with high-resolution data
        resampled to low resolution and low-resolution data. The training
        dataset is selected based on homogeneity of resampled high-resolution
        data being below specified threshold and quality mask (if given) of
        low resolution data. The homogeneity statistics are also used as weight
        factors for the training samples (more homogenous - higher weight).

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # Select good data (training samples) from low- and high-resolution
        # input images.
        fileNum = 0
        for highResFile, lowResFile in zip(self.highResFiles, self.lowResFiles):

            scene_HR = gdal.Open(highResFile)
            scene_LR = gdal.Open(lowResFile)

            # First subset and reproject low res scene to fit with
            # high res scene
            subsetScene_LR = utils.reprojectSubsetLowResScene(scene_HR, scene_LR)
            data_LR = subsetScene_LR.GetRasterBand(1).ReadAsArray()
            gt_LR = subsetScene_LR.GetGeoTransform()

            # Do the same with low res quality file (if provided) and flag
            # pixels which are considered to be of good quality
            if self.useQuality_LR:
                quality_LR = gdal.Open(self.lowResQualityFiles[fileNum])
                subsetQuality_LR = utils.reprojectSubsetLowResScene(scene_HR, quality_LR)
                subsetQualityMask = subsetQuality_LR.GetRasterBand(1).ReadAsArray()
                qualityPix = np.in1d(subsetQualityMask.ravel(),
                                     self.lowResGoodQualityFlags).reshape(subsetQualityMask.shape)
                del quality_LR
            else:
                qualityPix = np.ones(data_LR.shape).astype(bool)

            # Low resolution pixels with NaN value are always of bad quality
            qualityPix = np.logical_and(qualityPix, ~np.isnan(data_LR))

            # Then resample high res scene to low res pixel size while
            # extracting sub-low-res-pixel homogeneity statistics
            resMean, resStd = utils.resampleHighResToLowRes(scene_HR, subsetScene_LR)
            resMean[resMean == 0] = 0.000001
            resCV = np.sum(resStd/resMean, 2) / resMean.shape[2]
            resCV[np.isnan(resCV)] = 1000

            # Resampled high resolution pixels where at least one "parameter"
            # is NaN are also of bad quality
            resNaN = np.any(np.isnan(resMean), -1)
            qualityPix = np.logical_and(qualityPix, ~resNaN)

            windows = []
            extents = []
            # If moving window approach is used (section 2.3 of Gao paper)
            # then calculate the extent of each sampling window in low
            # resolution pixels
            if self.movingWindowSize > 0:
                for y in range(int(math.ceil(data_LR.shape[0]/self.movingWindowSize))):
                    for x in range(int(math.ceil(data_LR.shape[1]/self.movingWindowSize))):
                        windows.append([int(max(y*self.movingWindowSize-self.movingWindowExtension, 0)),
                                        int(min((y+1)*self.movingWindowSize+self.movingWindowExtension,
                                                data_LR.shape[0])),
                                        int(max(x*self.movingWindowSize-self.movingWindowExtension, 0)),
                                        int(min((x+1)*self.movingWindowSize+self.movingWindowExtension,
                                                data_LR.shape[1]))])
                        # Save the extents of this window in projection coordinates as
                        # UL and LR point coordinates
                        ul = utils.pix2point([x*self.movingWindowSize, y*self.movingWindowSize],
                                             gt_LR)
                        lr = utils.pix2point([(x+1)*self.movingWindowSize,
                                              (y+1)*self.movingWindowSize],
                                             gt_LR)
                        extents.append([ul, lr])

            # And always add the whole extent of low res image to also estimate
            # the regression tree for the whole image
            windows.append([0, data_LR.shape[0], 0, data_LR.shape[1]])

            goodData_LR = [None for _ in range(len(windows))]
            goodData_HR = [None for _ in range(len(windows))]
            weight = [None for _ in range(len(windows))]

            # For each window extract the good quality low res and high res pixels
            for i, window in enumerate(windows):
                rows = slice(window[0], window[1])
                cols = slice(window[2], window[3])
                qualityPixWindow = qualityPix[rows, cols]
                resCVWindow = resCV[rows, cols]

                # Good pixels are those where low res data quality is good and
                # high res data is homonogenous
                if self.autoAdjustCvThreshold:
                    g = np.logical_and.reduce((qualityPixWindow, resCVWindow < 1000,
                                               resCVWindow > 0))
                    if ~np.any(g):
                        self.cvHomogeneityThreshold = 0
                    else:
                        self.cvHomogeneityThreshold = np.percentile(resCVWindow[g],
                                                                    self.precentileThreshold)
                    print('Homogeneity CV threshold: %.2f' % self.cvHomogeneityThreshold)
                homogenousPix = np.logical_and(resCVWindow < self.cvHomogeneityThreshold,
                                               resCVWindow > 0)
                goodPix = np.logical_and(homogenousPix, qualityPixWindow)

                goodData_LR[i] = utils.appendNpArray(goodData_LR[i],
                                                     data_LR[rows, cols][goodPix])
                goodData_HR[i] = utils.appendNpArray(goodData_HR[i],
                                                     resMean[rows, cols, :][goodPix, :], axis=0)

                # Also estimate weight given to each pixel as the inverse of its
                # heterogeneity
                w = 1/resCVWindow[goodPix]
                weight[i] = utils.appendNpArray(weight[i], w)

                # Print some stats
                if np.prod(data_LR[rows, cols][qualityPixWindow].shape) > 0:
                    percentageUsedPixels = int(float(np.prod(goodData_LR[i].shape)) /
                                               float(np.prod(data_LR[rows, cols][qualityPixWindow].shape)) * 100)
                    print('Number of training elements for is ' +
                          str(np.prod(goodData_LR[i].shape)) + ' representing ' +
                          str(percentageUsedPixels)+'% of avaiable low-resolution data.')

            if self.export_stats:
                self.stats_dict = {"N_train": int(np.size(goodData_LR))}

            # Close all files
            del scene_HR
            del scene_LR
            del subsetScene_LR
            if self.useQuality_LR:
                subsetQuality_LR = None
            fileNum = fileNum + 1

        self.windowExtents = extents
        windowsNum = len(windows)

        # Once all the samples have been picked fit all the local and global
        # regressions
        self.reg = [None for _ in range(windowsNum)]
        for i in range(windowsNum):
            if i < windowsNum-1:
                local = True
            else:
                local = False
            if len(goodData_LR[i]) > 0:
                self.reg[i] = \
                    self._doFit(goodData_LR[i], goodData_HR[i], weight[i], local)


    def applySharpener(self, highResFilename, lowResFilename=None):
        ''' Apply the trained sharpener to a given high-resolution image to
        derive corresponding disaggregated low-resolution image. If local
        regressions were used during training then they will only be applied
        where their moving window extent overlaps with the high resolution
        image passed to this function. Global regression will be applied to the
        whole high-resolution image wihtout geographic constraints.

        Parameters
        ----------
        highResFilename: string
            Path to the high-resolution image file do be used during
            disaggregation.

        lowResFilename: string (optional, default: None)
            Path to the low-resolution image file corresponding to the
            high-resolution input file. If local regressions
            were trained and low-resolution filename is given then the local
            and global regressions will be combined based on residual values of
            the different regressions to the low-resolution image (see [Gao2012]
            2.3). If local regressions were trained and low-resolution
            filename is not given then only the local regressions will be used.


        Returns
        -------
        outImage: GDAL memory file object
            The file object contains an in-memory, georeferenced disaggregator
            output.
        '''

        # Open and read the high resolution input file
        highResFile = gdal.Open(highResFilename)
        inData = np.zeros((highResFile.RasterYSize, highResFile.RasterXSize,
                           highResFile.RasterCount))
        for band in range(highResFile.RasterCount):
            data = highResFile.GetRasterBand(band+1).ReadAsArray().astype(float)
            no_data = highResFile.GetRasterBand(band+1).GetNoDataValue()
            data[data == no_data] = np.nan
            inData[:, :, band] = data
        gt = highResFile.GetGeoTransform()

        shape = inData.shape
        ysize = shape[0]
        xsize = shape[1]

        # Temporarly get rid of NaN's
        nanInd = np.isnan(inData)
        inData[nanInd] = 0
        outWindowData = np.empty((ysize, xsize))*np.nan

        # Do the downscailing on the moving windows if there are any
        for i, extent in enumerate(self.windowExtents):
            if self.reg[i] is not None:
                [minX, minY] = utils.point2pix(extent[0], gt)  # UL
                [minX, minY] = [max(minX, 0), max(minY, 0)]
                [maxX, maxY] = utils.point2pix(extent[1], gt)  # LR
                [maxX, maxY] = [min(maxX, xsize), min(maxY, ysize)]
                windowInData = inData[minY:maxY, minX:maxX, :]
                outWindowData[minY:maxY, minX:maxX] = \
                    self._doPredict(windowInData, self.reg[i])

        # Do the downscailing on the whole input image
        if self.reg[-1] is not None:
            outFullData = self._doPredict(inData, self.reg[-1], chunk_size=self.chunk_size)
        else:
            outFullData = np.empty((ysize, xsize))*np.nan

        # Combine the windowed and whole image regressions
        # If there is no windowed regression just use the whole image regression
        if np.all(np.isnan(outWindowData)):
            outData = outFullData
        # If corresponding low resolution file is provided then combine the two
        # regressions based on residuals (see section 2.3 of Gao paper)
        elif lowResFilename is not None:
            lowResScene = gdal.Open(lowResFilename)
            outWindowScene = utils.saveImg(outWindowData,
                                           highResFile.GetGeoTransform(),
                                           highResFile.GetProjection(),
                                           "MEM")
            windowedResidual, _, _ = self._calculateResidual(outWindowScene, lowResScene)
            del outWindowScene
            outFullScene = utils.saveImg(outFullData,
                                         highResFile.GetGeoTransform(),
                                         highResFile.GetProjection(),
                                         "MEM")
            fullResidual, _, _ = self._calculateResidual(outFullScene, lowResScene)
            del outFullScene
            del lowResScene
            # windowed weight
            ww = (1/windowedResidual)**2/((1/windowedResidual)**2 + (1/fullResidual)**2)
            # full weight
            fw = 1 - ww
            outData = outWindowData*ww + outFullData*fw
        # Otherwised use just windowed regression
        else:
            outData = outWindowData

        # Fix NaN's
        nanInd = np.any(nanInd, -1)
        outData[nanInd] = np.nan

        outImage = utils.saveImg(outData,
                                 highResFile.GetGeoTransform(),
                                 highResFile.GetProjection(),
                                 "MEM")

        del highResFile
        del inData
        return outImage

    def residualAnalysis(self, disaggregatedFile, lowResFilename, lowResQualityFilename=None,
                         doCorrection=True):
        ''' Perform residual analysis and (optional) correction on the
        disaggregated file (see [Gao2012] 2.4).

        Parameters
        ----------
        disaggregatedFile: string or GDAL file object
            If string, path to the disaggregated image file; if gdal file
            object, the disaggregated image.

        lowResFilename: string
            Path to the low-resolution image file corresponding to the
            high-resolution disaggregated image.

        lowResQualityFilename: string (optional, default: None)
            Path to low-resolution quality image file. If provided then low
            quality values are masked out during residual analysis. Otherwise
            all values are considered to be of good quality.

        doCorrection: boolean (optional, default: True)
            Flag indication whether residual (bias) correction should be
            performed or not.


        Returns
        -------
        residualImage: GDAL memory file object
            The file object contains an in-memory, georeferenced residual image.

        correctedImage: GDAL memory file object
            The file object contains an in-memory, georeferenced residual
            corrected disaggregated image, or None if doCorrection was set to
            False.
        '''

        if not os.path.isfile(str(disaggregatedFile)):
            scene_HR = disaggregatedFile
        else:
            scene_HR = gdal.Open(disaggregatedFile)
        scene_LR = gdal.Open(lowResFilename)
        if lowResQualityFilename is not None:
            quality_LR = gdal.Open(lowResQualityFilename)
        else:
            quality_LR = None

        residual_HR, residual_LR, gt_res = self._calculateResidual(scene_HR, scene_LR, quality_LR)

        if self.disaggregatingTemperature:
            if doCorrection:
                corrected = (residual_HR + scene_HR.GetRasterBand(1).ReadAsArray()**4)**0.25
                correctedImage = utils.saveImg(corrected,
                                               scene_HR.GetGeoTransform(),
                                               scene_HR.GetProjection(),
                                               "MEM")
            else:
                correctedImage = None
            # Convert residual back to temperature for easier visualisation
            residual_LR = (residual_LR + 273.15**4)**0.25 - 273.15
        else:
            if doCorrection:
                corrected = residual_HR + scene_HR.GetRasterBand(1).ReadAsArray()
                correctedImage = utils.saveImg(corrected,
                                               scene_HR.GetGeoTransform(),
                                               scene_HR.GetProjection(),
                                               "MEM")
            else:
                correctedImage = None

        residualImage = utils.saveImg(residual_LR,
                                      gt_res,
                                      scene_HR.GetProjection(),
                                      "MEM")

        bias = float(np.nanmean(residual_LR))
        rmsd = float(np.nanmean(residual_LR**2)**0.5)
        print("LR residual bias: "+str(bias))
        print("LR residual RMSD: "+str(rmsd))
        if self.export_stats:
            valid = np.isfinite(residual_LR)
            self.stats_dict["N_test"] = int(np.sum(valid))
            self.stats_dict["bias"] = bias
            self.stats_dict["RMSD"] = rmsd
            with open("DMS_stats.yaml", "w") as fid:
                yaml.dump(self.stats_dict, fid)

        del scene_HR
        del scene_LR
        del quality_LR

        return residualImage, correctedImage

    def _doFit(self, goodData_LR, goodData_HR, weight, local):
        ''' Private function. Fits the regression tree.
        '''

        # For local regression constrain the number of tree
        # nodes (rules) - section 2.3
        if self.method == "dt" or self.method == "rf" or self.method == "xbg":
            if local:
                self.regressorOpt["max_leaf_nodes"] = 10
            else:
                self.regressorOpt["max_leaf_nodes"] = 30
            self.regressorOpt["min_samples_leaf"] = 10

        # If per leaf linear regression is used then use modified
        # DecisionTreeRegressor. Otherwise use the standard one.
        if self.downsample and not local:
            goodData_HR, goodData_LR, weight = downsample_dataset_aboulalebi(goodData_HR,
                                                                  goodData_LR,
                                                                  weight,
                                                                  self.downsample)
            if self.export_stats:
                self.stats_dict["N_train_subsampled"] = int(np.size(goodData_LR))

        if self.method == "dt":
            hr_scaler = None
            lr_scaler = None
            if self.perLeafLinearRegression:
                baseRegressor = \
                    DecisionTreeRegressorWithLinearLeafRegression(
                        **self.regressorOpt,
                        linearRegressionExtrapolationRatio=self.linearRegressionExtrapolationRatio
                    )
            else:
                baseRegressor = tree.DecisionTreeRegressor(**self.regressorOpt)

            reg = ensemble.BaggingRegressor(baseRegressor, **self.ensembleOpt)
            print("Fitting Ensemble Decision Tree Regressor")
            reg.fit(goodData_HR, goodData_LR, sample_weight=weight)

        elif self.method == "rf":
            hr_scaler = None
            lr_scaler = None
            reg = RandomForestRegressorWLLR(
                **self.ensembleOpt,
                **self.regressorOpt
                )
            print("Fitting Random Forest Regressor")
            reg.fit(goodData_HR, goodData_LR, sample_weight=weight)

        elif self.method == "xgb":
            hr_scaler = None
            lr_scaler = None
            reg = xgboost.XGBRegressor(**self.ensembleOpt)
            print("Fitting XGboost Regressor")
            reg.fit(goodData_HR, goodData_LR, sample_weight=weight)

        elif self.method == "gpr":
            hr_scaler = preprocessing.StandardScaler()
            lr_scaler = None
            goodData_HR = hr_scaler.fit_transform(goodData_HR)
            reg = GaussianProcessRegressorWLLR(**self.regressorOpt)
            print("Fitting Gaussian Process Regressor")
            reg.fit(goodData_HR, goodData_LR)

        elif self.method == "svr":
            hr_scaler = preprocessing.StandardScaler()
            goodData_HR = hr_scaler.fit_transform(goodData_HR)
            lr_scaler = preprocessing.StandardScaler()
            print("Fitting Support Vector Regressor")
            goodData_LR = lr_scaler.fit_transform(goodData_LR.reshape(-1, 1))
            reg = SupportVectorSharpener(**self.regressorOpt)
            reg.fit(goodData_HR, np.ravel(goodData_LR), sample_weight=weight)

        elif self.method == "ann":
            hr_scaler = preprocessing.StandardScaler()
            goodData_HR = hr_scaler.fit_transform(goodData_HR)
            lr_scaler = preprocessing.StandardScaler()
            print("Fitting MLP Artificial Neural Network")
            goodData_LR = lr_scaler.fit_transform(goodData_LR.reshape(-1, 1))
            reg = NeuralNetworkSharpener(**self.regressorOpt)
            reg.fit(goodData_HR, np.ravel(goodData_LR))

        else:
            raise TypeError("Method should be one of dt, rf, xgb or gpr")

        return {"reg": reg, "HR_scaler": hr_scaler, "LR_scaler": lr_scaler}

    def _doPredict(self, inData, reg, chunk_size=None):
        ''' Private function. Calls the regression tree.
        '''
        hr_scaler = reg.pop("HR_scaler")
        lr_scaler = reg.pop("LR_scaler")
        reg = reg["reg"]
        origShape = inData.shape
        if len(origShape) == 3:
            bands = origShape[2]
        else:
            bands = 1
        # Do the actual decision tree regression
        inData = inData.reshape((-1, bands))
        if hr_scaler:
            inData = hr_scaler.transform(inData)

        if chunk_size:
            n_pixels = inData.shape[0]
            outData = np.empty(inData.shape[0])
            n_chunks = np.int(np.ceil(n_pixels / chunk_size))
            for chunk in range(n_chunks):
                ini = np.int(chunk * chunk_size)
                end = np.int(np.minimum((chunk + 1) * chunk_size, n_pixels))
                print(f"Predicting on chunk {chunk} out of {n_chunks}", end="\r")
                outData[ini:end] = reg.predict(inData[ini:end, :])
            print(f"Finished predicting on {n_pixels} cases", end="\n")
        else:
            outData = reg.predict(inData)

        if lr_scaler:
            outData = lr_scaler.inverse_transform(outData.reshape([-1, 1]))

        outData = outData.reshape((origShape[0], origShape[1]))

        return outData

    def _calculateResidual(self, downscaledScene, originalScene, originalSceneQuality=None):
        ''' Private function. Calculates residual between overlapping
            high-resolution and low-resolution images.
        '''

        # First subset and reproject original (low res) scene to fit with
        # downscaled (high res) scene
        subsetScene_LR = utils.reprojectSubsetLowResScene(downscaledScene,
                                                          originalScene,
                                                          resampleAlg=gdal.GRA_NearestNeighbour)
        data_LR = subsetScene_LR.GetRasterBand(1).ReadAsArray().astype(float)
        gt_LR = subsetScene_LR.GetGeoTransform()

        # If quality file for the low res scene is provided then mask out all
        # bad quality pixels in the subsetted LR scene. Otherwise assume that all
        # low res pixels are of good quality.
        if originalSceneQuality is not None:
            subsetQuality_LR = utils.reprojectSubsetLowResScene(downscaledScene,
                                                                originalSceneQuality,
                                                                resampleAlg=gdal.GRA_NearestNeighbour)
            goodPixMask_LR = subsetQuality_LR.GetRasterBand(1).ReadAsArray()
            goodPixMask_LR = np.in1d(goodPixMask_LR.ravel(),
                                     self.lowResGoodQualityFlags).reshape(goodPixMask_LR.shape)
            data_LR[~goodPixMask_LR] = np.nan
            del subsetQuality_LR

        # Then resample high res scene to low res pixel size
        if self.disaggregatingTemperature:
            # When working with tempratures they should be converted to
            # radiance values before aggregating to be physically accurate.
            radianceScene = utils.saveImg(downscaledScene.GetRasterBand(1).ReadAsArray()**4,
                                          downscaledScene.GetGeoTransform(),
                                          downscaledScene.GetProjection(),
                                          "MEM")
            resMean, _ = utils.resampleHighResToLowRes(radianceScene,
                                                       subsetScene_LR)
            # Find the residual (difference) between the two)
            residual_LR = data_LR**4 - resMean[:, :, 0]
        else:
            resMean, _ = utils.resampleHighResToLowRes(downscaledScene,
                                                       subsetScene_LR)
            # Find the residual (difference) between the two
            residual_LR = data_LR - resMean[:, :, 0]

        # Smooth the residual and resample to high resolution
        residual = utils.binomialSmoother(residual_LR)
        residualDs = utils.saveImg(residual, subsetScene_LR.GetGeoTransform(),
                                   subsetScene_LR.GetProjection(), "MEM")
        residualScene_NN = utils.resampleWithGdalWarp(residualDs, downscaledScene,
                                                      resampleAlg="near")
        residualScene_BL = utils.resampleWithGdalWarp(residualDs, downscaledScene,
                                                      resampleAlg="bilinear")
        del residualDs

        # Bilinear resampling extrapolates by half a pixel, so need to clean it up
        residual = residualScene_BL.GetRasterBand(1).ReadAsArray()
        residual[np.isnan(residualScene_NN.GetRasterBand(1).ReadAsArray())] = np.NaN
        del residualScene_NN
        del residualScene_BL

        # The residual array might be slightly smaller then the downscaled because
        # of the subsetting of the low resolution scene. In that case just pad
        # the missing values with neighbours.
        downscaled = downscaledScene.GetRasterBand(1).ReadAsArray()
        if downscaled.shape != residual.shape:
            temp = np.zeros(downscaled.shape)
            temp[:residual.shape[0], :residual.shape[1]] = residual
            temp[residual.shape[0]:, :] = \
                temp[2*(residual.shape[0] - downscaled.shape[0]):residual.shape[0] - downscaled.shape[0], :]
            temp[:, residual.shape[1]:] = \
                temp[:, 2*(residual.shape[1] - downscaled.shape[1]):residual.shape[1] - downscaled.shape[1]]

            residual = temp

        del subsetScene_LR

        return residual, residual_LR, gt_LR


class NeuralNetworkSharpener(MLPRegressor):
    def __init__(self,
                 **mlp_kwargs):

        super(NeuralNetworkSharpener, self).__init__(**mlp_kwargs)


class SupportVectorSharpener(RandomizedSearchCV):

    def __init__(self,
                 kernel=['rbf'],
                 c=st.expon(scale=100),
                 gamma=st.expon(scale=.1),
                 tol=0.001,
                 shrinking=True,
                 cache_size=200,
                 max_iter=-1,
                 n_iter=10,
                 scoring=None,
                 n_jobs=None,
                 refit=True,
                 cv=None,
                 verbose=False,
                 pre_dispatch='2*n_jobs',
                 random_state=None,
                 error_score=np.nan,
                 return_train_score=False):

        base_estimator = SVR(tol=tol,
                             shrinking=shrinking,
                             cache_size=cache_size,
                             max_iter=max_iter)

        param_distributions = {"kernel": kernel, "C": c, "gamma": gamma}
        super(SupportVectorSharpener, self).__init__(base_estimator,
                                                     param_distributions,
                                                     n_iter=n_iter,
                                                     scoring=scoring,
                                                     n_jobs=n_jobs,
                                                     refit=refit,
                                                     cv=cv,
                                                     verbose=verbose,
                                                     pre_dispatch=pre_dispatch,
                                                     random_state=random_state,
                                                     error_score=error_score,
                                                     return_train_score=return_train_score)


def downsample_dataset(x_array, y_array, w, subsample=10):
    n_cases = x_array.shape[0]
    upper = np.percentile(y_array, 100 - subsample / 2.)
    lower = np.percentile(y_array, subsample / 2.)
    to_downsample = np.logical_and(y_array > lower,
                                   y_array < upper)
    downsampled_x = x_array[to_downsample]
    downsampled_y = y_array[to_downsample]
    downsampled_w = w[to_downsample]
    extreme_cases = np.sum(~to_downsample)
    print(f"Downsample majority values to fit {extreme_cases} cases of extreme temperatures") 
    sampling = np.random.uniform(size=np.sum(to_downsample))
    sampling = sampling < subsample / 100.
    print(f"Downsampled majority values to {np.sum(sampling)} cases")
    x_array = np.append(x_array[~to_downsample], 
                        downsampled_x[sampling], axis=0)
            
    y_array = np.append(y_array[~to_downsample], 
                        downsampled_y[sampling], axis=0)
    w = np.append(w[~to_downsample],
                  downsampled_w[sampling], axis=0)
    return x_array, y_array, w


def downsample_dataset_aboulalebi(x_array, y_array, w,
                                  n_clusters=5,
                                  subsample=1,
                                  p=2):
    import matplotlib.pyplot as plt
    n_cases = y_array.shape
    sampling = np.random.uniform(size=np.sum(n_cases))
    sampling = sampling <= subsample
    x_array = x_array[sampling]
    y_array = y_array[sampling]
    # plt.figure()
    # plt.hist(y_array, bins=n_clusters, histtype="step", color="b", density=True)
    w = w[sampling]
    clusters = KMeans(n_clusters=n_clusters).fit(np.concatenate([x_array,
                                                                 y_array.reshape(-1, 1)],
                                                                axis=1))

    counts = np.unique(clusters.labels_, return_counts=True)[1]
    n_min = np.min(counts)
    for i in range(n_clusters):
        cluster = clusters.labels_ == i
        ids = np.argwhere(cluster).reshape(-1)
        sampled = np.random.choice(ids, size=n_min, replace=False)
        if i == 0:
            sampled_x = x_array[sampled]
            sampled_y = y_array[sampled]
            sampled_w = w[sampled]
        else:
            sampled_x = np.append(sampled_x, x_array[sampled], axis=0)
            sampled_y = np.append(sampled_y, y_array[sampled], axis=0)
            sampled_w = np.append(sampled_w, w[sampled], axis=0)
    print(f"Downsampled values to {np.size(sampled_y)} cases")
    # plt.hist(sampled_y, bins=n_clusters, histtype="step", color="r", density=True)
    # plt.show()
    return sampled_x, sampled_y, sampled_w