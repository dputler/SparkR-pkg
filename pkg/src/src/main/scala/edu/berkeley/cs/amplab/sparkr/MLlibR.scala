package edu.berkeley.cs.amplab.sparkr

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.Utils
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.types._
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import org.apache.spark.api.python.SerDeUtil
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.feature._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.random.{RandomRDDs => RG}
import org.apache.spark.mllib.recommendation._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.mllib.stat.correlation.CorrelationNames
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian
import org.apache.spark.mllib.stat.test.ChiSqTestResult
import org.apache.spark.mllib.tree.{GradientBoostedTrees, RandomForest, DecisionTree}
import org.apache.spark.mllib.tree.configuration.{BoostingStrategy, Algo, Strategy}
import org.apache.spark.mllib.tree.impurity._
import org.apache.spark.mllib.tree.loss.Losses
import org.apache.spark.mllib.tree.model.{GradientBoostedTreesModel, RandomForestModel, DecisionTreeModel}
import org.apache.spark.mllib.util.MLUtils

object MLlibR {

  //
  // Helper Methods
  //

  // A method to convert a DataFrame to an RDD[LabeledPoint], which is needed to
  // estimate a model by MLlib and evaluate the model in the validation sample.
  // THIS PROBABLY SHOULD BE PART OF MLlib.
  def dfToLabeledPoints(df: DataFrame): RDD[LabeledPoint] = {
    df.map { thisRow =>
      var thisArray = new Array[Double](thisRow.length - 1)
      for (colInd <- 1 until thisRow.length) {
        thisArray(colInd - 1) = thisRow.getDouble(colInd)
      }
      val features = new DenseVector(thisArray)
      LabeledPoint(thisRow.getDouble(0), features)
    }
  }

  // A set of methods to get an RDD of tuples of scores and the actual labels
  // for a model, which is needed for model evaluation methods in Mllib.
  // THIS PROBABLY SHOULD BE PART OF MLlib.
  def ScoresLabels(modObj: LogisticRegressionModel,
                  vectors: RDD[LabeledPoint],
                  threshold: Double = 0.5): RDD[(Double, Double)] = {
    if (threshold == 0.0) {
      modObj.clearThreshold()
    } else {
      modObj.setThreshold(threshold)
    }
    vectors.map { point =>
      (modObj.predict(point.features), point.label)
    }
  }

  def ScoresLabels(modObj: DecisionTreeModel,
                  vectors: RDD[LabeledPoint]): RDD[(Double, Double)] = {
//    if (threshold == 0.0) {
//      modObj.clearThreshold()
//    } else {
//      modObj.setThreshold(threshold)
//    }
    vectors.map { point =>
      (modObj.predict(point.features), point.label)
    }
  }

  // Create the IdPoint class to enable predicted scores to be joined to a
  // table. THIS PROBABLY SHOULD BE PART OF MLlib.
  case class IdPoint(id: String, features: Vector)

  // A method to create an RDD[IdPoint] object from a DataFrame
  def dfToIdPoints(df: DataFrame): RDD[IdPoint] = {
    df.map { thisRow =>
      var thisArray = new Array[Double](thisRow.length - 1)
      for (colInd <- 1 until thisRow.length) {
        thisArray(colInd - 1) = thisRow.getDouble(colInd)
      }
      val features = new DenseVector(thisArray)
      IdPoint(thisRow.getString(0), features)
    }
  }

  // A set of methods to get the coefficients of regression models into an
  // Array[Double] structure that can be imported into R
  def getCoefs(modelObj: LinearRegressionModel) : Array[Double] = {
    var coefArray = new Array[Double](modelObj.weights.size)
    for (ind <- 0 until modelObj.weights.size) {
      coefArray(ind) = modelObj.weights(ind)
    }
    coefArray
  }
  def getCoefs(modelObj: LogisticRegressionModel) : Array[Double] = {
    var coefArray = new Array[Double](modelObj.weights.size)
    for (ind <- 0 until modelObj.weights.size) {
      coefArray(ind) = modelObj.weights(ind)
    }
    coefArray
  }

  // A method to calculate the misclassification rate of a model
  def misClassRate(sl: RDD[(Double, Double)]): Double = {
    sl.filter(tuple => tuple._1 != tuple._2).count.toDouble/sl.count().toDouble
  }

  // A method to calculate a binary confusion matrix. THIS PROBABLY SHOULD BE
  // PART OF MLlib.
  def binaryConfusionMatrix(scoreLabels: RDD[(Double, Double)]): Array[Double] = {
    val outcomes = scoreLabels.map { tuple =>
    // set the eventual cells up such that the rows are the actual outcomes, and
    // the columns are the predicted outcomes
      val cell00 = if (tuple._1 == 0.0 && tuple._2 == 0.0) 1.0 else 0.0
      val cell01 = if (tuple._1 == 1.0 && tuple._2 == 0.0) 1.0 else 0.0
      val cell10 = if (tuple._1 == 0.0 && tuple._2 == 1.0) 1.0 else 0.0
      val cell11 = if (tuple._1 == 1.0 && tuple._2 == 1.0) 1.0 else 0.0
      (cell00, cell01, cell10, cell11)
    }

    var s00 = 0.0
    var s01 = 0.0
    var s10 = 0.0
    var s11 = 0.0
    val theArray = outcomes.collect
    for (ind <- 0 until theArray.length) {
      val thisTuple = theArray(ind)
      s00 = thisTuple._1 + s00
      s01 = thisTuple._2 + s01
      s10 = thisTuple._3 + s10
      s11 = thisTuple._4 + s11
    }
    Array[Double](s00, s01, s10, s11)
  }

  // A method to calculate the residual deviance and null deviance of a binary
  // classication model.  THIS PROBABLY SHOULD BE PART OF MLlib.
  def binaryClassificationDeviance(slProb: RDD[(Double, Double)]): Array[Double] = {
      // Calculate the percentage of 1.0 responses
      val slProbArray = slProb.collect
      var pct1 = 0.0
      for (ind1 <- 0 until slProbArray.length) {
        val thisTupple = slProbArray(ind1)
        pct1 = thisTupple._2 + pct1
      }
      pct1 = pct1/slProbArray.length.toDouble
      // Calculate each individual record's contribution to the deviances
      val indDev = slProb.map { tuple =>
        val dev = (1.0 - tuple._2)*scala.math.log(1.0 - tuple._1) + tuple._2*scala.math.log(tuple._1)
        val ndev = (1.0 - tuple._2)*scala.math.log(1.0 - pct1) + tuple._2*scala.math.log(pct1)
        (dev, ndev)
      }
      // Sum up the individual deviance contributions
      val devArray = indDev.collect
      var tdev = 0.0
      var tndev = 0.0
      for (ind2 <- 0 until devArray.length) {
        val thisT = devArray(ind2)
        tdev = thisT._1 + tdev
        tndev = thisT._2 + tndev
      }
      tdev = -2.0*tdev
      tndev = -2.0*tndev
      Array[Double](tdev, tndev)
    }

  // A method to enable the creation of a BinaryClassificationMetrics object
  def BCMetrics(sl: RDD[(Double, Double)]): BinaryClassificationMetrics = {
    new BinaryClassificationMetrics(sl)
  }

  def getUpdaterFromString(regType: String): Updater = {
    if (regType == "l2") {
      new SquaredL2Updater
    } else if (regType == "l1") {
      new L1Updater
    } else if (regType == null || regType == "none") {
      new SimpleUpdater
    } else {
      throw new IllegalArgumentException("Invalid value for 'regType' paramter."
        + " This value can only be initialized using the string values 'l1', 'l2', or none.")
    }
  }

  //
  // Model APIs
  //

  // A method to estimate a logistic regression model using L-BFGS optimization
  def trainLogisticRegressionModelWithLBFGS(
      data: RDD[LabeledPoint],
      numIterations: Int,
      iWeights: Array[Double],
      regParam: Double,
      regType: String,
      intercept: Int,
      corrections: Int,
      tolerance: Double): LogisticRegressionModel = {
        val newIntrcpt = if (intercept == 1) {
          true
        } else {
          false
        }
        val initialWeights = new DenseVector(iWeights)
        val LogRegAlg = new LogisticRegressionWithLBFGS()
        LogRegAlg.setIntercept(newIntrcpt)
        LogRegAlg.optimizer
          .setNumIterations(numIterations)
          .setRegParam(regParam)
          .setNumCorrections(corrections)
          .setConvergenceTol(tolerance)
        LogRegAlg.optimizer.setUpdater(getUpdaterFromString(regType))
        LogRegAlg.run(data)
      }

  // A method to estimate a linear regression model using SGD optimization
  def trainLinearRegressionWithSGD(
        data: RDD[LabeledPoint],
        numIterations: Int,
        stepSize: Double,
        miniBatchFraction: Double,
        iWeights: Array[Double],
        regParam: Double,
        regType: String,
        intercept: Int): LinearRegressionModel = {
        val newIntrcpt = if (intercept == 1) {
          true
        } else {
          false
        }
        val initialWeights = new DenseVector(iWeights)
        val linRegAlg = new LinearRegressionWithSGD()
        linRegAlg.setIntercept(newIntrcpt)
        linRegAlg.optimizer
          .setNumIterations(numIterations)
          .setRegParam(regParam)
          .setStepSize(stepSize)
          .setMiniBatchFraction(miniBatchFraction)
        linRegAlg.optimizer.setUpdater(getUpdaterFromString(regType))
        linRegAlg.run(data, initialWeights)
    }


  // A method to estimate decision tree models
  // TODO: Properly handle the categorical features
  def trainClassificationTree(
      data: RDD[LabeledPoint],
      numClasses: Int,
      impurity: String,
      maxDepth: Int,
      maxBins: Int): DecisionTreeModel = {
        val categoricalFeaturesInfo = Map[Int, Int]()
        val dtModel = DecisionTree.trainClassifier(data,
                      numClasses,
                      categoricalFeaturesInfo,
                      impurity,
                      maxDepth,
                      maxBins)
        dtModel
  }

  //
  // Prediction method APIs
  //

  // Create scores case class for use with createDataFrame
  case class scores(id : String, score: Double)

  def IdScore(modObj: LogisticRegressionModel,
              vectors: RDD[IdPoint],
              sqlCtx: SQLContext): DataFrame = {
    modObj.clearThreshold()
    sqlCtx.createDataFrame(
      vectors.map { point =>
        scores(point.id, modObj.predict(point.features))
    })
  }

  def IdScore(modObj: LinearRegressionModel,
              vectors: RDD[IdPoint],
              sqlCtx: SQLContext): DataFrame = {
    sqlCtx.createDataFrame(
      vectors.map { point =>
        scores(point.id, modObj.predict(point.features))
    })
  }

  def IdScore(modObj: DecisionTreeModel,
              vectors: RDD[IdPoint],
              sqlCtx: SQLContext): DataFrame = {
    sqlCtx.createDataFrame(
      vectors.map { point =>
        scores(point.id, modObj.predict(point.features))
    })
  }
}
