##
## Helper functions
##

# Parse an R formula so it can be incorporated into a SparkSQL query to get the
# target and relevant predictors and to determine if the model should include an
# intercept. This is not yet provide all features of an R formula
parseFormula <- function(formula) {
  if (class(formula) != "formula") {
    stop("The provided argument is not a formula.")
  }
  formula.parts <- as.character(formula)
  preds <- unlist(strsplit(unlist(strsplit(formula.parts[3], " \\+ ")), " \\- "))
  intercept <- if (any(preds %in% c("1", "-1"))) {
    0L
  } else {
    1L
  }
  preds <- preds[!(preds %in% c("1", "-1"))]
  vars <- c(formula.parts[2], preds)
  list(as.list(vars), intercept)
}

# Turn a Spark DataFrame into an object that is RDD[LabeledPoint], which is
# needed by many MLlib methods.
dfToLabeledPoints <- function(df) {
  if (class(df) != "DataFrame") {
    stop("The provided argument is not a Spark DataFrame.")
  }
  lp <- SparkR:::callJStatic("edu.berkeley.cs.amplab.sparkr.MLlibR", "dfToLabeledPoints", df@sdf)
  lp
}

# Turn a Spark DataFrame into an object that is RDD[IdPoint], which is
# needed to join model scores to other Spark DataFrames for implementation
# purposes.
dfToIdPoints <- function(df) {
  if (class(df) != "DataFrame") {
    stop("The provided argument is not a Spark DataFrame.")
  }
  ip <- SparkR:::callJStatic("edu.berkeley.cs.amplab.sparkr.MLlibR", "dfToIdPoints", df@sdf)
  ip
}

# A function to get the class name of a jobj
getJClassName <- function(obj) {
  if (class(obj) != "jobj") {
    stop("The provided object is not a Java object")
  }
  a <- SparkR:::callJMethod(obj, "getClass")
  b <- unlist(strsplit(SparkR:::callJMethod(a, "getName"),"\\."))
  b[length(b)]
}

##
## MLlib model API functions
##

# A function for training a linear regression model
linearRegressionWithSGD <- function(formula,
                                    df,
                                    sqlCtx,
                                    iter = 100L,
                                    step = 1,
                                    batch_frac = 1,
                                    start_vals = NULL,
                                    reg_param = 0,
                                    reg_type = "none") {
  # Initial input error checking
  if (class(formula) != "formula") {
    stop("The provided formula is not a formula object.")
  }
  if (class(df) != "DataFrame") {
    stop("The data (df) must be a Spark DataFrame.")
  }
  if (is.null(batch_frac)) {
    batch_frac <- 1
  }
  batch_frac <- as.numeric(batch_frac)
  if (batch_frac > 1) {
    message("The minimum banch fraction cannot exeed 1.")
    batch_frac <- 1
  }
  if (batch_frac <= 0) {
    stop("The minimum banch fraction must be strictly positive.")
  }
  reg_param <- as.numeric(reg_param)
  if (!(reg_type %in% c("none", "l1", "l2"))) {
    stop("The provided regularization type must be one of 'none', 'l1', and 'l2'.")
  }
  # Parse the formula and prep/check the start values
  pf <- SparkR:::parseFormula(formula)
  fields <- pf[[1]]
  if (is.null(start_vals)) {
    start_vals <- rep(0, length(fields))
    if (pf[[2]] == 0) {
      start_vals <- start_vals[-1]
    }
  }
  length_start <- length(fields) - 1
  #if (pf[[2]] == 0) {
  #  length_start <- length_start - 1
  #}
  if (length(start_vals) != length_start) {
    stop(paste(length(start_vals), "start values were provided when", length_start, "are needed."))
  }
  start_vals <- as.list(as.numeric(start_vals))
  step <- as.numeric(step)
  # Get the model call
  the_call <- match.call()
  # Prepare the data
  estDF <- select(df, fields)
  estLP <- dfToLabeledPoints(estDF)
  SparkR:::callJMethod(estLP, "cache")
  # Estimate the model
  the_model <- SparkR:::callJStatic("edu.berkeley.cs.amplab.sparkr.MLlibR",
                                    "trainLinearRegressionWithSGD",
                                    estLP,
                                    iter,
                                    step,
                                    batch_frac,
                                    start_vals,
                                    reg_param,
                                    reg_type,
                                    as.integer(pf[[2]]))
  obj <- list(Model = the_model, Data = estLP, Fields = fields, call = the_call)
  class(obj) <- "LinearRegressionModel"
  obj
}

# The API for a logistic regression model estimated using the limited memory
# version of the BFGS optimization algorithm
logisticRegressionWithLBFGS <- function(formula,
                                        df,
                                        sqlCtx,
                                        iter = 100L,
                                        start_vals = NULL,
                                        reg_param = 0.0,
                                        reg_type = "none",
                                        corrections = 10L,
                                        tol = 1e-4) {
  # Input error checking
  if (class(formula) != "formula") {
    stop("The provided formula is not a formula object.")
  }
  if (class(df) != "DataFrame") {
    stop("The data (df) must be a Spark DataFrame.")
  }
  if (length(iter) != 1) {
    stop("The value of iter (the number of iterations) should be a single integer")
  }
  iter <- as.integer(iter)
  if (iter < 1L) {
    stop("The number of iterations must be strictly positive.")
  }
  if (length(reg_param) != 1) {
    stop("The value of iter (the number of iterations) should be a single numeric")
  }
  if (reg_param < 0) {
    stop("The regularization parameter cannot be negative.")
  }
  if (corrections < 1L) {
    stop("The number of corrections must be strictly positive.")
  }
  if (!(reg_type %in% c("none", "l2"))) {
    stop("The value of reg_type must be either 'none' or 'l2'.")
  }

  # Parse the formula and prepare the data
  the_call <- match.call()
  pf <- parseFormula(formula)
  fields <- pf[[1]]
  if (is.null(start_vals)) {
    start_vals <- rep(0, length(fields))
    if (pf[[2]] == 0) {
      start_vals <- start_vals[-1]
    }
  }
  length_start <- length(fields)
  if (pf[[2]] == 0) {
    length_start <- length_start - 1
  }
  if (length(start_vals) != length_start) {
    stop(paste(length(start_vals), "start values were provided when", length_start, "are needed."))
  }
  start_vals <- as.list(as.numeric(start_vals))
  estDF <- select(df, fields)
  estLP <- dfToLabeledPoints(estDF)
  SparkR:::callJMethod(estLP, "cache")
  use_intercept <- pf[[2]]
  the_model <- SparkR:::callJStatic("edu.berkeley.cs.amplab.sparkr.MLlibR",
                                    "trainLogisticRegressionModelWithLBFGS",
                                    estLP,
                                    iter,
                                    start_vals,
                                    reg_param,
                                    reg_type,
                                    use_intercept,
                                    corrections,
                                    tol)
  obj <- list(Model = the_model, Data = estLP, Fields = fields, Intercept = ifelse(use_intercept == 1L, TRUE, FALSE), call = the_call)
  class(obj) <- "LogisticRegressionModel"
  obj
}

# The API for a decision tree model
decisionTree <- function(formula,
                         df,
                         sqlCtx,
                         modType = c("classification", "regression"),
                         impurity = NULL,
                         nclasses = 2L,
                         maxDepth = 5L,
                         maxBins = 32L) {
  # Input error checking and field coercion to the correct type
  if (class(formula) != "formula") {
    stop("The provided formula is not a formula object.")
  }
  if (class(df) != "DataFrame") {
    stop("The data (df) must be a Spark DataFrame.")
  }
  modType = match.arg(modType)
  if (!is.null(impurity)) {
    if (modType == "classification") {
      if (!(impurity %in% c("gini", "entropy"))) {
        message(paste(impurity, "is not a valid classification impurity measure, and was replaced by 'gini'"))
        impurity <- "gini"
      }
    } else {
      if (impurity != "variance") {
        message(paste(impurity, "is not a valid regression impurity measure, and was replaced by 'variance'"))
        impurity <- "variance"
      }
    }
  } else {
    if (modType == "classification") {
      impurity <- "gini"
    } else {
      impurity <- "variance"
    }
  }
  nclasses <- as.integer(nclasses)
  if (nclasses < 2L) {
    stop("The number of classes must be two or higher.")
  }
  maxDepth <- as.integer(maxDepth)
  if (maxDepth < 1L) {
    stop("The maximum depth of a node must be strictly positive.")
  }
  maxBins <- as.integer(maxBins)
  if (maxBins < 2L) {
    stop("The maximum number of bins must be greater than one.")
  }
  # Parse the formula and prepare the data
  the_call <- match.call()
  pf <- SparkR:::parseFormula(formula)
  fields <- pf[[1]]
  estDF <- select(df, fields)
  estLP <- dfToLabeledPoints(estDF)
  SparkR:::callJMethod(estLP, "cache")
  if (modType == "classification") {
    the_model <- SparkR:::callJStatic("edu.berkeley.cs.amplab.sparkr.MLlibR",
                                      "trainClassificationTree",
                                      estLP,
                                      nclasses,
                                      impurity,
                                      maxDepth,
                                      maxBins)
  } else {
    the_model <- SparkR:::callJStatic("edu.berkeley.cs.amplab.sparkr.MLlibR",
                                      "trainRegressionTree",
                                      estLP,
                                      impurity,
                                      maxDepth,
                                      maxBins)
  }
  obj <- list(Model = the_model, Data = estLP, Fields = fields, Type = modType)
  if (modType == "classification") {
    obj$Classes <- nclasses
  }
  obj$call <- the_call
  class(obj) <- "DecisionTreeModel"
  obj
}


##
## Model Evaluation and Summary Functions
##

## I'M HERE WITH A DESIGN DELEMA. I WANT THIS OPEN TO BOTH ESTIMATION AND OTHER DATA
## AND WANT IT TO HANDLE MODEL OBJECTS BOTH WITH AND WITHOUT THE ABILITY TO SET
## THRESHOLDS. RIGHT NOW I TAKE THE jobj MODEL OBJECT AND THE LABELED POINTS, BUT
## THESE ARE WRAPPED UP INTO THE REMOVING OF THE THRESHOLD TO GET PROBABILITIES.
## I THINK THE ANSWER IS TO ALLOW NULL FOR THE THRESHOLD VALUE
# A function to create an RDD of label/score pair tuples
scoresLabels <- function(model, labeled_points, threshold = 0.5) {
  # The check below will expand in terms of model types
  #if (!any(class(model) %in% c("LogisticRegressionModel"))) {
  #  stop("The provided model is not of an appropriate type.")
  #}
  if (class(labeled_points) != "jobj") {
    stop("The provided labeled_points is not a jobj.")
  }
  if (getJClassName(model) %in% c("LogisticRegressionModel")) {
    sl <- SparkR:::callJStatic("edu.berkeley.cs.amplab.sparkr.MLlibR",
                               "ScoresLabels",
                               model,
                               labeled_points,
                               threshold)
  } else {
    sl <- SparkR:::callJStatic("edu.berkeley.cs.amplab.sparkr.MLlibR",
                               "ScoresLabels",
                               model,
                               labeled_points)
  }
  SparkR:::RDD(sl)
}

# A function to create an R table for a confusion matrix created from a Spark
# ScoresLabels object for a MLlib binary classification model
binaryConfusionMatrix <- function(sl) {
  if (class(sl) != "RDD") {
    stop("The provided argument is not a reference to a RDD.")
  }
  cml <- SparkR:::callJStatic("edu.berkeley.cs.amplab.sparkr.MLlibR",
                              "binaryConfusionMatrix",
                              sl@jrdd)
  cm <- matrix(unlist(cml), ncol = 2, nrow = 2, byrow = TRUE, dimnames = list(c("Actual 0", "Actual 1"), c("Predicted 0", "Predicted 1")))
  as.table(cm)
}

# A function to calculate residual and null devainces for a model
binaryClassificationDeviance <- function(slProb) {
  if (class(slProb) != "RDD") {
    stop("The provided argument is not a reference to a RDD.")
  }
  devs <- SparkR:::callJStatic("edu.berkeley.cs.amplab.sparkr.MLlibR",
                              "binaryClassificationDeviance",
                              slProb@jrdd)
  unlist(devs)
}

# A function to calculate the misclassification rate of a binary classification
# model
misClassRate <- function(score_label) {
  SparkR:::callJStatic("edu.berkeley.cs.amplab.sparkr.MLlibR",
                        "misClassRate",
                        score_label@jrdd)
}

# A function to extract the model coefficients from a MLlib regression model
getCoefs <- function(mod_obj) {
  coefs <- SparkR:::callJStatic("edu.berkeley.cs.amplab.sparkr.MLlibR",
                                "getCoefs",
                                mod_obj$Model)
  unlist(coefs)
}

# A function to create a BinaryClassificationMetrics object for a ScoresLabels
# object
BinaryClassificationMetrics <- function(sl) {
  if (class(sl) != "RDD") {
    stop("The provided argument is not a reference to a RDD.")
  }
  SparkR:::callJStatic("edu.berkeley.cs.amplab.sparkr.MLlibR",
                        "BCMetrics",
                        sl@jrdd)
}

# A function to get the area under a ROC curve for a model given a
# BinaryClassificationMetrics object
areaUnderROC <- function(bCMetrics) {
  SparkR:::callJMethod(bCMetrics, "areaUnderROC")
}

# A summary method for a MLlib linear regression model
summary.LinearRegressionModel <- function(mod_obj) {
  # The model call
  cat("Call:\n")
  print(mod_obj$call)
  # The coefficient estimate summary
  the_estimates <- getCoefs(mod_obj)
  the_coefficients <- mod_obj$Fields
  the_coefficients[1] <- "(Intercept)"
  coef_df <- data.frame(Estimate = format(the_estimates, digits = 3, nsmall = 2))
  row.names(coef_df) <- the_coefficients
  cat("\nCoefficients:\n")
  print(coef_df)
  # Model Fit statistics: TODO
}

# A summary method for a MLlib logistic regression model
summary.LogisticRegressionModel <- function(mod_obj) {
  # The model call
  cat("Call:\n")
  print(mod_obj$call)
  # The coefficient estimate summary
  the_estimates <- getCoefs(mod_obj)
  if (mod_obj$Intercept == TRUE) {
    the_intercept <- SparkR:::callJMethod(mod_obj$Model, "intercept")
    the_estimates <- c(the_intercept, the_estimates)
    the_coefficients <- mod_obj$Fields
    the_coefficients[1] <- "(Intercept)"
  } else {
    the_coefficients <- mod_obj$Fields[-1]
  }
  coef_df <- data.frame(Estimate = format(the_estimates, digits = 3, nsmall = 2))
  row.names(coef_df) <- the_coefficients
  cat("\nCoefficients:\n")
  print(coef_df)
  # Model Fit statistics: TODO
  sl1 <- scoresLabels(mod_obj$Model, mod_obj$Data, threshold = 0.0)
  deviances <- binaryClassificationDeviance(sl1)
  mcF <- 1 - (deviances[1]/deviances[2])
  aic <- 2*length(the_coefficients) + deviances[1]
  metrics <- BinaryClassificationMetrics(sl1)
  auROC <- areaUnderROC(metrics)
  deviances <- c(deviances, mcF, aic, auROC)
  dev_df <- data.frame(Value = format(deviances, digits = 2, nsmall = 3))
  row.names(dev_df) <- c("Residual Deviance", "Null Deviance", "McFadden R^2", "AIC", "Area Under ROC")
  cat("\nSummary Statistics:\n")
  print(dev_df)
  sl2 <- scoresLabels(mod_obj$Model, mod_obj$Data)
  cat("\nConfusion Matrix:\n")
  print(binaryConfusionMatrix(sl2))
}

# A summary method for a MLlib decision tree model
summary.DecisionTreeModel <- function(mod_obj) {
  # The model call
  cat("Call:\n")
  print(mod_obj$call)
  # The model summary
  the_summary <- SparkR:::callJMethod(mod_obj$Model, "toString")
  cat("\nTree Summary:\n")
  cat(paste(the_summary, "\n"))
  # Model Fit statistics
  sl <- scoresLabels(mod_obj$Model, mod_obj$Data)
  if (mod_obj$Type == "classification") {
    misclass <- misClassRate(sl)
    summary_vec <- misclass
    summary_names <- "Misclassification rate"
    if (mod_obj$Classes == 2) {
      metrics <- BinaryClassificationMetrics(sl)
      summary_vec <- c(summary_vec, areaUnderROC(metrics))
      summary_names <- c(summary_names, "Area under the ROC")
    }
    summary_df <- data.frame(Value = format(summary_vec, digits = 2, nsmall = 3))
    row.names(summary_df) <- summary_names
    cat("\nSummary Statistics:\n")
    print(summary_df)
    if (mod_obj$Classes == 2) {
      cat("\nConfusion Matrix:\n")
      print(binaryConfusionMatrix(sl))
    }
  }
}


##
## Model Prediction Methods
##

# The generic idScore method
idScore <- function(model, ...) {
  UseMethod("idScore", model)
}

idScore.LinearRegressionModel <- function(model, id, df, sqlCtx) {
  if (class(id) != "character") {
    stop("The identifier (id) field needs to be given a single item character vector.")
  }
  if (class(df) != "DataFrame") {
    stop("The data (df) must be a Spark DataFrame.")
  }
  fields <- as.list(c(id, model$Fields[-1]))
  scoreDF <- select(df, fields)
  scoreIP <- dfToIdPoints(scoreDF)
  SparkR:::callJMethod(scoreIP,"cache")
  scores <- SparkR:::callJStatic("edu.berkeley.cs.amplab.sparkr.MLlibR",
                                "IdScore",
                                model$Model,
                                scoreIP,
                                sqlCtx)
  dataFrame(scores)
}

# The idScore method for logistic regression
idScore.LogisticRegressionModel <- function(model, id, df, sqlCtx) {
  if (class(id) != "character") {
    stop("The identifier (id) field needs to be given a single item character vector.")
  }
  if (class(df) != "DataFrame") {
    stop("The data (df) must be a Spark DataFrame.")
  }
  # Create field order, making sure to place ID field first
  fields <- as.list(c(id, model$Fields[-1]))
  scoreDF <- select(df, fields)
  scoreIP <- dfToIdPoints(scoreDF)
  SparkR:::callJMethod(scoreIP, "cache")
  scores <- SparkR:::callJStatic("edu.berkeley.cs.amplab.sparkr.MLlibR",
                                "IdScore",
                                model$Model,
                                scoreIP,
                                sqlCtx)
  dataFrame(scores)
}

# The idScore method for decision trees
idScore.DecisionTreeModel <- function(model, id, df, sqlCtx) {
  if (class(id) != "character") {
    stop("The identifier (id) field needs to be given a single item character vector.")
  }
  if (class(df) != "DataFrame") {
    stop("The data (df) must be a Spark DataFrame.")
  }
  fields <- as.list(c(id, model$Fields[-1]))
  scoreDF <- select(df, fields)
  scoreIP <- dfToIdPoints(scoreDF)
  SparkR:::callJMethod(scoreIP,"cache")
  scores <- SparkR:::callJStatic("edu.berkeley.cs.amplab.sparkr.MLlibR",
                                "IdScore",
                                model$Model,

# Get all or a subset of the scored data into an R data frame. Needed until
# there is a more general way to create Spark DataFrames from R calls.
getScores <- function(id_scores, number = -1L) {
  the_ids <- SparkR:::callJStatic("edu.berkeley.cs.amplab.sparkr.MLlibR",
                                  "getIDs",
                                  id_scores@jrdd,
                                  as.integer(number))

  the_scores <- SparkR:::callJStatic("edu.berkeley.cs.amplab.sparkr.MLlibR",
                                    "getScores",
                                    id_scores@jrdd,
                                    as.integer(number))

  data.frame(ID = unlist(the_ids), Score = unlist(the_scores))
                                scoreIP,
                                sqlCtx)
  dataFrame(scores)
}
