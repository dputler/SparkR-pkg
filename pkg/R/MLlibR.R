##
## Helper functions
##

# Parse an R formula so it can be incorporated into a SparkSQL query to get the
# target and relevant predictors and to determine if the model should include an
# intercept
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
  list(vars, intercept)
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


##
## MLlib model API functions
##

# The API for a logistic regression model estimated using the limited memory
# version of the BFGS optimization algorithm
logisticRegressionWithLBFGS <- function(formula,
                                        df,
                                        sqlCtx,
                                        iter = 100L,
                                        reg_param = 0.0,
                                        corrections = 10L,
                                        tol = 1e-4) {
  # Input error checking
  if (class(formula) != "formula") {
    stop("The provided formula is not a formula object.")
  }
  if (class(df) != "character") {
    stop("The Spark DataFrame name (df) needs to be a character string.")
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
  # Parse the formula and prepare the data
  the_call <- match.call()
  pf <- parseFormula(formula)
  fields <- pf[[1]]
  q_string <- paste("SELECT", paste(fields, collapse = ", "), "FROM", df)
  estDF <- sql(sqlCtx, q_string)
  registerTempTable(estDF, "estDF")
  estLP <- dfToLabeledPoints(estDF)
  SparkR:::callJMethod(estLP, "cache")
  use_intercept <- pf[[2]]
  the_model <- SparkR:::callJStatic("edu.berkeley.cs.amplab.sparkr.MLlibR",
                                    "trainLogisticRegressionModelWithLBFGS",
                                    estLP,
                                    iter,
                                    reg_param,
                                    use_intercept,
                                    corrections,
                                    tol)
  obj <- list(Model = the_model, Data = estLP, Fields = fields, Intercept = ifelse(use_intercept == 1L, TRUE, FALSE), call = the_call)
  class(obj) <- "LogisticRegressionModel"
  obj
}


##
## Model Evaluation and Summary Functions
##

# A function to create an RDD of label/score pair tuples
scoresLabels <- function(model, labeled_points, threshold = 0.5) {
  # The check below will expand in terms of model types
  #if (!any(class(model) %in% c("LogisticRegressionModel"))) {
  #  stop("The provided model is not of an appropriate type.")
  #}
  if (class(labeled_points) != "jobj") {
    stop("The provided labeled_points is not a jobj.")
  }
  sl <- SparkR:::callJStatic("edu.berkeley.cs.amplab.sparkr.MLlibR",
                              "ScoresLabels",
                              model,
                              labeled_points,
                              threshold)
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

binaryClassificationDeviance <- function(slProb) {
  if (class(slProb) != "RDD") {
    stop("The provided argument is not a reference to a RDD.")
  }
  devs <- SparkR:::callJStatic("edu.berkeley.cs.amplab.sparkr.MLlibR",
                              "binaryClassificationDeviance",
                              slProb@jrdd)
  unlist(devs)
}

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
  # Model Fit statistics
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


##
## Model Prediction Methods
##

idScore <- function(model, ...) {
  UseMethod("idScore", model)
}

idScore.LogisticRegressionModel <- function(model, id, df, sqlCtx) {
  if (class(id) != "character") {
    stop("The identifier (id) field needs to be given a single item character vector.")
  }
  if (class(df) != "character") {
    stop("The Spark DataFrame name (df) needs to be a character string.")
  }
  fields <- model$Fields[-1]
  q_string <- paste("SELECT", id, ", ", paste(fields, collapse = ", "), "FROM", df)
  scoreDF <- sql(sqlCtx, q_string)
  registerTempTable(scoreDF, "scoreDF")
  scoreIP <- dfToIdPoints(scoreDF)
  SparkR:::callJMethod(scoreIP, "cache")
  scores <- SparkR:::callJStatic("edu.berkeley.cs.amplab.sparkr.MLlibR",
                                "IdScore",
                                model$Model,
                                scoreIP)
  SparkR:::RDD(scores)
}

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
}
