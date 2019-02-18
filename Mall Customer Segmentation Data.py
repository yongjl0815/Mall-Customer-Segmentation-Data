import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

file_name = "/Mall_Customers.csv"


# Load Data
sc= SparkContext()
sqlContext = SQLContext(sc)

df = sqlContext.read.format("csv").options(header='true', inferschema='true').load("../input/Mall_Customers.csv")

# check if loading was successful
df.take(1)

# check structure
df.cache()
df.printSchema()


# descriptive analysis
df.describe().toPandas().transpose()


# Change column names with space 
df = df.withColumnRenamed("Annual Income (k$)", "AI")
df = df.withColumnRenamed("Spending Score (1-100)", "SS")
df.columns


# Check for null value
df.where(df.CustomerID.isNull()).count()
df.where(df.Gender.isNull()).count()
df.where(df.Age.isNull()).count()
df.where(df.AI.isNull()).count()
df.where(df.SS.isNull()).count()


# Change Male and Female to integer values 
from pyspark.sql.functions import *
newDf = df.withColumn('Gender', regexp_replace('Gender', 'Male', '1'))
upDf = newDf.withColumn('Gender', regexp_replace('Gender', 'Female', '2'))

# Change Gender type to string
from pyspark.sql.types import IntegerType
upDf = upDf.withColumn("Gender", upDf["Gender"].cast(IntegerType()))
upDf.cache()
upDf.printSchema()


from pyspark.ml.feature import VectorAssembler

# set up data for ML
vectorAssembler = VectorAssembler(inputCols = ['Gender', 'Age', 'AI'], outputCol = 'features')
ml_df = vectorAssembler.transform(upDf)
ml_df = ml_df.select(['features', 'SS'])
ml_df.show(3)

# split data, train and test
splits = ml_df.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]


from pyspark.ml.regression import LinearRegression
# train model with linear regression
lr = LinearRegression(featuresCol = 'features', labelCol='SS', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

# test the model with test data
lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction","SS","features").show(5)

from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="SS",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))
test_result = lr_model.evaluate(test_df)
print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)


# Decision Tree Regressor
from pyspark.ml.regression import DecisionTreeRegressor

dt = DecisionTreeRegressor(featuresCol ='features', labelCol = 'SS')
dt_model = dt.fit(train_df)
dt_predictions = dt_model.transform(test_df)
dt_evaluator = RegressionEvaluator(
    labelCol="SS", predictionCol="prediction", metricName="rmse")
rmse = dt_evaluator.evaluate(dt_predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


# Gradient-boosted tree regression
from pyspark.ml.regression import GBTRegressor

gbt = GBTRegressor(featuresCol = 'features', labelCol = 'SS', maxIter=10)
gbt_model = gbt.fit(train_df)
gbt_predictions = gbt_model.transform(test_df)
gbt_predictions.select('prediction', 'SS', 'features').show(5)

gbt_evaluator = RegressionEvaluator(labelCol="SS", predictionCol="prediction", metricName="rmse")
rmse = gbt_evaluator.evaluate(gbt_predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)