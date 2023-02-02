#imports the needed libraries
import pyspark
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,isnan, when, count
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import MultilayerPerceptronClassifier


#loads the .csv file into a pyspark dataframe
spark = SparkSession.builder.getOrCreate()
df = spark.read.csv("nuclear_plants_small_dataset.csv" ,inferSchema=True,header=True)

#displays the pyspark dataframe (top 20 rows)
df.show()

#prints the dataframes schema
df.printSchema()

#searches the data for any missing values in pyspark
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()

#converts the pyspark dataframe into a pandas dataframe for stats summary
pandasDF = df.toPandas()

#searches the data for missing values in pandas
null = pandasDF.isnull().sum(axis=0)
print(null)

#groups dataframe by 'normal' and 'abnormal' to show summary stats
min = pandasDF.groupby("Status").min()
print(min)
max = pandasDF.groupby("Status").max()
print(max)
mean = pandasDF.groupby("Status").mean()
print(mean)
median = pandasDF.groupby("Status").median()
print(median)
mode = pandasDF.groupby("Status").agg(pd.Series.mode)
print(mode)
var = pandasDF.groupby("Status").var()
print(var)

#creates box plots of the data grouped by 'Status'
pandasDF.boxplot(column=["Power_range_sensor_1", "Power_range_sensor_2", "Power_range_sensor_3 ", "Power_range_sensor_4", "Pressure _sensor_1", "Pressure _sensor_2", "Pressure _sensor_3", "Pressure _sensor_4", "Vibration_sensor_1", "Vibration_sensor_2", "Vibration_sensor_3", "Vibration_sensor_4"],by="Status")
plt.show()

#displays the correlation matrix of the features
matrix = pandasDF.corr()
print(matrix)

#shuffle and split data 70/30
(trainingData, testData) = df.randomSplit([0.7, 0.3])

#counts number of examples in each group
trainingData.groupBy("Status").count().show()
testData.groupBy("Status").count().show()

#string indexer for status column
statusIndexer = StringIndexer(inputCol="Status", outputCol="indexedStatus").fit(df)

#vector assembler for vector indexer on mulitple columns
assembler = VectorAssembler(inputCols=["Power_range_sensor_1", "Power_range_sensor_2", "Power_range_sensor_3 ", "Power_range_sensor_4", "Pressure _sensor_1", "Pressure _sensor_2", "Pressure _sensor_3", "Pressure _sensor_4", "Vibration_sensor_1", "Vibration_sensor_2", "Vibration_sensor_3", "Vibration_sensor_4"], outputCol="features")

#trains a decision tree
dt = DecisionTreeClassifier(labelCol="indexedStatus", featuresCol="features")

#creates pipline
dt_pipeline = Pipeline(stages=[statusIndexer, assembler, dt])

#trains decision tree model
dt_model = dt_pipeline.fit(trainingData)

#produces predictions
dt_predictions = dt_model.transform(testData)

dt_evaluator = MulticlassClassificationEvaluator(labelCol="indexedStatus", predictionCol="prediction", metricName="accuracy")
dt_accuracy = dt_evaluator.evaluate(dt_predictions)
print("dt test error: %g" % (1 - dt_accuracy))

#support vector machine
svm = LinearSVC(labelCol="indexedStatus", featuresCol="features")
svm_pipeline = Pipeline(stages=[statusIndexer, assembler, svm])
svm_model = svm_pipeline.fit(trainingData)
svm_predictions = svm_model.transform(testData)
svm_evaluator = MulticlassClassificationEvaluator(labelCol="indexedStatus", predictionCol="prediction", metricName="accuracy")
svm_accuracy = svm_evaluator.evaluate(svm_predictions)
print("svm test error: %g" % (1 - svm_accuracy))

#artificial neural network
layers = [12,5,5,2]
mlp = MultilayerPerceptronClassifier(labelCol="indexedStatus", featuresCol="features", layers = layers, blockSize=128)
mlp_pipeline = Pipeline(stages=[statusIndexer, assembler, mlp])
mlp_model = mlp_pipeline.fit(trainingData)
mlp_predictions = mlp_model.transform(testData)
mlp_evaluator = MulticlassClassificationEvaluator(labelCol="indexedStatus", predictionCol="prediction", metricName="accuracy")
mlp_accuracy = mlp_evaluator.evaluate(mlp_predictions)
print("mlp test error: %g" % (1 - mlp_accuracy))

#loads big dataset into dataframe
big_df = spark.read.csv("nuclear_plants_big_dataset.csv" ,inferSchema=True,header=True)

#MapReduce for min max values
map = big_df.rdd.map(lambda x : (x,1))
reduce = map.reduceByKey(lambda a,b: a+b)
max = reduce.max()
min = reduce.min()
print("Max value: " + str(max))
print("Min value: " + str(min))