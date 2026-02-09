import sys
import boto3
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import MapType, StringType, DoubleType

# Initialize Glue
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# --- CONFIGURATION ---
S3_LOGS = "PATH TO THE EXTRACTED DRIVER LOGS DATA" # Hard coded for now. It will be generated dynamically using datetime function
S3_FUEL = "PATH TO THE EXTRACTED INVOICE DATA" # Hard coded for now. It will be generated dynamically using datetime function
OUTPUT_PATH = "OUTPUT PATH FOR ANOMALIES RECORDS"# Hard coded for now. It will be generated dynamically using datetime function

KNOWN_PROVINCES = ['AB', 'ON', 'MB', 'SK', 'BC', 'QC', 'NB', 'NS', 'PE', 'NL']

# 1. Load Data
driver_logs_df = spark.read.parquet(S3_LOGS)
fuel_invoices_df = spark.read.parquet(S3_FUEL)

# 2. Data Transformation (Jurisdiction Breakdown)
enriched_logs = driver_logs_df.withColumn(
    "jurisdiction_map", 
    F.from_json(F.to_json(F.col("jurisdiction_breakdown")), MapType(StringType(), DoubleType()))
)

exploded_logs = enriched_logs.select(
    "*",
    F.explode("jurisdiction_map").alias("Jurisdiction", "split_percentage")
)

# 3. Anomaly Checks (Logs)
exploded_logs = exploded_logs.withColumn(
    "anomaly_unknown_jurisdiction",
    F.when(~F.col("Jurisdiction").isin(KNOWN_PROVINCES), True).otherwise(False)
)

windowSpecOdo = Window.partitionBy("vehicle_id").orderBy("raw_Date", "start_odometer")
exploded_logs = exploded_logs.withColumn("prev_end_odo", F.lag("end_odometer").over(windowSpecOdo))
exploded_logs = exploded_logs.withColumn(
    "anomaly_gap_miles",
    F.when((F.col("prev_end_odo").isNotNull()) & (F.abs(F.col("start_odometer") - F.col("prev_end_odo")) > 1.0), True).otherwise(False)
)

# 4. Aggregation and Join
exploded_logs = exploded_logs.withColumn("jurisdiction_distance", F.col("Distance_km") * (F.col("split_percentage") / 100.0))

daily_logs = exploded_logs.groupBy("vehicle_id", "raw_Date", "Jurisdiction").agg(
    F.sum("jurisdiction_distance").alias("total_km"),
    F.max(F.col("anomaly_unknown_jurisdiction").cast("int")).alias("has_unknown_jur"),
    F.max(F.col("anomaly_gap_miles").cast("int")).alias("has_odo_gap")
).withColumnRenamed("raw_Date", "log_date")

daily_fuel = fuel_invoices_df.groupBy("vehicle_id", "PurchaseDate", "Jurisdiction").agg(
    F.sum(F.col("Quantity").cast("float")).alias("total_liters")
).withColumnRenamed("PurchaseDate", "fuel_date")

combined_df = daily_logs.join(daily_fuel, on=['vehicle_id', 'Jurisdiction'], how="outer")

# 5. Metrics & Join-based Anomalies
combined_df = combined_df.withColumn("miles", F.col("total_km") * 0.621371) \
                         .withColumn("gallons", F.col("total_liters") * 0.264172)

combined_df = combined_df.withColumn("mpg", F.col("miles") / F.col("gallons")) \
    .withColumn("anomaly_unreasonable_mpg", F.when((F.col("mpg") < 5) | (F.col("mpg") > 10), True).otherwise(False)) \
    .withColumn("anomaly_missing_fuel", F.when((F.col("miles") > 0) & (F.col("total_liters").isNull()), True).otherwise(False)) \
    .withColumn("anomaly_logical_mismatch", F.when((F.col("total_liters") > 0) & (F.col("miles").isNull() | (F.col("miles") == 0)), True).otherwise(False)) \
    .withColumn("anomaly_unknown_jurisdiction", F.col("has_unknown_jur") == 1) \
    .withColumn("anomaly_gap_miles", F.col("has_odo_gap") == 1)

# --- Vehicle Filtering ---

# List of all anomaly columns
anomaly_cols = ["anomaly_unreasonable_mpg", "anomaly_missing_fuel", "anomaly_logical_mismatch", "anomaly_unknown_jurisdiction", "anomaly_gap_miles"]

# 6. Check if ANY row for a vehicle has an anomaly
# We create a flag that is True if ANY anomaly exists in any row for that vehicle_id
vehicle_window = Window.partitionBy("vehicle_id")
combined_df = combined_df.withColumn("vehicle_has_any_anomaly", 
    F.max(
        F.when(
            F.col("anomaly_unreasonable_mpg") | 
            F.col("anomaly_missing_fuel") | 
            F.col("anomaly_logical_mismatch") | 
            F.col("anomaly_unknown_jurisdiction") | 
            F.col("anomaly_gap_miles"), 
            1
        ).otherwise(0)
    ).over(vehicle_window) == 1
)

# 7. Split Data
# Anomalous vehicles stay in S3
anomalous_data = combined_df.filter(F.col("vehicle_has_any_anomaly") == True)
anomalous_data.write.mode("overwrite").parquet(OUTPUT_PATH)

# Clean vehicles go to RDS
clean_data = combined_df.filter(F.col("vehicle_has_any_anomaly") == False).drop("vehicle_has_any_anomaly")

# 8. Write Clean Data to Relational DB

job.commit()
