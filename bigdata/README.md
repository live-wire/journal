# Supercomputing for Big Data :laptop:

---
## Lab 1 
[manual](https://github.com/Tclv/SBD-2018/blob/master/doc/manual.md)

Useful Spark commands and notes:
- Narrow Dependency: (operations like map), wide dependency : (operations like reduce)
- Lazy evaluation: Not computed till requested. (Till then only the DAG is calculated)
- Create RDD using `sc.parallelize` to enter Spark's programming paradigm
- Load CSVs using `sc.textFile('<name>')`
- Filtered representations of the RDD using command like:
```
val filterRDD = raw_data.map(_.split(",")).filter(x => x(1) == "3/10/14:1:01")
filterRDD.foreach(a => {println(a.mkString(" "))})
```
- Use Dataframes, and Datasets(type checked dataframes) for defining schema of data being read and type checking etc.
```
case class SensorData (
    sensorName: String,
    timestamp: Timestamp,
    numA: Double,
    numB: Double,
    numC: Long,
    numD: Double,
    numE: Long,
    numF: Double
)

val ds = spark.read.schema(schema)
              .option("timestampFormat", "MM/dd/yy:hh:mm")
              .csv("./lab1/sensordata.csv")
              .as[SensorData]
```
-  


---
## Lecture 1
- 3V model
	- Volume
	- Velocity
	- Variety
- 80% of the data generated is Unstructured
- Big data pipeline phases
	- Sense/Acquire
	- Store/Ingest
	- Retrieve/filter
	- Analyze
	- Visualize
- Spark uses in RDD's (in memory) so are significantly faster than traditional map reduce (at-least for batch processing)














































## LAb 0
- Hadoop is not a replacement for Relational Database
- It complements online transaction processing and online analytical processing
- Used for structured and unstructured data (large quantities)
- Not good for:
	- Hadoop is not good to process transactions due to its lack random access.
	- It is not good when the work cannot be parallelized or when there are dependencies within the data, that is, record one must be processed before record two.
	- It is not good for low latency data access. 
	- Not good for processing lots of small files

- Ambari GUI for managing hadoop cluster.


