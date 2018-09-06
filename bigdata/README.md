# Supercomputing for Big Data :laptop:

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


