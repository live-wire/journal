## Supercomputing for Big Data

- Hadoop is not a replacement for Relational Database
- It complements online transaction processing and online analytical processing
- Used for structured and unstructured data (large quantities)
- Not good for:
	- Hadoop is not good to process transactions due to its lack random access.
	- It is not good when the work cannot be parallelized or when there are dependencies within the data, that is, record one must be processed before record two.
	- It is not good for low latency data access. 
	- Not good for processing lots of small files

- Ambari GUI for managing hadoop cluster.