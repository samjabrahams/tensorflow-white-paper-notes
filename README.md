# TensorFlow White Paper Notes

_[White Paper available at this link](http://download.tensorflow.org/paper/whitepaper2015.pdf)_

## Abstract

* **TensorFlow** is both an interface for expressing ML algorithms and an implentation to execute them
* Code can be transported across various machine architectures with little to no changes
* Has been used at Google for all manner of machine learning tasks
* Reference implementation and API released under Apache 2.0 license

## 1 Introduction

* Google Brain started in 2011, and **DistBelief** was its first-generation of scalable distributed ML system
* DistBelief was used for a large number of research and commercial tasks.
* TensorFlow, Google's second-generation ML system, was designed from lessons learned in the process of engineering and using DistBelief
* The TensorFlow API is used to describe a dataflow-like model, and the implementation then maps those models onto the underlying machine hardware.
* This allows users to have a single system that runs on a broad spectrum of machines, reducing overhead caused from rewriting code for different hardware.
* Focus of development was to maintain flexibility for research purposes while  attaining enough performance to be used in production.
* Can express various types of parallelism by replicating the dataflow model across multiple machines and running them in parallel. 
	* Some functions within TensorFlow allow for less consistency in parallelism if desired
* TensorFlow is more flexible, faster, and supports more ML models than DistBelief

## 2 Programming Model and Basic Concepts

* TensorFlow computations are represented by _directed graphs_, which are composed by _nodes_
* Some nodes are able to maintain and update a persistent state and/or have some sort of branching and looping structures
	* This branching/looping is modeled similarly to [MSR's Naid](http://research.microsoft.com:8082/pubs/201100/naiad_sosp2013.pdf)
* Graphs are constructed using supported languages (C++/Python as of writing)
* A Node has zero or more inputs/outputs, and it represents an _operation_
* Values of 'normal' edges (the connection between one node's output to another node's input) are _tensors_, n-dimensional arrays.
	* The type of each element in the tensor is inferred while the graph is being constructed, prior to running training
* There are 'special' edges, called _control dependencies_: no model data is transferred on these edges, rather they indicate that the source node must finish execution before the destination node begins execution
	* Can be thought of as a baton in a relay race. Attaching a control dependency means that the next node can't begin running until the previous node 'hands off' the baton.
	* Used by client to enforce happens-before relations and in reference implementation to manage memory usage

### Operations and Kernels

* Operations have names and represent an abstract computation, such as "matrix multiply" or "add"
* Opperations can require _attributes_. Attributes must be explicitly provided or be possible to infer prior to running the graph
	* A common use of attributes is to declare which data type the operation is being performed with (i.e. float tensors vs. int32 tensors)
* A _kernel_ is an implementation of an operation designed for specific types of devices, such as CPU or GPU
* The TensorFlow library includes several built-in operations/kernels. The table below lists some of them:

Category | Examples
---|---
Element-wise mathematical operations | Add, Sub, Mul, Div, Exp, Log, Greater, Less, Equal
Array operations | Concat, Slice, Split, Constant, Rank, Shape, Shuffle
Matrix operations | MatMul, MatrixInverse, MatrixDeterminant
Stateful operations | Variable, Assign, AssignAdd
Neural-net building blocks | SoftMax, Sigmoid, ReLU, Convolution2D, MaxPool
Checkpointing operations | Save, Restore
Queue and synchronization operations | Enqueue, Dequeue, MutexAcquire, MutexRelease
Control flow operations | Merge, Switch, Enter, Leave, NextIteration

### Sessions

* Clients interact with TensorFlow by creating a _Session_, which supports two main functions: _Extend_ and _Run_
	* The Extend method adds additional nodes and edges to the existing dataflow model
	* Run takes as argument a set of named nodes to be computed as well as an optional set of tensors to be used in place of certain node outputs. It then uses the graph to figure all nodes required to compute the requested outputs, and performs them in a order that respects their dependencies.
* Most TensorFlow programs setup a graph within a Session once, and then run the full graph or subsets of the graph multiple times.

### Variables

* A _Variable_ is a handle to a persistent and mutable tensor which survives each execution of a graph
* For ML tasks, learned parameters are usually held in TensorFlow Variables

## 3 Implementation

* There are three primary components in a TensorFlow system: the _client_, the _master_, and _worker processes_
	* The client uses a Session interface to communicate with the master
	* The master schedules and coordinates worker processes and relays results back to the client
	* Worker processes are responsible for maintaining access to devices such as CPU/GPU cores and execute graph nodes on their respective devices
* There are both local and distributed implementations of TensorFlow, but only the local version has been open-sourced as of November 2015

### Devices

* Each device has both a device type and a name
	* Names are composed of the device's type, its index in a worker process, and (when used in a distributed setting) an identification of the job and task of the worker process
	* Example device names:  
	Local: `/job:localhost/device:cpu:0`  
	Distributed: `/job:worker/task:17/device:gpu:3`
* A device object manages its device's memory and executes kernels as requested

### Tensors

* Typed, multi-dimensional array
* Memory management of tensors is handled automatically
* Available types (from the [TensorFlow website](https://www.tensorflow.org/versions/master/resources/dims_types.html#data-types)):  

Data type | Python type | Description
--- | --- | ---
`DT_FLOAT` | `tf.float32` | 32 bits floating point
`DT_DOUBLE` | `tf.float64` | 64 bits floating point
`DT_INT64` | `tf.int64` | 64 bits signed integer
`DT_INT32` | `tf.int32` | 32 bits signed integer
`DT_INT16` | `tf.int16` | 16 bits signed integer
`DT_INT8` | `tf.int8` | 8 bits signed integer
`DT_UINT8` | `tf.uint8` | 8 bits unsigned integer
`DT_STRING` | `tf.string` | Variable length byte arrays.  Each element of a Tensor is a byte array
`DT_BOOL` | `tf.bool` | Boolean
`DT_COMPLEX64` | `tf.complex64` | Complex number made of two 32 bits floating points: real and imaginary parts
`DT_QINT32` | `tf.qint32` | 32 bits signed integer used in quantized Ops
`DT_QINT8` | `tf.qint8` | 8 bits signed integer used in quantized Ops
`DT_QUINT8` | `tf.quint8` | 8 bits unsigned integer used in quantized Ops

## 3.1 Single-Device Execution

**NOTE:** To reiterate- in this context, "single device" means using a single CPU core or single GPU, _not_ a single machine. Similarly, "multi-device" does _not_ refer to multiple machines, but to multiple CPU cores and/or GPUs. See "3.3 Distributed Execution" for multiple machine discussion.

* Execution of single-worker process, single-device job:
	1. All nodes required to compute the desired output node(s) are determined
	2. Each node is given a count of dependencies that need to be completed before it can begin execution
	3. When a node's dependency count is zero, it is added to a ready queue
	4. The ready queue delegates node kernel execution to device objects
	5. When a node completes execution, the counts of all dependant nodes are decremented
	6. Repeat steps 3-5 until the desired output is computed

## 3.2 Multi-Device Execution

* There are two main challenges introduced when using multiple devices:
	* Deciding which device should process each node
	* Managing communication between devices as necessary after assigning nodes

### Node Placement

* One of the main responsibilities of the TensorFlow implementation is to map computation onto available devices
* The following is a simplified version of this mapping algorithm:
	1. A cost model is input into the algorithm
		* The cost model contains estimates of of the input/output tensors (in bytes) and estimated computation time for each node in the graph
	2. Using the cost model, the algorithm simulates an execution of the graph to make node-placement decisions as described below:
		1. Starting with the source nodes, a set of feasible devices is considered for each node ready to be executed
			* A "feasible" device is one that has a kernel implementation for the given operation
			* A node is ready for execution once its dependencies have finished running
		2. If a node has multiple feasible devices, the computation time of the node is examined with respect to placing the node on each possible device
			* This examination takes into account the execution time of the operation, given the device type, as well as the costs of possibly introducing communication in order to recieve inputs from other devices.
		3. The device that would finish the operation the soonest is selected as the node's device.
		4. Repeat steps 1-3 for each node in the graph execution until all nodes have been allocated to devices
	3. After the simulation, the real execution runs using the node-placement decisions made during the simulation
* Section 4.3 will describe some extensions to help guide the placement algorithm
* The placement algorithm's development is an ongoing process as of writing

### Cross-Device Communication

* After the nodes have been placed onto their respective devices, the execution graph is split into subgraphs- one per device
* Any edge between nodes on different devices is replaced by two new edges:
	* The outputing node will have an edge between it and a new _Send_ node, placed within the subgraph of its device
	* The recieving node will have an edge between it and a new _Receive_ node, placed within the subgraph of its device
* The Send and Receive nodes coordinate data transfer across devices, isolating cross-device communication to the implementation of the Send and Receive nodes
* All users of a particular tensor on a particular device use a single Receive node, as opposed to having one Receive node per user per device. This minimizes data transmission between devices as well as memory allocated on the receiving device
* This method of communication also allows individual node scheduling to be handled by the worker processes as opposed to the master
	* The Send and Receive nodes provide synchronization between worker processes and devices, which enables the master to only issue a single Run request per graph execution  per worker process
	* This improves scalability and fine-grain control over node execution

## 3.3 Distributed Execution

### Fault Tolerance

## 4 Extensions

## 4.1 Gradient Computation

## 4.2 Partial Execution

## 4.3 Device Constraints

## 4.4 Control Flow

## 4.5 Input Operations

## 4.6 Queues

## 4.7 Containers

## Optimizations

## 5.1 Common Subexpression Elimination

## 5.2 Controlling Data Communication and Memory Usage

## 5.3 Asynchronous Kernels

## 5.4 Optimized Libraries for Kernel Implementations

## 5.5 Lossy Compression

## 6 Status and Experience

## 7 Common Programming Idioms

### Data Parallel Training

### Model Parallel Training

### Concurrent Steps for Model Computation Pipelining

## 8 Performance

## 9 Tools

## 9.1 TensorBoard: Visualization of Graph Structures and Summary Statistics

### Visualization of Computation Graphs

### Visualization of Summary Data

## 9.2 Performance tracing

## 10 Future Work

## 11 Related Work

## 12 Conclusions