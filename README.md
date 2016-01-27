# TensorFlow White Paper Notes

## Features

* Notes broken down section by section, as well as subsection by subsection
* Relevant links to documentation, resources, and references throughout
* SVG versions of figures/graphs
* So many bullet points!

### To-do list

* Create and utilize anchor tags throughout notes for self-navigating

_[White Paper available at this link](http://download.tensorflow.org/paper/whitepaper2015.pdf)_

- - -

# TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems

## Abstract

* [**TensorFlow**](https://www.tensorflow.org/) is both an interface for expressing machine learning algorithms and an implentation to execute them
* Code can be transported across various machine architectures with little to no changes to the code
* Has been used at Google for all manner of machine learning tasks
* [Reference implementation and API released under Apache 2.0 license](https://github.com/tensorflow/tensorflow)

- - -

## 1 Introduction

* Google Brain started in 2011, and [**DistBelief**](http://static.googleusercontent.com/media/research.google.com/en//archive/large_deep_networks_nips2012.pdf) was its first-generation scalable, distributed machine learning system
* DistBelief was used for a large number of research and commercial tasks
* TensorFlow, Google's second-generation machine learning system, was designed from lessons learned in the process of engineering and using DistBelief
* The TensorFlow API is used to describe a dataflow-like model, and the implementation then maps those models onto the underlying machine hardware
* This allows users to have a single system that runs on a broad spectrum of machines, reducing overhead caused from rewriting code for different hardware
* Focus of development was to maintain flexibility for research purposes while  attaining enough performance to be used in production
* Can express various types of parallelism by replicating the dataflow model across multiple machines and running them in parallel
	* Some functions within TensorFlow allow for less consistency in parallelism if desired
		* Larger, multiple machine uses of TensorFlow can take advantage of less-strict synchronization requirements
* TensorFlow is more flexible, faster, and supports more machine learning models than DistBelief

- - -

## 2 Programming Model and Basic Concepts

* TensorFlow computations are represented by [_directed graphs_](https://www.tensorflow.org/versions/master/api_docs/python/framework.html#Graph), which are composed of _nodes_
* Some nodes are able to maintain and update a persistent state and/or have some sort of branching and looping structures
	* This branching/looping is modeled similarly to [MSR's Naiad](http://research.microsoft.com:8082/pubs/201100/naiad_sosp2013.pdf)
* Graphs are constructed using supported front-end languages (C++/Python as of writing)
* A Node has zero or more inputs/outputs, and it represents an [_operation_](https://www.tensorflow.org/versions/master/api_docs/python/framework.html#Operation)
* Values of 'normal' edges (the connection between one node's output to another node's input) are [_tensors_](https://www.tensorflow.org/versions/master/api_docs/python/framework.html#Tensor), n-dimensional arrays.
	* The type of each element in the tensor is inferred while the graph is being constructed, prior to execution
* There are 'special' edges, called [_control dependencies_](https://www.tensorflow.org/versions/master/api_docs/python/framework.html#Graph.control_dependencies): no model data is transferred on these edges, rather they indicate that the source node must finish execution before the destination node begins execution
	* Can be thought of as a baton in a relay race. Attaching a control dependency means that the next node can't begin running until the previous node 'hands off' the baton.
	* Used by client to enforce happens-before relations
	* Used in reference implementation to manage memory usage



### Operations and Kernels

* Operations have names and represent an abstract computation, such as ["matrix multiply"](https://www.tensorflow.org/versions/master/api_docs/python/math_ops.html#matmul) or ["add"](https://www.tensorflow.org/versions/master/api_docs/python/math_ops.html#add)
* Operations can optionally require _attributes_. Attributes must be explicitly provided or be possible to infer prior to running the graph
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

_Check out [this directory in the TensorFlow repository](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/kernels) for kernel implementations_

### Sessions

* Clients interact with TensorFlow by creating a [_Session_](https://www.tensorflow.org/versions/master/api_docs/python/client.html#Session), which supports two main functions: _Extend_ and [_Run_](https://www.tensorflow.org/versions/master/api_docs/python/client.html#Session.run)
	* The Extend method adds additional nodes and edges to the existing dataflow model
		* _Note: Extend is called automatically by TensorFlow, not directly by the client_
	* Run takes as argument a set of named nodes to be computed as well as an optional set of tensors to be used in place of certain node outputs. It then uses the graph to figure all nodes required to compute the requested outputs, and performs them in a order that respects their dependencies.
* Most TensorFlow programs setup a graph within a Session once, and then run the full graph or subsets of the graph multiple times.

### Variables

* A [_Variable_](https://www.tensorflow.org/versions/master/api_docs/python/state_ops.html#Variable) is a handle to a persistent and mutable tensor which survives each execution of a graph
* For ML tasks, learned parameters are usually held in TensorFlow Variables

_See the official [How-To](https://www.tensorflow.org/versions/master/how_tos/variables/index.html) to learn more about TensorFlow Variables_

- - -

## 3 Implementation

* There are three primary components in a TensorFlow system: the _client_, the _master_, and _worker processes_
	* The client uses a Session interface to communicate with the master
	* The master schedules and coordinates worker processes and relays results back to the client
	* Worker processes are responsible for maintaining access to devices such as CPU/GPU cores and execute graph nodes on their respective devices
* There are both local and distributed implementations of TensorFlow, but only the local version has been open-sourced as of writing

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
* Available types (from the [TensorFlow documentation](https://www.tensorflow.org/versions/master/resources/dims_types.html#data-types)):  

Data type | Python type | Description
--- | --- | ---
`DT_FLOAT` | `tf.float32` | 32 bits floating point
`DT_DOUBLE` | `tf.float64` | 64 bits floating point
`DT_INT8` | `tf.int8` | 8 bits signed integer
`DT_INT16` | `tf.int16` | 16 bits signed integer
`DT_INT32` | `tf.int32` | 32 bits signed integer
`DT_INT64` | `tf.int64` | 64 bits signed integer
`DT_UINT8` | `tf.uint8` | 8 bits unsigned integer
`DT_STRING` | `tf.string` | Variable length byte arrays.  Each element of a Tensor is a byte array
`DT_BOOL` | `tf.bool` | Boolean
`DT_COMPLEX64` | `tf.complex64` | Complex number made of two 32 bits floating points: real and imaginary parts
`DT_QINT8` | `tf.qint8` | 8 bits signed integer used in quantized Ops
`DT_QINT32` | `tf.qint32` | 32 bits signed integer used in quantized Ops
`DT_QUINT8` | `tf.quint8` | 8 bits unsigned integer used in quantized Ops

## 3.1 Single-Device Execution

_**NOTE:** To reiterate- in this context, "single device" means using a single CPU core or single GPU, **not** a single machine. Similarly, "multi-device" does not refer to multiple machines, but to multiple CPU cores and/or GPUs. See "3.3 Distributed Execution" for multiple machine discussion._

* Overview of the execution of a single-worker process, single-device job:
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
			* This examination takes into account the execution time of the operation (given the device type), as well as the costs of possibly introducing communication between devices.
		3. The device that would finish the operation the soonest is selected as the node's device.
		4. Repeat steps 1-3 for each node in the graph execution until all nodes have been allocated to devices
	3. After the simulation, the real execution runs using the node-placement decisions made during the simulation
* Section 4.3 will describe some extensions to help guide the placement algorithm
* Improving the placement algorithm's development is an ongoing process as of writing

### Cross-Device Communication

* After the nodes have been placed onto their respective devices, the execution graph is split into subgraphs- one per device
* Any edge between nodes on different devices is replaced by two new edges:
	* The outputing node will have an edge between it and a new _Send_ node, placed within the subgraph of its device
	* The recieving node will have an edge between it and a new _Receive_ node, placed within the subgraph of its device
	* See Figure 4 for illustration of adding Send/Receive nodes
* The Send and Receive nodes coordinate data transfer across devices
	* This isolates cross-device communication to the implementation of the Send and Receive nodes
* All users of a particular tensor on a particular device use a single Receive node, as opposed to having one Receive node per user per device. This minimizes data transmission between devices as well as memory allocated on the receiving device
	* This means that any given device should receive the output of any given operation only once, and should store that output only once in memory
* This method of communication also allows individual node scheduling to be handled by the worker processes as opposed to the master
	* The Send and Receive nodes provide synchronization between worker processes and devices, which enables the master to only issue a single Run request per graph execution  per worker process
	* This improves scalability and fine-grain control over node execution

## 3.3 Distributed Execution

* Similar to multi-device execution. Send/Receive nodes that communicate across worker processes use mechanisms such as TCP or RDMA to transmit data from machine to machine

### Fault Tolerance

* There are two main ways failures are detected:
	* Errors between Send and Receive nodes
	* Periodic health-checks from the master process to the worker processes
* When a failure is detected, the entire graph execution is aborted and restarted
* Because TensorFlow Variables persist across each graph execution, there are mechanisms to save and restore their state
	* Each Variable node is connected to a Save node, which periodically write the contents of the Variable to persistent storage
	* Additionally, each Variable is connected to a Restore node, which is enabled when the graph execution is restarted. The Restore node reads state data from persistent storage and applies it to the Variable node

- - -

## 4 Extensions

_The following subsections describe advanced features and extensions of the programming model introduced in Section 2_

## 4.1 Gradient Computation

* TensorFlow has built-in gradient computation
* If a tensor, _C_, depends on a set of previous nodes, the gradient of _C_ with respect to those previous nodes can be automatically computed with a built-in function, even if there are many layers in between them
	* See [`tf.gradients()`](https://www.tensorflow.org/versions/master/api_docs/python/train.html#gradients) for usage
* Gradients are computed by creating additional nodes and edges in the graph as described below (see Figure 5):
	* When computing the gradient of a tensor, _C_, with respect to some dependency, _I_, TensorFlow first finds the forward path from _I_ to _C_ in the model graph. This is shown as the left-hand side of Figure 5
	* Once the path between the two is found, TensorFlow starts at _C_ and moves backward toward _I_. For every operation on this backward path, a node is added to the graph, composing the partial gradients of each added node via the [chain rule](https://en.wikipedia.org/wiki/Chain_rule). This is shown as the right-hand side of Figure 5
		* Partial gradients are computed via a "gradient function", which corresponds to an operation on the forward path. These gradient functions are provided alongside operation kernels
		* The gradient function takes as input the partial derivatives already computed along the backwards path and, optionally, the inputs and/or outputs of the corresponding operation on the forward path
		* For example, in Figure 5, the _dReLU_ operation (gradient function for the Rectified Linear Unit operation) takes in the previously computed partial derivatives (indicated by arrows coming from "..."), as well as the inputs from the _ReLU_ operation (indicated by arrows coming from _Add_, as the outputs of _Add_ are the inputs of _ReLU_). _dReLU_ does not, however, take in the outputs from the _ReLU_ function (indicated by the grayed out arrows coming from _ReLU). Once the partial gradients are computed, _dReLU_ outputs them to the next gradient function, in this case _dAdd_
	* The partial gradients for any node outputs that are **not** dependencies of _C_ are set to zero. This can occur when a node has multiple outputs, but only some of them connect to _C_
	* This process continues until the partial derivatives of _C_ with respect to _I_ are found
* Memory management for is an active area of improvement for the automatic gradient computation algorithm.
	* Tensors early in the computation graph may be needed at the end of gradient calculation, causing them to stick around in GPU memory
	* Current options for memory management improvements include improved heuristics to determine graph execution order, recomputing tensors as opposed to storing them in memory, and utilizing host CPU memory instead of leaving long-lived tensors in GPU memory

## 4.2 Partial Execution

* TensorFlow has built-in functionality to run smaller chunks of a defined execution graph, as well as the ability to insert pre-defined data as a replacement for any edge in the graph
* Each node in the graph is given a name upon instantiation, and each output of a node is referred to by number starting from zero
	* e.g. "bar:0" is the first output of node "bar"
* The Session's [run method](https://www.tensorflow.org/versions/master/api_docs/python/client.html#Session.run) takes two arguments, `fetches` and `feed_dict`, which define the subgraph to be executed:
	* `fetches` is a list of desired operation nodes and/or output tensors to be executed. If outputs are requested, the Run function will return the calculated tensors to the client (assuming the Run function is successful)
	* `feed_dict` is a dictionary of optional inputs, which map named node outputs (_`name:port`_) to defined tensor values. This allows a user to effectively define the 'start' of a subgraph. Additionally, `feed_dict` is used to define data in Placeholder objects
* The execution graph is then transformed based on the values fed to `fetches` and `feed_dict`
	* Each output tensor specified in `fetches` is connected to a **fetch** node, which stores and returns its value to the client once Run is successfully completed
		* Note: no fetch nodes are created for operation nodes named in `fetches`, as TensorFlow makes a distinction between opererations and the outputs of those operations
		* An example of an operation a user may specify as a `fetches` parameter to a Run command is the operation returned by [`tf.initialize_all_variables`](https://www.tensorflow.org/versions/master/api_docs/python/state_ops.html#initialize_all_variables). The operation doesn't provide an output, but it is run in the execution graph
	* Each named node output (`node:port`) specified in `feed_dict` is replaced with a **feed** node, which takes on the value of the tensor mapped to the named output. Each node in the execution graph that depends on the named output will take in data from the feed node in its place
		* Because a feed node has no dependencies, it is the start of its own execution chain
* Once the **fetch** and **feed** nodes have been inserted, TensorFlow determines which nodes need to be executed. It moves backwards, starting at the fetch nodes, and uses the dependencies of the graph to determine all nodes that must be executed in the modified graph in order to compute the desired outputs

## 4.3 Device Constraints

* Users can provide partial constraints on nodes about which devices they can run on
	* Examples: only allowing a node to run on GPUs, specifying a specific worker process/task, or ensuring that it is grouped with specific nodes
	* Note: By default, GPUs are given priority for device placement if the given operation has both a CPU and a GPU kernel implementation
* These constraints require modifications to the placement algorithm described in Section 3.2: 
	* After finding the feasible set of devices for each node, TensorFlow uses the constraints to determine which nodes must be grouped on the same device
	* For each of these groups, TensorFlow computes the intersection of feasible devices for each node in the group
	* Final selection of devices is determined using similar heuristics as described in Section 3.2, ensuring fast execution while taking device restrictions, such as memory, into account

_Aside: I'm not sure if this functionality is available in the open source implementation of TensorFlow yet. As of now I can only find information regarding placing nodes on **specific** devices. [Read more about manual device placement here](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/g3doc/how_tos/using_gpu/index.md#Manual-device-placement). Let me know if you can find the documentation for this feature!_

## 4.4 Control Flow

_These features are still being developed, and the API is not yet public or stable. Track the [open issue on GitHub](https://github.com/tensorflow/tensorflow/issues/208) to stay up-to-date!_

* TensorFlow incorporates a few primitive control flow operators which allow for the skipping of subgraph execution and the expression of iteration. Using these primitive operators, higher-level constructs such as `if` and `while` statements can be compiled into TensorFlow graphs
* Each iteration in a loop is identified with a unique tag, and the present value of that execution state is represented by a frame
* Inputs can enter an iteration whenever they are available
	* This allows multiple iterations to be executed concurrently
* Because a loop may contain nodes placed of separate devices, managing the state of loops becomes a problem of distributed termination detection. That is, there needs to be a way to detect the termination of nodes across devices.
* TensorFlow solves this by adding additional control nodes to the graph. These nodes manage the beginning and end of each loop iteration and determine when the loop should end.
	* At every iteration, the device that owns the control node for a loop sends out control messages to the other devices used in that loop
* The implementation of TensorFlow also takes `if` statements into account when computing gradients, as it must include or omit nodes as necessary to properly step backwards through the execution graph

## 4.5 Input Operations

* In addition to using the `feed_dict` parameter in the [`Session.run`](https://www.tensorflow.org/versions/master/api_docs/python/client.html#Session.run) method to manually feed in input data, TensorFlow supports reading tensors in directly from files
* Using this feature can reduce data transfer overhead when using TensorFlow on a distributed implementation (specifically when the client is on a different machine from the worker process):
	* Using `feed_dict` will cause data to first be sent from the storage system to the client, and then from client to the worker process
	* Reading from the file will cause data to be sent directly from the storage system to the worker process
* Data can be read in as individual data examples or in batches of examples
* TensorFlow classes for reading data:
	* Text/CSV: [`tf.TextLineReader`](https://www.tensorflow.org/versions/master/api_docs/python/io_ops.html#TextLineReader)
		* [Basics of CSV parsing in TensorFlow](https://www.tensorflow.org/versions/master/how_tos/reading_data/index.html#csv-files)
	* Fixed Length records: [`tf.FixedLengthRecordReader`](https://www.tensorflow.org/versions/master/api_docs/python/io_ops.html#FixedLengthRecordReader)
		* [Basics of fixed length record parsing in TensorFlow](https://www.tensorflow.org/versions/master/how_tos/reading_data/index.html#fixed-length-records)
	* TensorFlow data format: [`tf.TFRecordReader`](https://www.tensorflow.org/versions/master/api_docs/python/io_ops.html#TFRecordReader)
		* [Basics of TensorFlow data format handling](https://www.tensorflow.org/versions/master/how_tos/reading_data/index.html#standard-tensorflow-format)

## 4.6 Queues

* TensorFlow includes [Queues](https://www.tensorflow.org/versions/master/api_docs/python/io_ops.html#queues), which allow for the enqueuing and dequeuing of tensor objects. This enables asynchronous graph execution and the handing off of data between concurrent nodes
	* Enqueue operations can block until there is space available in the queue
	* Dequeue operations can block until a minimum number of elements are placed in the queue
* [`FIFOQueue`](https://www.tensorflow.org/versions/master/api_docs/python/io_ops.html#FIFOQueue) is a standard 'first-in, first-out' queue
* [`RandomShuffleQueue`](https://www.tensorflow.org/versions/master/api_docs/python/io_ops.html#RandomShuffleQueue) is a queue that randomly shuffles its elements periodically, which can be useful for machine learning algorithms that want to randomize training data
* An example use of queues is to allow input data to be prefetched from the storage system while previous data is still being processed

## 4.7 Containers

* A _Container_ is the mechanism that manages long-lived (i.e. survives multiple executions of a graph) mutable state
	* Container objects hold the values of _Variable_ objects
* There is a default container that persists until the process terminates
* Other named containers may be initialized
* Containers allow for the sharing of state between otherwise unrelated computation graphs defined on separate Sessions

- - -

## 5 Optimizations

_This section describes certain performance/resource usage optimizations used in the implementation of TensorFlow_

## 5.1 Common Subexpression Elimination

* Before execution, TensorFlow does a pass over the computation graph and reduces nodes with identical inputs and operations down to a single node.
* This prevents redundant execution of the same computation

## 5.2 Controlling Data Communication and Memory Usage

* Proper scheduling of operations can create dramatic improvements on data transfer and memory usage rates by reducing the amount of time intermediate data needs to be stored in memory
	* GPUs benefit from this a great deal, as they have scarce memory
	* Can also reduce competition amongst operations to use network resources
* One particular example used in the implementation TensorFlow is the scheduling of Recieve nodes (see "3.2 Cross Device Communication")
	* Recieve nodes, without thoughtful scheduling, may execute much earlier than necessary
		* This could cause data to be stored in the memory of devices for much longer
	* TensorFlow attempts to delay the execution of Recieve nodes until just before their results are needed.

## 5.3 Asynchronous Kernels

* TensorFlow supports non-blocking kernels, which are good for environments that can't afford having many active threads, and they allow nodes to wait for I/O or other events without blocking an execution thread
	* Normal, synchronous kernels complete their execution at the end of the Compute method
	* Asynchronous kernels use a slightly different interface: the Compute method is passed a lambda/callback that should be invoked at the end of the kernel's execution
* Examples of asynchronous kernels built into TensorFlow: Recieve, Enqueue, and Dequeue

## 5.4 Optimized Libraries for Kernel Implementations

* TensorFlow makes use of several existing, optimized libraries for many kernel implementations
* Library for linear algebra:
	* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) 
* Libraries for matrix multiplification on different devices:
	* [BLAS](http://www.netlib.org/blas/)
	* [cuBLAS (CUDA BLAS)](https://developer.nvidia.com/cublas)
* Libraries for convolutional kernels for deep neural networks:
	* [cuda-convnet](https://code.google.com/p/cuda-convnet/)
	* [cuDNN](https://developer.nvidia.com/cudnn)

## 5.5 Lossy Compression

* Because some machine learning algorithms still work well with less precise arithmetic, TensorFlow often uses lossy compression on higher-precision numbers when sending data between devices
	* This most often happens when sending data between devices on different machines, but it sometimes compresses data sent on the same machine
* For example, a 32-bit floating point number may be converted into a (for all intents and purposes) 16-bit floating point number before being sent to another device, where it is converted into a lossy version of the original 32-bit number
	* The 16-bit floating number is really just a 32-bit floating point with 16-bits less precision after the decimal
	* The conversion back to 32-bit floating point just fills in zeros to replace the lost precision after the decimal
		* 'Filling in with zeros' is used as the 16-bit -> 32-bit conversion menthod because it is faster than using stochastic rounding, even though the latter is more mathematically correct
		
- - -

## 6 Status and Experience

* The system, documentation, and examples for TensorFlow can be found at [tensorflow.org](https://www.tensorflow.org/)
* Currently, there are front-ends for Python and C++, and it's expected that more will be added over time (created both from internal Google users and the open-source community)

### Advice and Lessons Learned

_The following are "words of wisdom" coming from the experience of porting Google's_ [Inception](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) _neural network into TensorFlow. After successfully doing so, the team was rewarded with a 6-fold improvement on training time over DistBelief's implementation. This advice will hopefully be useful to others as they build their own models._

1. _Build tools to gain insight into the exact number of parameters in a given model_
	* This can help you catch subtle flaws in a complex network architecture, such as operations and variables instantiated incorrectly
2. _Start small and scale up_
	* The TensorFlow team started by importing a small network used by DistBelief
	* Debugging a small network gave insight into the edge cases of certain operations, while having to do the same on a larger network would be nearly impossible to figure out
3. _Always ensure that the objective (loss function) matches between machine learning systems when learning is turned off_
	* By setting the learning rate to zero (i.e. turning off learning), the TensorFlow team was able to identify unexpected behavior stemming from randomly initialized variables in the model
4. _Make a single machine implementation match before debugging a distributed implementation_
	* This helped the TensorFlow team separate and debug differences in training performance between DistBelief and TensorFlow
	* Once the single machine implementation worked, they were able to find bugs related to race conditions and non-atomic operations in the distributed model
5. _Guard against numerical errors_
	* Different numerical libraries handle non-finite floating point numbers differently
	* Checking for non-finite floating point values can allow one to detect errors in real time, guarding against numerical instability
6. _Analyze pieces of a network and understand the magnitude of numerical error_
	* By running subsections of the neural network on both DistBelief and TensorFlow in parallel, the team was able to ensure that the implemented algorithms were indeed identical
	* Note that because the networks used floating point numbers, there is a given amount of numerical error that should be expected and taken into account when comparing the two systems

- - -

## 7 Common Programming Idioms

_This section describes how TensorFlow's basic dataflow graphs can be used to speed up training neural network models on large datasets using techniques developed by the TensorFlow team._

_The techniques presented here assume that the model is using stochastic gradient descent with mini-batches of around 100-1000 examples_

### Data Parallel Training

* Users can parallelize the computation of the gradient, separating mini-batch elements onto different devices
	* For example, a mini-batch size of 1000 elements can be split into 10 smaller, parallel computations of 100 elements. After they all finish running, their results can be combined to achieve the same result as if the entire calculation was performed in a single, sequential computation
* This translates to having many replicas of a computational subgraph and using a single client thread to coordinate the training loop of those replicas
* The above approach can be modified further by making it asynchronous. Instead of having a single client thread, there are multiple clients (one for each subgraph replica), and each replica updates the trained parameters asynchronously
	* See Section 4 (pages 3-6) of [_Large Scale Distributed Deep Networks_](http://static.googleusercontent.com/media/research.google.com/en//archive/large_deep_networks_nips2012.pdf) for further description of asynchronous approach

### Model Parallel Training

* Can run separate portions of the computation graph on different devices simultaneously on the same batch of examples
	* See Figure 8 for a visual example of sequence-to-sequence learning parallelized across three devices

### Concurrent Steps for Model Computation Pipelining

* Can also run a small set of concurrent training steps on a single device
	* Similar concept to asynchronous data parallelism, but the parallelism is only on a single device
	* This can "fill in the gaps" of device utilization, when parallel execution on all devices might not make full use of computation cycles
	* See Figure 9 for a visual example

- - -

## 8 Performance

_Stay tuned for future versions of the TensorFlow white paper, which will include performance evaluations for single machine and distributed implementations of TensorFlow_

- - -

## 9 Tools

_This section discusses additional tools, developed by the TensorFlow team, that work alongside the graph modeling and execution features described above._

## 9.1 TensorBoard: Visualization of Graph Structures and Summary Statistics

_[TensorBoard](https://www.tensorflow.org/versions/master/resources/faq.html#tensorboard) was designed to help users visualize the structure of their graphs, as well as understand the behavior of their models_

### Visualization of Computation Graphs

* TensorBoard includes features that allow for more digestible visualizations 
	* TensorBoard can take models with tens of thousands of nodes and collapse them into high-level blocks, highlighting subgraphs with identical structures
	* It separates out "high-degree" nodes from the rest of the graph to further reduce visual clutter
		* _Note: I haven't found a proper definition of "high-degree" nodes in TensorFlow. The paper says they "often serve book-keeping functions". I imagine they are operations similar to [`tf.initialize_all_variables`](https://www.tensorflow.org/versions/master/api_docs/python/state_ops.html#initialize_all_variables), which are necessary to run the execution graph in TensorFlow but aren't really part of the mathematical model_
* The TensorBoard visualization is interactive
	* Users can pan, zoom, and expand the collapsed blocks in order to get a lower-level view of the model

### Visualization of Summary Data

* TensorBoard supports Summary operations that can be inserted into the execution graph to examine and track various values over time
	* _Scalar summaries_: e.g. average loss function over set of examples, total execution time of the model
	* _Histogram based summaries_: e.g. distribution of parameter values in a neural network layer
	* _Image-based summaries_: e.g. visualization of filter weights learned in a convolutional neural network
* Typically, Summary nodes are setup to monitor specific interesting values and are executed periodically during normal graph execution
* After Summary nodes are executed, the client writes the summary data to a log file. TensorBoard monitors this log to display summary information over time
	* The "time" used in the visualization of summary data can be: wall-clock time; absoute time; or "steps", the number of graph executions that have occurred since the first execution in the TensorFlow program

## 9.2 Performance tracing

* A tool called EEG is used to examine fine-grained information about the ordering/performance of TensorFlow graphs
	* Works for both single machine and distributed implementations of TensorFlow
	* _Note: EEG is not open sourced as of writing_
* Helps users understand bottlenecks in a TensorFlow program

_The following is a brief overview of what EEG does under the hood_

* Traces are collected via a number of sources including:
	* [Linux `ftrace`](http://elinux.org/Ftrace)
	* Internal Google tracing tools
	* [The CUDA Profiling Tools Interface](https://developer.nvidia.com/cuda-profiling-tools-interface)
* The trace logs enable EEG to recreate the execution of the graph with microsecond-level precision. Events from the traces, within a timerange, are extracted and visualized according to the resolution of the client's user interface
	* The user can zoom in on portions of the graph, and EEG will update the visualization to show finer details
* Any significant delays due to communication, synchronization, or direct memory access issues are identified and highlighted in the UI

_Please see pages 14 and 15 of the [November 2015 white paper](http://download.tensorflow.org/paper/whitepaper2015.pdf) to see a specific example of EEG visualization along with descriptions of the current UI_

- - -

## 10 Future Work

_This section lists areas of improvement and extension for TensorFlow identified for consideration by the TensorFlow team_

_Extensions_:

* A "function" mechanism, where users can specify subgraphs of TensorFlow execution to be reusable
	* In the TensorFlow team's design of this mechanism (not yet implemented), these kinds of functions can be reusable across front-end languages. That is, a user could define a subgraph function in Python and use it in C++ without redefining it

_Improvements_:

* Continue work on a just-in-time compiler that can take in an execution subgraph and output an optimized routine for that subgraph
	* Such a compiler might also take in some runtime profiling information as input
	* The compiler should be able to perform loop fusion, block and tile for locality, and specialize routines for particular shapes and sizes of tensors, along with other optimizations
* Improving the node placement and node scheduling algorithms
	* Instead of using man-made heuristics, have the system learn how to make good placement/scheduling decisions

- - -

## 11 Related Work

### Open source, single machine systems with portions of similar functionality

_Systems designed primarily for neural networks:_

* [Caffe](http://caffe.berkeleyvision.org/)
* [Chainer](http://chainer.org/)
* [Computational Network Toolkit](https://cntk.codeplex.com/)
* [Theano](http://deeplearning.net/software/theano/)
* [Torch](http://torch.ch/)

_Systems that support symbolic differentiation:_

* [Chainer](http://chainer.org/)
* [Theano](http://deeplearning.net/software/theano/)

_Systems with a core written in C++:_

* [Caffe](http://caffe.berkeleyvision.org/)

### Comparisons with DistBelief and Project Adam

_Similarities shared with [DistBelief](http://static.googleusercontent.com/media/research.google.com/en//archive/large_deep_networks_nips2012.pdf) and [Project Adam](https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-chilimbi.pdf):_

* They allow computations to be distributed across many devices on many machines
* Users can specify models using relatively high-level descriptions

_Differences between TensorFlow and DistBelief/Project Adam:_

* TensorFlow's dataflow graph model is more flexible and better at expressing more types of machine learning models and algorithms
* Allows for the expression of stateful Parameter nodes as variables
* Variable update operations are simply additional nodes in the graph
	* Both DistBelief and Project Adam have subsystems dedicated to handling parameters

### Comparison with [Hallide](http://people.csail.mit.edu/fredo/tmp/Halide-5min.pdf) image processing system

* Both use similar intermediate data representation to their dataflow graphs
* However, Hallide has additional high level knowledge of its operations, which it uses to create optimized code that combine multiple operations
* Hallide is single-machine only
	* The TensorFlow team hopes to extend TensorFlow's capabilities to incorporate Hallide's techniques into a distrubted setting

### Related distributed dataflow graph systems

_Systems that represent complex workflows as dataflow graphs_

* [Dryad](http://www.michaelisard.com/pubs/eurosys07.pdf)
* [Flume](https://flume.apache.org/)

_Systems that support data-dependent control flow_

* [CIEL](http://www.cs.princeton.edu/courses/archive/fall13/cos518/papers/ciel.pdf)
	* Iteration is implemented as a directed acyclic graph (DAG) that dynamically unfolds
* [Naiad](http://research.microsoft.com/en-us/projects/naiad/)
	* Iteration is implemented as a static graph with cycles

_Systems optimized for accessing the same data repeatedly_

* [Spark](http://spark.apache.org/)
	* Uses resilient distributed datasets (RDDs) to cache outputs of earlier operations, in case they are needed again

_Systems that execute dataflow graphs across heterogenous devices, including GPUs_

* [Dandelion](http://research-srv.microsoft.com/pubs/201110/sosp13-dandelion-final.pdf)

#### Features that TensorFlow incorporates from the above distributed systems

_Feature implementations that are most similar to TensorFlow are listed after the feature_

* The dataflow scheduler (i.e. the module that selects the next node to run)
	* CIEL, Dryad, Flume, Spark
* Distributed architecture: using a single, optimized dataflow graph and caching information about the graph to lower coordination overhead 
	* Naiad
* Works best when there is enough RAM in the cluster to hold all working variables/computations
	* Naiad, Spark
* Iteration: multiple replicas of the same graph can execute at once while sharing the same set of persistent variables. Replicas can share the variables asynchronously or use mechanisms such as queues to access them synchronously
	* Hybrid of many approaches
* Iteration: a node only executes when all of its dependencies have completed
	* CIEL
* Graph iteration is represented as a static, cyclic graph

- - -

## 12 Conclusions

* TensorFlow is a flexible dataflow graph programming model
* There are both single machine and distributed implementations of TensorFlow
* TensorFlow was developed using prior experience in Google, as well as methods used in other previous systems
* An [open source implementation of TensorFlow is available](https://github.com/tensorflow/tensorflow)
	* As of writing, only a single-device implementation has been released from Google

- - -

## Figures

#### Figure 1: Example TensorFlow code fragment
```python
	import tensorflow as tf
	
	# 100-d vector, init to zeros
	b = tf.Variable (tf.zeros([100])
	
	# 784x100 matrix with random values
	W = tf.Variable(tf.random_uniform([784,100], -1, 1))
	
	# Placeholder for input
	x = tf.placehoder(name="x")
	
	# Rectified linear unit of (W*x +b)
	relu = tf.nn.relu(tf.matmul(W, x) + b)
	
	# Cost computed as a function of relu
	C = [...]
	
	# Instantiate a Session
	s = tf.Session()
	
	for step in xrange(0, 10):
		# Create a 100-d vector for input
		input = ...construct 100-D input array ...
		
		# Find the cost, using the constructed vector as the placeholder input
		result = s.run(C, feed_dict = {x: input})
		print step, result
```

#### Figure 2: Corresponding computation graph for Figure 1
<img src="http://cdn.rawgit.com/samjabrahams/tensorflow-white-pages-notes/master/img/figure2.svg" id="figure2" style="max-height: 300px"></img>

#### Figure 3: Single machine (left) and distributed system (right) structure
<img src="http://cdn.rawgit.com/samjabrahams/tensorflow-white-pages-notes/master/img/figure3.svg" id="figure3" style="max-height: 300px"></img>

#### Figure 4: Before and after insertion of Send/Recieve nodes
<img src="http://cdn.rawgit.com/samjabrahams/tensorflow-white-pages-notes/master/img/figure4.svg" id="figure4" style="max-height: 300px"></img>

#### Figure 5: Gradients computed for graph in figure 2
<img src="http://cdn.rawgit.com/samjabrahams/tensorflow-white-pages-notes/master/img/figure5.svg" id="figure5" style="max-height: 300px"></img>

#### Figure 6: Before and after graph transformation for partial execution
<img src="http://cdn.rawgit.com/samjabrahams/tensorflow-white-pages-notes/master/img/figure6.svg" id="figure6" style="max-height: 300px"></img>

#### Figure 7: Synchronous and asynchronous data parallel training
<img src="http://cdn.rawgit.com/samjabrahams/tensorflow-white-pages-notes/master/img/figure7.svg" id="figure7" style="max-height: 300px"></img>

#### Figure 8: Model parallel training
<img src="http://cdn.rawgit.com/samjabrahams/tensorflow-white-pages-notes/master/img/figure8.svg" id="figure8" style="max-height: 300px"></img>

#### Figure 9: Concurrent steps
<img src="http://cdn.rawgit.com/samjabrahams/tensorflow-white-pages-notes/master/img/figure9.svg" id="figure9" style="max-height: 300px"></img>