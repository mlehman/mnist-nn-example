# Neural Network Example using MNIST Data Set

An example of neural networks implemented in Scala (with the [jblas](http://jblas.org/) linear algebra library) for recognizing hand-written digits.

## Building & Running the Example

Requires java and [sbt](http://www.scala-sbt.org/). 

Building and running:

    $ sbt run
    
This will run the example in src/main/scala/example/FeedForwardExample.scala
    
## Configuration Examples

### 2 layer network:
 
    val network = NeuralNetwork(
      Layer(trainSet.numInputs, 50, HyperbolicTangent):+Layer(trainSet.numOutputs, SoftMax),
      objective = CrossEntropyError)

### 3 layer network:

    val network = NeuralNetwork(
      Layer(trainSet.numInputs, 300, Logistic):+Layer(200, Logistic):+Layer(trainSet.numOutputs, SoftMax),
      objective = CrossEntropyError,
      weightDecay = 0.001)

## Saving / Loading

    NeuralNetwork.save(network,"my-network.obj")
    val network = NeuralNetwork.load("my-network.obj")