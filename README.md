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

## Training

    val trainer = Trainer(
      numIterations = 3000,
      miniBatchSize = 100,
      numParallel = 0, //Try using more of those cores!
      learningRate = ConstantRate(0.3), //Others to try: AnnealingRate(0.35, iterations = 5000)
      momentumMultiplier = 0.9, 
      gradientChecker = None, //To check gradients try: Some(GradientChecker(numChecks = 10, accuracy = 8))
      evalIterations = 1000)

      trainer.train(network, trainSet)

## Saving / Loading

    NeuralNetwork.save(network,"my-network.obj")
    val network = NeuralNetwork.load("my-network.obj")