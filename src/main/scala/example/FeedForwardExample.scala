package example

import neuralnetwork._
import neuralnetwork.data._
import neuralnetwork.training._

object FeedForwardExample {

  val trainingSetDir = "data/train"
  val testSetDir = "data/test"

  def main(args: Array[String]) {

    val trainSet = ImageTileDataSet(trainingSetDir, numOfClasses = 10, imageDimension = 28)
    println("Training Examples: " + trainSet.numExamples)

    val network = NeuralNetwork(
      Layer(trainSet.numInputs, 100, Logistic):+Layer(trainSet.numOfClasses, SoftMax),
      objective = CrossEntropyError,
      weightDecay = 0.0001)

    println("Initial Classification Accuracy: %f".format(network.eval(trainSet)))

    val trainer = Trainer(
      numIterations = 5000,
      miniBatchSize = 100,
      learningRate = ConstantRate(0.3), //Others to try: AnnealingRate(0.35, iterations = 5000)
      momentumMultiplier = 0.9, 
      gradientChecker = None //To check gradients try: Some(GradientChecker(numChecks = 10, accuracy = 8))
      )
      
    trainer.train(network, trainSet)

    val testSet = ImageTileDataSet(testSetDir, numOfClasses = 10, imageDimension = 28)

    println("Test Examples: " + testSet.numExamples)
    println("Test Classification Accuracy: %f".format(network.eval(testSet)))

  }

}


