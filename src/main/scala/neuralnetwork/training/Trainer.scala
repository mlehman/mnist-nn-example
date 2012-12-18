package neuralnetwork.training

import neuralnetwork._
import neuralnetwork.data._


class Trainer(val numIterations: Int, val miniBatchSize: Int, val learningRate: LearningFunction,
  val momentumMultiplier: Double, val gradientChecker: Option[GradientChecker],  val evalIterations: Int) {

  def train(network: NeuralNetwork, trainingSet: DataSet) {

    val momentums = network.layers.map { _.weights.mul(0) }

    trainingSet.miniBatches(miniBatchSize).take(numIterations).zipWithIndex.foreach {
      case (batch, iteration) =>

        if ((iteration + 1) % evalIterations == 0) {
          val loss = network.loss(trainingSet)
          println("Iteration:%5d, Loss: %.5f, Accuracy: %f".format(iteration + 1, loss, network.eval(trainingSet)))
          for (check <- gradientChecker) {
	        check(network, batch)
          }
        }

        val gradients = network.errorGradients(batch)

        network.layers.zip(gradients).zip(momentums).foreach {
          case ((layer, gradient), momentum) =>
            val delta = momentum.muli(momentumMultiplier).subi(gradient)
            layer.weights.addi(delta.mul(learningRate(iteration)))
        }
    }
  }
}

object Trainer {

  def apply(numIterations: Int, miniBatchSize: Int = 100, learningRate: LearningFunction = ConstantRate(0.5),
    momentumMultiplier: Double = 1.0, gradientChecker: Option[GradientChecker] = None, evalIterations: Int = 100) =
    new Trainer(numIterations, miniBatchSize, learningRate, momentumMultiplier, gradientChecker, evalIterations)

}