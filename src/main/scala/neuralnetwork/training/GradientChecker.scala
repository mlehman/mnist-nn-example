package neuralnetwork.training

import org.jblas.DoubleMatrix
import scala.util.Random

import neuralnetwork._
import neuralnetwork.data._

class GradientChecker(val numChecks: Int, val accuracy: Int) {
  
  def apply(network: NeuralNetwork, data: DataSet) {

    val gradients = network.errorGradients(data)
    print("Checking gradients: ")

    (0 until network.layers.size).reverse.foreach {
      i =>
        print("layer(" + i + ")")
        
        assertValid(gradients(i), network.layers(i).weights)

        checkGradient(gradients(i), network.layers(i).weights, data, numChecks, accuracy,
          (weights: DoubleMatrix) => network.copy(network.layers.updated(i, network.layers(i).copy(weights))))

        print("passed! ")
    }

  }

  def checkGradient(gradients: DoubleMatrix, weights: DoubleMatrix, data: DataSet, checks: Int, accuracy: Int, stepNetwork: (DoubleMatrix) => NeuralNetwork) {
    val threshold = 1e-5
    val fd = FiniteDifferenceApproximator(h = 1e-2, accuracy)

    randomIndices(weights.rows, weights.columns)(checks).foreach {
      case (row, col) =>
        print(".")
        val gradient = gradients.get(row, col)

        val stepWeights = weights.dup()
        val approximatedGradient = fd {
          step =>
            val weight = weights.get(row, col)
            stepWeights.put(row, col, weight + step)
            stepNetwork(stepWeights).loss(data)
        }

        val diff = math.abs(gradient - approximatedGradient)
        assert((diff < threshold) || (diff / (math.abs(gradient) + math.abs(approximatedGradient)) < threshold),
          "Gradient looks like an error. Actual: %g, Estimated: %g, Diff: %g".format(gradient, approximatedGradient, diff))
    }
  }

  def randomIndices(rows: Int, columns: Int)(n: Int) = Seq.fill(n)((Random.nextInt(rows), Random.nextInt(columns)))

  def assertValid(gradients: DoubleMatrix, weights: DoubleMatrix) {
    assert(gradients.rows == weights.rows && gradients.columns == weights.columns,
      "Gradient %dx%d did not match weights %dx%d.".format(gradients.rows, gradients.columns, weights.rows, weights.columns))
    for (row <- (0 until gradients.rows); column <- (0 until gradients.columns)) {
      val gradient = gradients.get(row, column)
      assert(!gradient.isNaN, "Gradient contained NaN.")
      assert(!gradient.isInfinity, "Gradient contained infinity.")
    }
  }

}

object GradientChecker {
  def apply (numChecks: Int, accuracy: Int): GradientChecker = new GradientChecker(numChecks, accuracy)
}