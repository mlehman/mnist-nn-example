package neuralnetwork

import org.jblas.DoubleMatrix
import org.jblas.MatrixFunctions._

trait ObjectiveFunction extends DifferentiableFunction {
  def apply(y: DoubleMatrix, t: DoubleMatrix): Double
}

object CrossEntropyError extends ObjectiveFunction {
  def apply(y: DoubleMatrix, t: DoubleMatrix): Double =
    -((log(y).muli(t).columnSums).mean)

  def derivative(x: DoubleMatrix, y: DoubleMatrix): DoubleMatrix =
    x.sub(y).muli(1.0 / y.columns)
}

class WeightDecay(coefficient: Double) {
  def apply(layers: List[Layer], loss: Double): Double =
    loss + (layers.map{ l => pow(l.weights,2).sum }.sum / 2 * coefficient)

  def derivative(layer: Layer, layerDerivative: DoubleMatrix): DoubleMatrix =
    layerDerivative.add(layer.weights.mul(coefficient))
}

object WeightDecay {
  def apply(coefficient: Double): WeightDecay = new WeightDecay(coefficient)
}