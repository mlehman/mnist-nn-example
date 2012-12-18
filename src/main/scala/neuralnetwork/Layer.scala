package neuralnetwork

import org.jblas.DoubleMatrix

case class LayerState(compositionOutput: Option[DoubleMatrix], activationOutput: DoubleMatrix)

class Layer(val weights: DoubleMatrix, val activation: ActivationFunction) extends Serializable {
  
  def numInputs = weights.rows
  def numOutputs = weights.columns

  def composition(x: DoubleMatrix): DoubleMatrix = weights.transpose.mmul(x)

  def apply(x: DoubleMatrix): LayerState = {
    val c = composition(x)
    LayerState(Some(c), activation(c))
  }

  def copy(weights: DoubleMatrix = this.weights,
    activation: ActivationFunction = this.activation): Layer =
    new Layer(weights, activation)
  
}

class PartialLayer(val numOutputs: Int, val activation: ActivationFunction)

class LayerList(val layers: List[Layer]) {
  def :+(partial: PartialLayer): LayerList = new LayerList(layers:+Layer(layers.last.numOutputs, partial.numOutputs, partial.activation))
}

object Layer {
  
  implicit def layer2LayerList(layer: Layer) = new LayerList(List(layer))
  implicit def layerList2list(list: LayerList) = list.layers

  def apply(numInputs: Int, numOutputs: Int, activation: ActivationFunction): Layer =
    new Layer(randomWeights(numInputs, numOutputs), activation)
  
  def apply(numOutputs: Int, activation: ActivationFunction): PartialLayer =
    new PartialLayer(numOutputs, activation)

  def randomWeights(numInputs: Int, numOutputs: Int, absMax: Double = 0.01): DoubleMatrix = {
    val r = DoubleMatrix.rand(numOutputs, numInputs)
    r.muli(2).subi(1).muli(absMax).transpose
  }

}