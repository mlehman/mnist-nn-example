package neuralnetwork

import org.jblas.DoubleMatrix
import org.jblas.MatrixFunctions._
import neuralnetwork.data._

class NeuralNetwork(val layers: List[Layer], val objective: ObjectiveFunction, weightDecay: Double) extends Serializable {

  @transient val decay = new WeightDecay(weightDecay)

  def copy(layers: List[Layer]): NeuralNetwork =
    new NeuralNetwork(layers, objective, weightDecay)

  def forwardProp(inputs: DoubleMatrix): List[LayerState] =
    layers.scanLeft(LayerState(None, inputs)) {
      case (LayerState(_, x), layer) => layer(x)
    }.tail

  def errorGradients(data: DataSet): Seq[DoubleMatrix] = {

    val outputs = forwardProp(data.inputs)
    val errorDerivative = objective.derivative(outputs.last.activationOutput, data.targets)

    val derivatives = (0 until layers.size).scanRight(errorDerivative) {
      case (i, priorDerivative) =>
        val priorDerivativeWeighted = if (i < layers.size - 1) {
          layers(i + 1).weights.mmul(priorDerivative)
        } else priorDerivative
        val derivative = layers(i).activation.derivative(outputs(i).compositionOutput.get, outputs(i).activationOutput)
        derivative.mul(priorDerivativeWeighted)
    }

    derivatives.zipWithIndex.map {
      case (derivative, i) =>
        val gradient = if (i > 0) {
          outputs(i - 1).activationOutput.mmul(derivative.transpose)
        } else {
          data.inputs.mmul(derivative.transpose)
        }
        if(i < layers.size) {
          decay.derivative(layers(i), gradient)
        } else gradient
    }
  }

  def loss(data: DataSet): Double = {
    val outputs = forwardProp(data.inputs)
    decay(layers, objective(outputs.last.activationOutput, data.targets))
  }

  def eval(data: DataSet): Double = {
    val outputs = forwardProp(data.inputs)
    val scores = outputs.last.activationOutput.columnArgmaxs.zip(data.targets.columnArgmaxs).map {
      case (p, t) =>
        if (p == t) 1 else 0
    }
    scores.sum.toDouble / scores.length
  }
}

object NeuralNetwork {
  import java.io._

  def apply(layers: List[Layer], objective: ObjectiveFunction, weightDecay: Double = 0.0): NeuralNetwork = {
    new NeuralNetwork(layers, objective, weightDecay)
  }

  def printDimensions(n: String, m: DoubleMatrix) {
    println("Matrix '%s': %dx%d".format(n, m.rows, m.columns))
  }
  
  def save(network: NeuralNetwork, file: String) {
    val os = new ObjectOutputStream(new FileOutputStream(file))
    try {
      os.writeObject(network)
    } finally {
      os.close()
    }
  }
  
  def load(file: String): NeuralNetwork = {
    val is = new ObjectInputStream(new FileInputStream(file))
    try {
      is.readObject().asInstanceOf[NeuralNetwork]
    } finally {
      is.close()
    }
  }
}