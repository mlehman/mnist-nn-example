package neuralnetwork

import org.jblas.DoubleMatrix
import org.jblas.MatrixFunctions._

trait ActivationFunction extends DifferentiableFunction {
  def apply(x: DoubleMatrix): DoubleMatrix
}

object SoftMax extends ActivationFunction {
  def apply(x: DoubleMatrix): DoubleMatrix = {
    val normalizer = logSumExp(x)
    val logProb = x.subi(normalizer.repmat(x.rows, 1))
    exp(logProb) 
  }
  
  def logSumExp(input: DoubleMatrix): DoubleMatrix = {
    val maxSmall = input.columnMaxs
    val maxBig = maxSmall.repmat(input.rows, 1)
    val l = logi(expi(input.sub(maxBig)).columnSums)
    l.addi(maxSmall)
  }

  def derivative(x: DoubleMatrix, y: DoubleMatrix): DoubleMatrix =
    y.mul(0).addi(1)
}

object Logistic extends ActivationFunction {
  def apply(x: DoubleMatrix): DoubleMatrix =
    powi(expi(x.neg).addi(1), -1)

  def derivative(x: DoubleMatrix, y: DoubleMatrix): DoubleMatrix =
    y.mul(y.neg.addi(1))
}

object HyperbolicTangent extends ActivationFunction {
  def apply(x: DoubleMatrix): DoubleMatrix =
    tanh(x)

  def derivative(x: DoubleMatrix, y: DoubleMatrix): DoubleMatrix =
    pow(y,2).negi.addi(1)
}