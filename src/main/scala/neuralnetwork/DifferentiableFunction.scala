package neuralnetwork

import org.jblas.DoubleMatrix
import org.jblas.MatrixFunctions._

trait DifferentiableFunction extends Serializable {
  def derivative(x: DoubleMatrix, y: DoubleMatrix): DoubleMatrix
}