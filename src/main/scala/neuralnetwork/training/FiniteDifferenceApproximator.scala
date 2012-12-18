package neuralnetwork.training

class FiniteDifferenceApproximator(val distanceWithCoefficents: Seq[(Int, Double)], val h: Double) {

  def apply(f: (Double) => Double) = (distanceWithCoefficents.map {
    case (dist, weight) => f(h * dist) * weight
  }).sum / h

}

object FiniteDifferenceApproximator {
  
  val firstDerivativeCoefficients: Map[Int, List[Double]] = Map(
      2 -> List(-1/2D,1/2D),
      4 -> List(1/12D, -2/3D, 2/3D, -1/12D),
      6 -> List(-1/60D, 3/20D, -3/4D, 3/4D, -3/20D, 1/60D),
      8 -> List(1/280D, -4/105D, 1/5D, -4/5D, 4/5D, -1/5D, 4/105D, -1/280D))
      
  val firstDerivativeDistanceWithCoefficent = distanceWithCoefficent(firstDerivativeCoefficients)
  
  def distanceWithCoefficent(coefficientsMap: Map[Int, List[Double]]): Map[Int, Seq[(Int, Double)]] = coefficientsMap.mapValues {
    coefficients => (((-coefficients.length / 2) to -1) ++ (1 to (coefficients.length / 2))).zip(coefficients)
  }
  
  def apply(h: Double, accuracy: Int = 8): FiniteDifferenceApproximator = firstDerivativeDistanceWithCoefficent.get(accuracy) match {
    case Some(distanceWithCoefficents) => new FiniteDifferenceApproximator(distanceWithCoefficents, h)
    case None => throw new Exception("Unsupported accuracy.")
  }  

}