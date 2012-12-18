package neuralnetwork.training

trait LearningFunction {
  def apply(iteration: Int): Double
}

class ConstantRate(rate: Double) extends LearningFunction {
  def apply(iteration: Int): Double = rate
}

object ConstantRate {
  def apply(rate: Double): ConstantRate = new ConstantRate(rate)
}

class AnnealingRate(rate: Double, iterations: Int) extends LearningFunction {
  def apply(iteration: Int): Double = rate / (1 + iteration.toDouble / iterations)
}

object AnnealingRate {
  def apply(rate: Double, iterations: Int): AnnealingRate = new AnnealingRate(rate, iterations: Int)
}