package neuralnetwork.data

import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import org.jblas.DoubleMatrix
import scala.util.Random
import scala.Array.canBuildFrom


abstract trait DataSet {
  
  val inputs: DoubleMatrix
  val targets: DoubleMatrix
  
  val numExamples = inputs.columns
  val numInputs = inputs.rows
  val numOutputs = targets.rows
  
  def copy(inputs: DoubleMatrix, targets: DoubleMatrix): DataSet
  
  def batch(columns: Array[Int] ): DataSet = {
    copy(inputs.getColumns(columns), targets.getColumns(columns))
  }
  
  def miniBatches(batchSize: Int): Iterator[DataSet] = {
    Stream.continually((0 until numExamples)).flatten.grouped(batchSize).map {
      columns =>
        batch(columns.toArray)
    }
  }
    
}

class ImageTileDataSet(val inputs: DoubleMatrix, val targets: DoubleMatrix, val numClasses: Int, val imageDimension: Int) extends DataSet {
  def copy(inputs: DoubleMatrix, targets: DoubleMatrix) = new ImageTileDataSet(inputs, targets, numClasses, imageDimension)
}

object ImageTileDataSet {

  case class ImageExample(pixels: Array[Double], target: Int) {
    def isBlank(): Boolean = { pixels.sum == 0 }
  }

  def apply(directory: String, numClasses: Int, imageDimension: Int): ImageTileDataSet = {

    def extractPixels(image: BufferedImage, x: Int, y: Int, d: Int): Array[Double] = {
      val pixels = for {
        py <- (y until y + d)
        px <- (x until x + d)
      } yield ((image.getRGB(px, py) >> 16) & 0xff) / 255.0
      pixels.toArray
    }
    
    print("Loading tilesets '" + directory + "': ( ")
    val trainingExamples = (0 until numClasses).flatMap {
      i =>
        val tileSet = new File(directory, i + ".jpg")
        print( i + " ")
        val image = ImageIO.read(tileSet)

        val images = for {
          y <- (0 until image.getHeight by imageDimension)
          x <- (0 until image.getWidth by imageDimension)

        } yield ImageExample(extractPixels(image, x, y, imageDimension), i)
        images.filter { !_.isBlank }
    }
    println(")")

    val numOfExamples = trainingExamples.size
    val imagePixels = imageDimension * imageDimension
    val inputs = DoubleMatrix.zeros(numOfExamples, imagePixels)
    val targets = DoubleMatrix.zeros(numOfExamples, numClasses)

    Random.shuffle(trainingExamples).zipWithIndex.foreach {
      case (example, row) =>
        example.pixels.zipWithIndex.foreach {
          case (pixel, col) => inputs.put(row, col, pixel)
        }
        targets.put(row, example.target, 1.0)
    }

    new ImageTileDataSet(inputs.transpose, targets.transpose, numClasses, imageDimension)
  }

  def printImage(image: DoubleMatrix, dim: Int) {
    val chars = "  .:-=+*#%@".zipWithIndex.map(_.swap).toMap
    def pixelAsChar(p: Double): Char = chars((p * 10).floor.toInt)

    (0 until image.columns by dim).foreach {
      rowIndex =>
        (rowIndex until (rowIndex + dim)).foreach {
          pixelIndex => print(pixelAsChar(image.get(0, pixelIndex)) + " ")
        }
        println()
    }
  }

}