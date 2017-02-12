package joe.spark

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.feature.{LabeledPoint, OneHotEncoder, StringIndexer}
import org.apache.spark.ml.linalg.{SparseVector, Vectors}
import org.apache.spark.ml.regression.GeneralizedLinearRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object BaseballRegression extends Logging {

  def main(args: Array[String]) {
    // local[*] will create a worker for every logically core on our computer.
    // This configuration is just for local testing, you'll need a different context for deploying to a cluster.
    val session = SparkSession.builder().appName("Baseball Regression").master("local[*]").getOrCreate()
    // We suppress a lot of logging so we can see our standard out more clearly.
    LogManager.getLogger("org.apache.spark").setLevel(Level.WARN)

    // Implicits for encoding DataSets
    import session.implicits._

    // Now we have a DataFrame containing all our players and their stats.
    val players = session.read.option("header","true").csv(FileUtils.resolvePathFromClasspath("baseball_players.csv"))

    // Some interesting stats before we get to the ML!
    info("")
    info("-----------------------------------")
    val countOfPlayers = players.count()
    val countOfTeams = players.select("Team").distinct().count()
    info(s"Data contains $countOfPlayers players in $countOfTeams teams.")

    val youngestPlayer = players.sort("Age").head()
    info(s"The youngest player is ${youngestPlayer.getAs[String]("Name")} (${youngestPlayer.getAs[Double]("Age")} years old).")

    val heaviestPlayer = players.sort(desc("Weight")).head()
    info(s"The heaviest player is ${heaviestPlayer.getAs[String]("Name")} (${heaviestPlayer.getAs[Double]("Weight")}lb).")
    info("-----------------------------------")
    info("")

    // Now we get to the main event!
    val maxAge = players.select(max("Age")).head().getAs[String](0).toDouble
    val maxHeight = players.select(max("Height")).head().getAs[String](0).toDouble
    val maxWeight = players.select(max("Weight")).head().getAs[String](0).toDouble

    val indexedPlayers = new StringIndexer()
      .setInputCol("Position")
      .setOutputCol("PositionIndex")
      .fit(players)
      .transform(players)

    val hotPlayers = new OneHotEncoder()
      .setInputCol("PositionIndex")
      .setOutputCol("PositionIndexHot")
      .transform(indexedPlayers)

    val playerFeatures = hotPlayers.map { row =>
      val weight   = row.getAs[String]("Weight").toDouble
      val height   = row.getAs[String]("Height").toDouble
      val age      = row.getAs[String]("Age").toDouble
      val position = row.getAs[SparseVector]("PositionIndexHot")
      val features = Array(height / maxHeight, age / maxAge) ++ position.toArray
      LabeledPoint(weight / maxWeight, Vectors.dense(features))
    }

    val lr = new GeneralizedLinearRegression()
      .setFamily("gaussian")
      .setLink("identity")
      .setMaxIter(10)
      .setRegParam(0.3)

    val lrModel = lr.fit(playerFeatures)

    info(s"coefficients: " + lrModel.coefficients)
    info(s"intercept: " + lrModel.intercept)

    val inputs = session.createDataset(Seq(Bob(Vectors.dense(77.0 / maxHeight, 30.0 / maxAge, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))))
    val result = lrModel.transform(inputs)
    info("Guess: " + result.select("prediction").head().getAs[Double](0) * maxWeight)

    session.stop()
  }

}

case class Bob(features: org.apache.spark.ml.linalg.Vector)