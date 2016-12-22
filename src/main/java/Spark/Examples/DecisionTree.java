package Spark.Examples;

import java.util.HashMap;
import java.util.Map;

import scala.Tuple2;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.util.MLUtils;

public class DecisionTree {

	public static void main(String[] args) {

		SparkConf sparkConf = new SparkConf().setAppName("Decision Tree");
		JavaSparkContext jsc = new JavaSparkContext(sparkConf);

		// Load and parse the data file.
		String datapath = "data/mllib/sample_libsvm_data.txt";
		JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), datapath).toJavaRDD();

		// Split the data into training and test sets (30% held out for testing)
		JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});
		JavaRDD<LabeledPoint> trainingData = splits[0];
		JavaRDD<LabeledPoint> testData = splits[1];

		// Set parameters.
		//  Empty categoricalFeaturesInfo indicates all features are continuous.
		Integer numClasses = 2;
		Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
		String impurity = "gini";
		Integer maxDepth = 5;
		Integer maxBins = 32;

		// Train a DecisionTree model for classification.
		DecisionTreeModel model = org.apache.spark.mllib.tree.DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins);

		// Evaluate model on test instances, get pair of true/predicted class
		JavaPairRDD<Double, Double> predictionAndLabel = testData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
			@Override
			public Tuple2<Double, Double> call(LabeledPoint p) {
				return new Tuple2<>(model.predict(p.features()), p.label());
			}
		});

		Double testErr = predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
			@Override
			public Boolean call(Tuple2<Double, Double> pl) {
				return !pl._1().equals(pl._2());
			}
		}).count() / (double) testData.count();

		System.out.println("Test Error: " + testErr);
		System.out.println("Learned classification tree model:\n" + model.toDebugString());

		// Save and load model
		model.save(jsc.sc(), "myDecisionTreeClassificationModel");

		// If I had to load the model trained whithout having to train it again
		DecisionTreeModel sameModel = DecisionTreeModel.load(jsc.sc(), "myDecisionTreeClassificationModel");
	}
}
