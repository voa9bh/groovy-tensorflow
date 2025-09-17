/*
 *  Licensed to the Apache Software Foundation (ASF) under one
 *  or more contributor license agreements.  See the NOTICE file
 *  distributed with this work for additional information
 *  regarding copyright ownership.  The ASF licenses this file
 *  to you under the Apache License, Version 2.0 (the
 *  "License"); you may not use this file except in compliance
 *  with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an
 *  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 *  KIND, either express or implied.  See the License for the
 *  specific language governing permissions and limitations
 *  under the License.
 */

import groovy.transform.Field
import org.apache.wayang.basic.model.DLModel
import org.apache.wayang.basic.model.op.*
import org.apache.wayang.basic.model.op.nn.CrossEntropyLoss
import org.apache.wayang.basic.model.op.nn.Linear
import org.apache.wayang.basic.model.op.nn.Sigmoid
import org.apache.wayang.basic.model.optimizer.Adam
import org.apache.wayang.core.util.Tuple
import org.apache.wayang.basic.model.optimizer.Optimizer
import org.apache.wayang.basic.operators.*
import org.apache.wayang.core.api.WayangContext
import org.apache.wayang.core.plan.wayangplan.Operator
import org.apache.wayang.core.plan.wayangplan.WayangPlan
import org.apache.wayang.java.Java
import org.apache.wayang.tensorflow.Tensorflow

import static org.apache.wayang.basic.operators.LocalCallbackSink.createCollectingSink

class Iris implements Runnable {
  URI TEST_PATH = getClass().classLoader.getResource("iris_test.csv").toURI()
  URI TRAIN_PATH = getClass().classLoader.getResource("iris_train.csv").toURI()
  Map LABEL_MAP = ["Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2]

  static void main(String[] args) {
    new Iris().run()
  }

  void run() {
    var trainSource = fileOperation(TRAIN_PATH, true)
    var testSource = fileOperation(TEST_PATH, false)

    /* labels & features */
    Operator trainData = trainSource.field0
    Operator trainLabel = trainSource.field1
    Operator testData = testSource.field0
    Operator testLabel = testSource.field1

    int[] noShape = null
    var features = new Input(noShape, Input.Type.FEATURES)
    var labels = new Input(noShape, Input.Type.LABEL, Op.DType.INT32)

    /* model */
    Op l1 = new Linear(4, 32, true)
    Op s1 = new Sigmoid().with(l1.with(features))
    Op l2 = new Linear(32, 3, true).with(s1)
    DLModel model = new DLModel(l2)

    /* training options */
    Op criterion = new CrossEntropyLoss(3).with(model.out, labels)
    Optimizer optimizer = new Adam(0.1f) // optimizer with learning rate
    int batchSize = 45
    int epoch = 10
    var option = new DLTrainingOperator.Option(criterion, optimizer, batchSize, epoch)
    option.setAccuracyCalculation(new Mean(0).with(
      new Cast(Op.DType.FLOAT32).with(
        new Eq().with(new ArgMax(1).with(model.out), labels)
      )))
    var trainingOp = new DLTrainingOperator<>(model, option, float[], Integer)

    var predictOp = new PredictOperator<>(float[], float[])

    /* map to label */
    var bestFitOp = new MapOperator<>((array) -> array.toList().withIndex().max { it.v1 }.v2, float[], Integer)

    /* sink */
    var predicted = []
    var predictedSink = createCollectingSink(predicted, Integer)

    var groundTruth = []
    var groundTruthSink = createCollectingSink(groundTruth, Integer)

    trainData.connectTo(0, trainingOp, 0)
    trainLabel.connectTo(0, trainingOp, 1)
    trainingOp.connectTo(0, predictOp, 0)
    testData.connectTo(0, predictOp, 1)
    predictOp.connectTo(0, bestFitOp, 0)
    bestFitOp.connectTo(0, predictedSink, 0)
    testLabel.connectTo(0, groundTruthSink, 0)

    var wayangPlan = new WayangPlan(predictedSink, groundTruthSink)

    new WayangContext().with {
      register(Java.basicPlugin())
      register(Tensorflow.plugin())
      execute(wayangPlan)
    }

    println "labels:       $LABEL_MAP"
    println "predicted:    $predicted"
    println "ground truth: $groundTruth"

    var correct = predicted.indices.count { predicted[it] == groundTruth[it] }
    println "test accuracy: ${correct / predicted.size()}"
  }

  def fileOperation(URI uri, boolean random) {
    var textFileSource = new TextFileSource(uri.toString()) // <1>
    var line2tupleOp = new MapOperator<>((line) -> line.split(",").with { // <2>
      new Tuple(it[0..-2]*.toFloat() as float[], LABEL_MAP[it[-1]])
    }, String, Tuple)

    var mapData = new MapOperator<>((tuple) -> tuple.field0, Tuple, float[]) // <3>
    var mapLabel = new MapOperator<>((tuple) -> tuple.field1, Tuple, Integer) // <3>

    if (random) {
      Random r = new Random()
      var randomOp = new SortOperator<>((e) -> r.nextInt(), String, Integer) // <4>
      textFileSource.connectTo(0, randomOp, 0)
      randomOp.connectTo(0, line2tupleOp, 0)
    } else {
      textFileSource.connectTo(0, line2tupleOp, 0)
    }

    line2tupleOp.connectTo(0, mapData, 0)
    line2tupleOp.connectTo(0, mapLabel, 0)

    new Tuple<>(mapData, mapLabel)
  }
}
