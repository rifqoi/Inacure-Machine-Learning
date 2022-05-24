const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");

function readImage(path) {
  //reads the entire contents of a file.
  //readFileSync() is synchronous and blocks execution until finished.
  const imageBuffer = fs.readFileSync(path);
  //Given the encoded bytes of an image,
  //it returns a 3D or 4D tensor of the decoded image. Supports BMP, GIF, JPEG and PNG formats.
  const tfimage = tf.node.decodeImage(imageBuffer, 3);
  const resized = tf.image.resizeBilinear(tfimage, [224, 224]).toFloat();
  // Normalize the image
  const offset = tf.scalar(255.0);
  const normalized = tf.scalar(1.0).sub(resized.div(offset));
  //We add a dimension to get a batch shape
  const batched = normalized.expandDims(0);
  return batched;
}

function argMax(array) {
  return Array.prototype.map
    .call(array, (x, i) => [x, i])
    .reduce((r, a) => (a[0] > r[0] ? a : r));
}

async function imageClassification(path) {
  const image = readImage(path);
  // Load the model.
  const output = {};
  const model = await tf.loadGraphModel(
    "file://vgg19_model/vgg19_saved_model/model.json"
  );

  const predictions = await model.predict(image).dataSync();
  console.log("Classification Results:", predictions);
  // let finalPredictions = argMax(predictions);
  return output;
}

if (process.argv.length !== 3)
  throw new Error("Incorrect arguments: node classify.js <IMAGE_FILE>");

imageClassification(process.argv[2]);
