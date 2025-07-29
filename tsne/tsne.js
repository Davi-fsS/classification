import TSNE from "tsne-js";
import fs from "fs";

let embeddings = JSON.parse(fs.readFileSync("../nearest_neighbors/embeddings.json"));

let tsneInput = [];

for(let embedding of embeddings){
    if(embedding["number"] >= 500 && embedding["number"] < 600){
        tsneInput.push(embedding)
    }
}

let model = new TSNE({
    dim: 2,
    perplexity: 30.0,
    earlyExaggeration: 4.0,
    learningRate: 100.0,
    nIter: 1000,
    metric: "euclidean"
});

model.init({
    data: tsneInput.map(i => i["embedding"]),
    type: "dense"
});

model.run();

let output = model.getOutput();

console.log(output)