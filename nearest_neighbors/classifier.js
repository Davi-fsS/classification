import { cos_sim } from "@huggingface/transformers";
import fs from "fs";

const embeddings = JSON.parse(fs.readFileSync("./embeddings.json"));

const trains = embeddings.filter(e => e["split"] == "train")
const tests = embeddings.filter(e => e["split"] == "test")

// calcular para um dado de teste, a distancia (cos_sin) com todos os dados de treinamento
function compare(test){
    let distances = [];

    for(let train of trains){
        const distance = cos_sim(test["embedding"], train["embedding"])
    
        distances.push({
            distance: distance,
            class: train["class"]
        })
    }

    return distances
}

function getKNearestNeighbors(distances, k){
    const sortedDistances = distances.sort((a, b) => {
        if(a["distance"] > b["distance"]){
            return -1;
        } else {
            return 1;
        }
    })

    return sortedDistances.slice(0, k);
}

function countClasses(knn){
    const classCount = {};

    for(let n of knn){
        if(classCount[n["class"]]){
            classCount[n["class"]] = classCount[n["class"]] + 1
        }
        else{
            classCount[n["class"]] = 1
        }
    }

    return classCount
}

function getMaxClass(classCount){
    let maxClass = null;
    let maxClassCount = 0;

    for(let cls in classCount){
        if(classCount[cls] > maxClassCount){
            maxClassCount = classCount[cls];
            maxClass = cls;
        }
    }

    return maxClass;
}

function knnClassifier(test, k){
    const distances = compare(test);

    const knn = getKNearestNeighbors(distances, k);

    const classCount = countClasses(knn);

    const predictedClass = getMaxClass(classCount);

    return predictedClass
}

console.log(knnClassifier(tests[0], 5))