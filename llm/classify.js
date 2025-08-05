import { createPartFromUri, createUserContent, GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";
import fs from "fs";

dotenv.config();

const genai = new GoogleGenAI({ apiKey: process.env.GOOGLE_GEN_AI_API_KEY });

const embeddings = JSON.parse(fs.readFileSync("../nearest_neighbors/embeddings.json"));

const testInstances = embeddings.filter(e => e["split"] == "test").map(e => {
    return {
        trueClass: e["class"],
        path:  "../nearest_neighbors" + e["path"].slice(1)
    }
})

function readImg(path){
    return fs.readFileSync(path, { encoding: "base64" })
}

function toInlineData(imageB64){
    return {
        inlineData: {
            mimeType: "image/jpeg",
            data: imageB64
        }
    }
}

const outputConfig = {
    responseMimeType: "application/json",
    responseSchema: {
        type: Type.ARRAY,
        items: {
            type: Type.OBJECT,
            properties: {
                category: {
                    type: Type.STRING,
                    enum: ["dog", "cat"]
                }
            }
        }
    }
}

const prompt = `  
    Identifique se a imagem contém gatos ou cachorros.
    Retorne se uma das seguintes categorias de acordo com o
    conteúdo da imagem: 
    "cat" caso a imagem contenha um ou mais gatos, ou
    "dog" caso a imagem contenha um ou mais cachorros
`

async function geminiRequest(contents){
    const response = await genai.models.generateContent({
        model: "gemini-2.0-flash",
        contents: contents,
        config: outputConfig
    });

    return response
}

function calculateAccuracy(results){
    let correct = 0;

    for(let result of results){
        if(result["predictedClass"] == result["trueClass"]){
            correct++
        }
    }

    return correct / results.length
}

async function llmClassifier(path){
    const imgB64 = readImg(path);
    const imgInlineData = toInlineData(imgB64);

    const contents = [imgInlineData, {text: prompt}]

    const response = await geminiRequest(contents);

    return JSON.parse(response.text)[0]["category"]
}

console.log(await llmClassifier(testInstances[500]["path"]))

// vendo a acuracidade da google
const requests = testInstances.slice(0,10).map(i => llmClassifier(i["path"]))

await Promise.all(requests);

for(let i = 0; i < testInstances.length; i++){
    testInstances[i]["predictedClass"] = await requests[i]
}

console.log(calculateAccuracy(testInstances))