import { pipeline } from "@huggingface/transformers";
import fs from "fs";

const imgEmbedder = await pipeline("image-feature-extraction", "Xenova/clip-vit-base-patch32", {dtype: "fp32"});

async function embedImg(img) {
    return imgEmbedder(img, { pooling: "cls", normalize: true }).then(t => t.tolist());
}

const images = fs.readdirSync("./train").map(f => "./train/" + f);

let startIndex = 0;

while (startIndex < images.length){
    let endIndex = startIndex + 500;

    let imgsToEmbed = images.slice(startIndex, endIndex);

    const embeddings = await embedImg(imgsToEmbed);
    const output = []

    for(let i = 0; i < embeddings.length; i++){
        output.push({
            "path": images[i + startIndex],
            "embedding": embeddings[i]
        })
    }

    fs.writeFileSync(`embeddings/embedding_${startIndex}.json`, JSON.stringify(output))

    startIndex = endIndex;
}