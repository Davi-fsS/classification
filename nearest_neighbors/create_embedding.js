import { pipeline } from "@huggingface/transformers";
import fs from "fs";

const imgEmbedder = await pipeline("image-feature-extraction", "Xenova/clip-vit-base-patch32", {dtype: "fp32"});

async function embedImg(imgs){
    const embed = imgEmbedder(imgs, { pool: "cls", normalize: true })

    return embed.then(t => t.tolist());
}

const images = fs.readdirSync("./train").map(f => "./train/" + f);