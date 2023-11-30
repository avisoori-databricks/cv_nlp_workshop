# Databricks notebook source
# Image classification: batch scoring a corpus of images to identify whether they are shoes, shirts, dresses, etc. and Style tagging: labeling images as trendy, contemporary, etc.
# Similarity: determining if a given image is similar to other items

# COMMAND ----------

# MAGIC %pip install git+https://github.com/huggingface/transformers 
# MAGIC %pip install accelerate==0.21.0 bitsandbytes

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS conagra_demo;
# MAGIC USE CATALOG conagra_demo;
# MAGIC CREATE VOLUME IF NOT EXISTS conagra_images;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generating Image Descriptions

# COMMAND ----------

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
#Load the model
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl", load_in_8bit=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# COMMAND ----------

image = Image.open("/Volumes/rl_demo/default/rl_images/rl_1.jpeg").convert("RGB")
display(image)

# COMMAND ----------

prompt = "Describe the image in detail? Mention style and type. Please be concise"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
)
generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
print(generated_text)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Image Classification (zero-shot)

# COMMAND ----------

from transformers import pipeline
# More models in the model hub.
model_name = "openai/clip-vit-large-patch14-336"
classifier = pipeline("zero-shot-image-classification", model = model_name)


# COMMAND ----------

image = Image.open("/Volumes/conagra_demo/default/conagra_images/im1.jpeg").convert("RGB")
display(image)

# COMMAND ----------

image_to_classify = image
labels_for_classification =  ["chips","popcorn","fries"]
scores = classifier(image_to_classify, candidate_labels = labels_for_classification)
scores

# COMMAND ----------

import mlflow.pyfunc
from transformers import pipeline
import json

class ZeroShotImageClassifier(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        # Initialize the model
        model_name = "openai/clip-vit-large-patch14-336"
        self.classifier = pipeline("zero-shot-image-classification", model=model_name)
        # Load labels
        with open(context.artifacts["labels"], 'r') as file:
            self.labels = json.load(file)

    def predict(self, context, model_input):
        # Prediction logic
        results = []
        for image_path in model_input.iloc[:, 0]:
            scores = self.classifier(image_path, candidate_labels=self.labels)
            results.append(scores)
        return results

# Save labels to a file
labels_for_classification = ["jean", "shirt", "jacket","blazer", 'socks']
labels_path = "labels.json"
with open(labels_path, 'w') as file:
    json.dump(labels_for_classification, file)

# Log the model
with mlflow.start_run(run_name = "rl_image_classifier"):
    mlflow.pyfunc.log_model(
        artifact_path="zero_shot_image_classifier",
        python_model=ZeroShotImageClassifier(),
        artifacts={"labels": labels_path}
    )



# COMMAND ----------

import mlflow
from pyspark.sql.functions import struct, col
logged_model = 'runs:/741bf2ed8d384e12835b8453ff1c9552/zero_shot_image_classifier'

# Load model as a Spark UDF. Override result_type if the model does not return double values.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type='string')


# COMMAND ----------

import pandas as pd

# Replace these paths with the actual paths to your test images
test_image_paths = [
    "/Volumes/rl_demo/default/rl_images/rl_3.webp",
    "/Volumes/rl_demo/default/rl_images/rl_2.webp",
    "/Volumes/rl_demo/default/rl_images/rl_1.jpeg"
]

image_df = spark.createDataFrame(pd.DataFrame(test_image_paths, columns=["image_path"]))


# COMMAND ----------

display(image_df)

# COMMAND ----------

image_df.withColumn('predictions', loaded_model(struct(*map(col, image_df.columns))))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Zero Shot Object detection

# COMMAND ----------

from transformers import pipeline

checkpoint = "google/owlvit-base-patch32"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection")

# COMMAND ----------

image = image = Image.open("/Volumes/conagra_demo/default/conagra_images/im2.jpeg").convert("RGB")
display(image)

# COMMAND ----------

predictions = detector(
    image,
    candidate_labels=["bowl", "coffee filter", "kettle", "pitcher", "bazooka"],
)
predictions

# COMMAND ----------

from PIL import ImageDraw

draw = ImageDraw.Draw(image)

for prediction in predictions:
    box = prediction["box"]
    label = prediction["label"]
    score = prediction["score"]

    xmin, ymin, xmax, ymax = box.values()
    draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
    draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="white")

image

# COMMAND ----------

# MAGIC %md
# MAGIC ### Image Segmentation with SAM

# COMMAND ----------

from transformers import pipeline
generator =  pipeline("mask-generation", device = 0, points_per_batch = 256)
outputs = generator(image, points_per_batch = 256)

# COMMAND ----------

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    

plt.imshow(np.array(image))
ax = plt.gca()
for mask in outputs["masks"]:
    show_mask(mask, ax=ax, random_color=True)


# COMMAND ----------

del classifier
del detector
del generator

import torch
import gc

gc.collect()
torch.cuda.empty_cache()


# COMMAND ----------


