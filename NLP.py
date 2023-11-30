# Databricks notebook source
# MAGIC %pip install setfit

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generating Embeddings with pretrained encoder model

# COMMAND ----------

from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss

from setfit import SetFitModel, SetFitTrainer, sample_dataset
import numpy as np
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.decomposition import PCA
import plotly.express as px
import gc

# COMMAND ----------

dataset = load_dataset('SetFit/yelp_review_full')

# COMMAND ----------

dataset

# COMMAND ----------

yelp_sample = dataset['train'].to_pandas()
yelp_sample = yelp_sample[yelp_sample['label'].isin([0, 4])].sample(n=1000)


# COMMAND ----------

# Simulate the few-shot regime by sampling 8 examples per class
train_dataset = sample_dataset(dataset["train"], label_column="label", num_samples=8)
eval_dataset = dataset["test"]

# COMMAND ----------

# Load a SetFit model from Hub
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")


# COMMAND ----------

def generate_embeddings(text):  
  return  model.model_body.encode(text, convert_to_tensor=True).cpu().squeeze().numpy()

# COMMAND ----------

max_seq_length = 1024

# Truncate the text column to maximum sequence length
yelp_sample['truncated_text'] = yelp_sample['text'].apply(lambda text: text[:max_seq_length])


# COMMAND ----------

yelp_sample['Embeddings'] = yelp_sample['truncated_text'].apply(generate_embeddings)


# COMMAND ----------

display(yelp_sample)

# COMMAND ----------

yelp_sample['label'] = yelp_sample['label'].apply(lambda x: 'positive' if x >= 4 else 'negative')


# COMMAND ----------

embeddings = yelp_sample['Embeddings'].tolist()
pca = PCA(n_components=3)
pca_result = pca.fit_transform(embeddings)
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2', 'PC3'])
pca_df['labels'] = yelp_sample.label.to_list()
pca_df['text'] = yelp_sample.text.to_list()

# COMMAND ----------

fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='labels', hover_data=['text'])
fig.update_traces(hovertemplate='Text: %{customdata[0]}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Finetuning Embedding model, generating embeddings, and visualizing embeddings

# COMMAND ----------


# Create trainer
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss_class=CosineSimilarityLoss,
    metric="accuracy",
    batch_size=16,
    num_iterations=20, # The number of text pairs to generate for contrastive learning
    num_epochs=1, # The number of epochs to use for contrastive learning
    column_mapping={"text": "text", "label": "label"} # Map dataset columns to text/label expected by trainer
)


# COMMAND ----------

# Train and evaluate
trainer.train()
metrics = trainer.evaluate()

# COMMAND ----------

import numpy as np
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.decomposition import PCA
import plotly.express as px

# COMMAND ----------

model.model_body.encode('hey how are you?', convert_to_tensor=True).cpu().squeeze().numpy()

# COMMAND ----------

preds = model(["This restaurant is awesome!!", "I can't stand the pizza they serve here smh"])
preds

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generating Embeddings

# COMMAND ----------

max_seq_length = 1024

# Truncate the text column to maximum sequence length
yelp_sample['truncated_text'] = yelp_sample['text'].apply(lambda text: text[:max_seq_length])



# COMMAND ----------

yelp_sample['Embeddings'] = yelp_sample['truncated_text'].apply(generate_embeddings)


# COMMAND ----------

display(yelp_sample)

# COMMAND ----------

embeddings = yelp_sample['Embeddings'].tolist()
pca = PCA(n_components=3)
pca_result = pca.fit_transform(embeddings)
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2', 'PC3'])
pca_df['labels'] = yelp_sample.label.to_list()
pca_df['text'] = yelp_sample.text.to_list()


# COMMAND ----------

fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='labels', hover_data=['text'])
fig.update_traces(hovertemplate='Text: %{customdata[0]}')


# COMMAND ----------

embedding_df = spark.createDataFrame(yelp_sample)
embedding_df.count()

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG dais23_data_sharing;
# MAGIC USE DATABASE dais23_ml_db;
# MAGIC

# COMMAND ----------

embedding_df.write.mode('overwrite').saveAsTable('finetuned_embedding_table')

# COMMAND ----------


