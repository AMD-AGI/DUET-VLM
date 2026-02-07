import spacy
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already present
nltk.data.path.append('/wekafs/aditysin/nltk_data')
nltk.download('stopwords', download_dir='/wekafs/aditysin/nltk_data', quiet=True)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define question words
QUESTION_WORDS = {"who", "what", "where", "when", "why", "how", "which", "whom", "whose"}

# Stopwords (NLTK)
STOPWORDS = set(stopwords.words('english'))

# Important punctuation to keep
IMPORTANT_PUNCT = {"?", "!", ":"}



def salient_tokens_finder(sentence):
    doc = nlp(sentence)
    candidates = []

    for token in doc:
        text = token.text.strip()
        lower = text.lower()

        # Rule 1: Keep question words (override stopword filtering)
        if lower in QUESTION_WORDS:
            candidates.append(text)
            continue  # skip further checks

        # Rule 2: Keep important punctuation (like ? ! :)
        if text in IMPORTANT_PUNCT:
            candidates.append(text)
            continue

        # Rule 3: Skip stopwords
        if lower in STOPWORDS:
            continue

        # Rule 4: Keep nouns, pronouns, verbs, proper nouns
        if token.pos_ in {"NOUN", "PROPN", "PRON", "VERB", "ADJ", "ADV"}:
            candidates.append(text)

    # Deduplicate while preserving order
    seen = set()
    result = []
    for word in candidates:
        if word not in seen:
            result.append(word)
            seen.add(word)

    return result




if __name__ == "__main__":
    # Example
    sentence = "Who discovered the theory of relativity in New York?"
    print(salient_tokens_finder(sentence))
    # Adding Adjectives and adverbs
    sentence = "Who discovered slowly the best theory of relativity in the lowest New York?"
    print(salient_tokens_finder(sentence))
    # r, tc = get_saliency(sentence)
    # print(f"Results: {r[:10]}, Target class: {tc}")
    # plot_saliency_bar(sentence)

# from transformers import BertTokenizer, BertForSequenceClassification
# from captum.attr import LayerIntegratedGradients
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap

# # 1. Load pretrained fine-tuned model (sentiment analysis on SST-2)
# model_name = "textattack/bert-base-uncased-SST-2"
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForSequenceClassification.from_pretrained(model_name)
# model.eval()

# # 2. Forward function for Captum
# def forward_func(input_ids, attention_mask):
#     return model(input_ids=input_ids, attention_mask=attention_mask).logits

# # 3. Setup LayerIntegratedGradients on embeddings layer
# lig = LayerIntegratedGradients(forward_func, model.bert.embeddings)

# # 4. Saliency function
# def get_saliency(sentence):
#     inputs = tokenizer(sentence, return_tensors="pt")
#     input_ids = inputs["input_ids"]
#     attention_mask = inputs["attention_mask"]

#     # Predict class
#     logits = model(**inputs).logits
#     target_class = logits.argmax(dim=-1).item()

#     # Compute attributions
#     attributions, delta = lig.attribute(
#         inputs=input_ids,
#         additional_forward_args=(attention_mask,),
#         target=target_class,
#         return_convergence_delta=True
#     )

#     # Aggregate across embedding dimensions
#     scores = attributions.sum(dim=-1).squeeze(0).detach()
#     scores = scores.abs()  # importance magnitude
#     scores = scores / scores.max()  # normalize 0–1

#     tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
#     results = [(tok, float(score)) for tok, score in zip(tokens, scores) if tok not in ["[CLS]", "[SEP]", "[PAD]"]]
#     # sorted(results, key=lambda x: x[1], reverse=True)
#     return results, target_class

# 5. Bar plot visualization
# 5. Optional: Heatmap visualization
# def plot_saliency_bar(sentence):
#     results, target_class = get_saliency(sentence)
#     tokens, scores = zip(*results)

#     # Custom colormap (white → red)
#     cmap = LinearSegmentedColormap.from_list("white_red", ["#ffffff", "#ff0000"])
#     colors = [cmap(score) for score in scores]

#     plt.figure(figsize=(6, 4))
#     bars = plt.bar(tokens, scores, color=colors, edgecolor="black")
#     plt.xticks(rotation=45, ha="right", fontsize=12)
#     plt.ylabel("Normalized Saliency", fontsize=12)
#     plt.title(f"Token Saliency (class={target_class})", fontsize=14)
#     plt.tight_layout()
#     plt.savefig("saliency_bar.png")
#     plt.show()

# ==================================== NLTK removing stop words ==========================
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
# from transformers import BertTokenizer, BertForSequenceClassification
# from captum.attr import LayerIntegratedGradients

# # Optional: nltk stopwords
# import nltk
# from nltk.corpus import stopwords
# nltk.download("stopwords")
# STOPWORDS = set(stopwords.words("english"))

# # 1. Load pretrained fine-tuned model (sentiment analysis on SST-2)
# model_name = "textattack/bert-base-uncased-SST-2"
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForSequenceClassification.from_pretrained(model_name)
# model.eval()

# # 2. Forward function for Captum
# def forward_func(input_ids, attention_mask):
#     return model(input_ids=input_ids, attention_mask=attention_mask).logits

# # 3. Setup LayerIntegratedGradients on embeddings layer
# lig = LayerIntegratedGradients(forward_func, model.bert.embeddings)

# # 4. Saliency function
# def get_saliency(sentence):
#     inputs = tokenizer(sentence, return_tensors="pt")
#     input_ids = inputs["input_ids"]
#     attention_mask = inputs["attention_mask"]

#     # Predict class
#     logits = model(**inputs).logits
#     target_class = logits.argmax(dim=-1).item()

#     # Compute attributions
#     attributions, delta = lig.attribute(
#         inputs=input_ids,
#         additional_forward_args=(attention_mask,),
#         target=target_class,
#         return_convergence_delta=True
#     )

#     # Aggregate across embedding dimensions
#     scores = attributions.sum(dim=-1).squeeze(0).detach()
#     scores = scores.abs()  # importance magnitude
#     scores = scores / scores.max()  # normalize 0–1

#     tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

#     # Filter special tokens + stopwords
#     results = [
#         (tok, float(score))
#         for tok, score in zip(tokens, scores)
#         if tok not in ["[CLS]", "[SEP]", "[PAD]"]
#         and tok.lower() not in STOPWORDS
#     ]

#     return results, target_class

# # 5. Bar plot visualization
# def plot_saliency_bar(sentence, top_k=None):
#     results, target_class = get_saliency(sentence)
#     tokens, scores = zip(*results)

#     # Sort by importance if top_k requested
#     if top_k:
#         sorted_pairs = sorted(zip(tokens, scores), key=lambda x: x[1], reverse=True)[:top_k]
#         tokens, scores = zip(*sorted_pairs)

#     # Custom colormap (white → red)
#     cmap = LinearSegmentedColormap.from_list("white_red", ["#ffffff", "#ff0000"])
#     colors = [cmap(score) for score in scores]

#     plt.figure(figsize=(12, 4))
#     bars = plt.bar(tokens, scores, color=colors, edgecolor="black")
#     plt.xticks(rotation=45, ha="right", fontsize=12)
#     plt.ylabel("Normalized Saliency", fontsize=12)
#     plt.title(f"Token Saliency (class={target_class})", fontsize=14)
#     plt.tight_layout()
#     plt.savefig("saliency_bar.png")
#     plt.show()

# # Example
# sentence = "Who discovered the theory of relativity in New York?"
# r, tc = get_saliency(sentence)
# print(f"Results: {r[:10]}, Target class: {tc}")
# plot_saliency_bar(sentence)

# ==================================== FIRST version ==========================
# from transformers import BertTokenizer, BertForSequenceClassification
# import torch
# from captum.attr import IntegratedGradients

# # Load model
# model_name = "bert-base-uncased"
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
# model.eval()

# def salient_tokens_bert(sentence):
#     inputs = tokenizer(sentence, return_tensors="pt")
#     input_ids = inputs["input_ids"]
#     outputs = model(**inputs)
#     ig = IntegratedGradients(model)
    
#     def forward_func(input_ids):
#         return model(input_ids)[0]
#     import ipdb; ipdb.set_trace()
#     attributions, delta = ig.attribute(inputs=input_ids.long(), target=1, return_convergence_delta=True)
    
#     tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
#     scores = attributions.sum(dim=-1).squeeze(0).tolist()
    
#     return list(zip(tokens, scores))

# sentence = "Who discovered the theory of relativity in New York?"
# print(salient_tokens_bert(sentence)[:10])
