# multitask-prompting

### Overview

Prompting uses the capability of large language models (LLM's) to "fill in the blank" in order to classify the meaning of text. I conducted research on how a single model can use prompting to simultaneously learn multiple tasks. [RoBERTa](https://arxiv.org/abs/1907.11692) was the primary model I experimented with. Here are the [results](https://urf.columbia.edu/sites/default/files/symposium/LI%20Evan-%20Poster.pdf). This repository contains the code I used to train and evaluate models during my experiments.
<br>

Here is a diagram from [this paper](https://aclanthology.org/2021.acl-long.295.pdf) that explains the difference between prompting and head-based fine-tuning for language models.
<img width="1157" alt="image" src="https://user-images.githubusercontent.com/42917263/188295612-225713b2-688d-4cdb-b92e-09f2a62b6059.png">
The paper [How Many Data Points is a Prompt Worth?](https://arxiv.org/pdf/2103.08493.pdf) performs some cool experiments showing the power of prompts in low resource settings.
<br>


### RobertaPrompt

RobertaPrompt wraps around HuggingFace's [RobertaForMaskedLM](https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaForMaskedLM) class and allows developers to train and test a Roberta model using prompting based on a prompt definition <br>

Suppose we have two tasks: given an argument and a topic, we must detect if the argument was in support or against the topic, and also whether or not the argument contained a fallacy (if it does, what exact fallacy the argument contains). <br>

A prompt definition would contain: <br>
1) A template for each task. A template is a consistent text pattern associated with a task so the model recognizes which task needs to be completed. For example, <br>
```
"Stance detection task. Topic: {insert topic here} and Argument: {insert argument here}. The stance is: <mask>"
```
2) A policy function for each task. The policy function maps the token that the model uses to fill in the blank with the predicted label.

My experiments trained a Roberta model to accomplish the exact tasks mentioned above - you can take a look at some example predictions in the <b>prompting_example.ipynb</b> notebook

## Training and Testing
First, load a base model. A GPU as the device is highly reccomended.
```
pmodel = RobertaPrompt(model='roberta-large', device = torch.device('cuda'), prompt = argument_prompt)
```
Start training immediately by specifying the paths to a training and validation dataset. Training statistics will be displayed in stdout.
```
pmodel.train("sample_train_set.tsv", "sample_val_set.tsv", output_dir="sample_model", epochs=10)
```
After training is finished, evaluate the model on a test set using the following function and save the test results
```
pmodel.test("sample_test_set.tsv", save_path='stats.txt')
```
You should see text content in this format in the file specificed by ```save_path```. Overall f1 scores are included, along with more fine-grained statistics on model performance for each label <br> <br>
<img width="465" alt="image" src="https://user-images.githubusercontent.com/42917263/188295865-3158d842-d84d-4aec-b432-0eebfac4b141.png">

One can then use this model and fine-tune it on other tasks with different prompts.

### Data
Sample data for fallacious argument and stance detection is from [Argotario](https://aclanthology.org/D17-2002/).
