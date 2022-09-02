# multitask-prompting

### Overview

Prompting uses the capability of large language models (LLM's) to "fill in the missing text" in order to classify the meaning of text. I conducted research on how a single model can use prompting to simultaneously learn multiple tasks. [RoBERTa](https://arxiv.org/abs/1907.11692) was the primary model I experimented with. Here are the [results](https://urf.columbia.edu/sites/default/files/symposium/LI%20Evan-%20Poster.pdf).

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
