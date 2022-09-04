import csv
import torch
import random
import numpy as np
import time
import datetime
import os
import json
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
from transformers import RobertaTokenizer, RobertaForMaskedLM, AdamWeightDecay, WarmUp

class Prompt:
    # A class to format train and test samples for prompting.
    # Provides an interface that RobertaPrompt uses to format train and test data.
    def __init__(self, templates: dict, policy_functions: dict):
        self.templates = templates
        self.policy_fn = policy_functions
        self.state_str = self.__str__()
    def test_sample(self, inputs: list, task: str) -> str:
        return self.templates[task].format(*inputs)
    def train_sample(self, inputs: list, label: str, task: str) -> str:
        sample = self.templates[task].format(*inputs, label)
        return sample.replace("<mask>", label)
    def get_prediction(self, pred: str, task: str) -> str:
        return self.policy_fn[task](pred)
    def add_task(self, task_name: str, task_template: str, policy_function):
        self.policy_fn[task_name] = policy_function
        self.templates[task_name] = task_template
        self.state_str = self.__str__()
    def remove_task(self, task_name):
        del self.policy_fn[task_name]
        del self.templates[task_name]
        self.state_str = self.__str__()
    def __str__(self):
        tasks = ["Task: {}, Template: {}".format(task, template) for task, template in self.templates.items()]
        return "=== Prompts ===\n" + "\n".join(tasks)
        


class RobertaPrompt:
    def __init__(self, device: torch.device, prompt: Prompt, model='roberta-large', new_tokens = None, max_length = 512):
        '''
            device - a device of type torch.device to train the model on
            model - type of roberta model, either be one of 'roberta-large', 'roberta-medium', 'roberta-small', or some path to a trained model
            new_tokens - a list of new token strings to add to the models vocabulary
            max_length - max token indices able to be fed into roberta
        '''
        self.tokenizer = RobertaTokenizer.from_pretrained(model, add_special_tokens=True)
        if new_tokens: self.tokenizer.add_tokens(new_tokens)
        self.model = RobertaForMaskedLM.from_pretrained(model, max_length=max_length)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.device = device if device else torch.device('cpu')
        self.model.to(self.device)
        self.prompt = prompt
        self.base_type = model

    def reload(self, model_path: str):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForMaskedLM.from_pretrained(model_path)
        self.model.to(self.device)
  
    def infer(self, inputs: list, task: str) -> str:

        '''
            given a sample (defined by inputs and task), this method returns the predicted masked token
            this method assumes that 'sent' is in the correct formatted prompt
        '''
        sent = self.prompt.test_sample(inputs, task)
        preds = self._infer(sent)
        return self.prompt.get_prediction(preds, task)

    def _infer(self, sent: str):
        '''
            helper function for infer(self, inputs: list, task: str) -> str:
            takes a raw str and fills in the <mask> tokens
        '''
        token_ids = self.tokenizer.encode(sent, return_tensors='pt').to(self.device)
        masked_position = (token_ids.squeeze() == self.tokenizer.mask_token_id).nonzero()
        masked_pos = [mask.item() for mask in masked_position]

        with torch.no_grad():
            output = self.model(token_ids)

        last_hidden_state = output[0].squeeze()

        preds = []
        for mask_index in masked_pos:
            mask_hidden_state = last_hidden_state[mask_index]
            masked_idx = torch.topk(mask_hidden_state, k=1, dim=0)[1]
            masked_word = self.tokenizer.decode(masked_idx.item()).strip()
            preds.append(masked_word)
        return preds[0] #currently assume only one masked token for classification tasks - add on to make in future
  
    def test(self, test_set: str, save_path='stats.txt') -> str:

        '''
            this method returns f1 score, precision, recall, and accuracy
            and writes the statistics to stats.txt under the model folder (if it exists, else it writes to the current directory) 
        '''
        y_preds = []
        y_true = []
        with open(test_set, 'r') as f:
            samples = csv.reader(f, delimiter='\t')
            for sample in samples:
                label, task = sample[-2:]
                inputs = sample[:-2]
                pred = self.infer(inputs, task)
                y_preds.append(pred)
                y_true.append(label)

        stats = []
        stats.append("macro f1: " + str(f1_score(y_true, y_preds, average='macro')))
        stats.append("micro f1: " + str(f1_score(y_true, y_preds, average='micro')))
        stats.append("weighted f1: " + str(f1_score(y_true, y_preds, average='weighted')))
        stats.append(classification_report(y_true, y_preds))
        stats = "\n".join(stats)

        with open(save_path, 'w') as f:
            f.write(stats)

        return stats

    def train(self, train_path: str, val_path: str, output_dir='roberta-prompt-model', epochs=10) -> str:
        '''
            this method trains and saves a model (based on best validation loss) to 'output_dir' given
            a train set and a test set
        '''
        self.output_dir = output_dir
        optimizer = AdamWeightDecay(learning_rate = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                                    epsilon = 1e-8 # args.adam_epsilon  - default is 1e-8.
                                    )
        train_dataloader, val_dataloader = self.load_training_data(train_path, val_path)

        scheduler = WarmUp(initial_learning_rate  = 2e-5, decay_schedule_fn = optimizer, warmup_steps = 0)

        def format_time(elapsed):
            elapsed_rounded = int(round((elapsed)))
            return str(datetime.timedelta(seconds=elapsed_rounded))

        def save_model(output_dir):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)

        # Training based off run_glue.py at https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        training_stats = {}

        total_t0 = time.time()

        best_val_loss = float('inf')

        for epoch in range(epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
            print('Training...')

            t0 = time.time()

            total_train_loss = 0

            for step, batch in enumerate(train_dataloader):

                # Have updates to the training process
                if step % 40 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)
                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                # Unpack this training batch from our dataloader. 
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # clear gradients
                self.model.zero_grad()        

                # forward pass through the model
                result = self.model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels,
                            return_dict=True)
                loss = result.loss
                logits = result.logits
                total_train_loss += loss.item()

                # backward pass
                loss.backward()
                # clip gradients https://machinelearningmastery.com/exploding-gradients-in-neural-networks/
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # Update parameters
                optimizer.step()
                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)
            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)
            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))
            print("")
            print("Running Validation...")

            t0 = time.time()
            # eval mode during validation
            self.model.eval()

            # Tracking variables 
            total_eval_loss = 0

            # Evaluate data for one epoch
            for batch in val_dataloader:

                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                with torch.no_grad():        
                    result = self.model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask,
                                labels=b_labels,
                                return_dict=True)

                loss = result.loss
                logits = result.logits
                    
                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()

                avg_val_loss = total_eval_loss / len(val_dataloader)
  
                if avg_val_loss <= best_val_loss:
                  print("SAVING NEW MODEL ... ")
                  best_val_loss = avg_val_loss
                  save_model(output_dir)

                # Measure how long the validation run took.
                validation_time = format_time(time.time() - t0)
                print("  Validation Loss: {0:.2f}".format(avg_val_loss))
                print("  Validation took: {:}".format(validation_time))

                # Record all statistics from this epoch.
                training_stats[epoch+1] = {
                        'Training Loss': avg_train_loss,
                        'Valid. Loss': avg_val_loss,
                        'Training Time': training_time,
                        'Validation Time': validation_time
                    }
        print("")
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

        with open(output_dir + '/training_stats', 'w') as f:
          dump = json.dump(training_stats, f, indent=4)

        return training_stats

    def load_training_data(self, train: str, val: str):
        '''
            helper function for train
        '''
        train_arr = []
        val_arr = []
        with open(train, 'r') as f:
            rd = csv.reader(f, delimiter='\t')
            for sample in rd:
                label, task = sample[-2:]
                inputs = sample[:-2]
                train_arr.append([self.prompt.test_sample(inputs, task), self.prompt.train_sample(inputs, label, task)])
        with open(val, 'r') as f:
            rd = csv.reader(f, delimiter='\t')
            for sample in rd:
                label, task = sample[-2:]
                inputs = sample[:-2]
                val_arr.append([self.prompt.test_sample(inputs, task), self.prompt.train_sample(inputs, label, task)])
        return self.get_dloaders(train_arr, val_arr)

    def get_dloaders(self, train, val):
        '''
            helper function for train that returns the dataloaders for training and validation data
        '''
        train_dataset = self.encode_data(train)
        val_dataset = self.encode_data(val)

        batch_size = 16
 
        train_dataloader = DataLoader(
                    train_dataset,  # The training samples.
                    sampler = RandomSampler(train_dataset), # Select batches randomly
                    batch_size = batch_size # Trains with this batch size.
                )
        validation_dataloader = DataLoader(
                    val_dataset, # The validation samples.
                    sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                    batch_size = batch_size # Evaluate with this batch size.
                )

        return train_dataloader , validation_dataloader

    def encode_data(self, dataset: list):
        '''
            helper function for train - uses tokenizer to encode a text dataset
        '''
        input_ids = []
        attention_masks = []
        labels = []
        
        for masked, label in dataset:

            encoded_dict = self.tokenizer.encode_plus(
                                masked,                     
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = 128,           # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                                truncation="longest_first"
                        )
            label = self.tokenizer.encode_plus(
                            label,                     
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 128,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                            truncation="longest_first"
                        )["input_ids"]
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            labels.append(label)
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.cat(labels, dim=0)
        dataset = TensorDataset(input_ids, attention_masks, labels)
        return dataset

    
    def __str__(self) -> str:
        base = self.base_type
        tasks = ["Task: {}, Template: {}".format(task, template) for task, template in self.prompt.templates.items()]
        ret =  "======== Base Model ============\n{}\n\n======== Tasks ============\n{}\n".format(base, "\n".join(tasks))
        return ret