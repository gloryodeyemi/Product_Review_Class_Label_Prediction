import time
import pandas as pd
import torch
import fasttext
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, \
    DistilBertForSequenceClassification, XLNetTokenizer, XLNetForSequenceClassification, RobertaTokenizer, \
    RobertaForSequenceClassification


class Models:
    def __init__(self, save_path, train_data, result_path):
        self.save_path = save_path
        self.train_data = train_data
        self.result_path = result_path
        self.model = None
        self.tokenizer = None

    def load_model(self, model_name):
        """
        Load the model from the transformers' library.
        :param model_name: The name of the model.
        :return: The model and tokenizer.
        """
        model_name = model_name.lower()

        if model_name == 'bert':
            self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', num_labels=2)
        elif model_name == 'distilbert':
            self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', num_labels=2)
        elif model_name == 'xlnet':
            self.model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased')
            self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', num_labels=2)
        elif model_name == 'roberta':
            self.model = RobertaForSequenceClassification.from_pretrained('roberta-base')
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', num_labels=2)
        else:
            print("Invalid model name!")
        return self.model, self.tokenizer

    def fine_tune_model(self, model_name):
        """
        Fine-tunes a language model on a new data.
        :param model_name: the name of the model.
        """
        print(f"\nTraining {model_name} model...")
        init_time = time.time()
        # load the model and tokenizer
        model, tokenizer = self.load_model(model_name)

        # tokenize and encode the sentences
        train_inputs = tokenizer(list(self.train_data['sentence']), padding=True, truncation=True,
                                 return_tensors="pt", max_length=512)
        train_labels = torch.tensor(self.train_data['label'].values)

        # create DataLoader for fine-tuning
        train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        # define optimizer and loss function
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        criterion = torch.nn.CrossEntropyLoss()

        # fine-tune the language model
        model.train()
        for epoch in range(5):
            total_loss = 0.0
            for batch in train_dataloader:
                inputs, attention_mask, labels = batch
                optimizer.zero_grad()
                outputs = model(inputs, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(train_dataloader)
            print(f"Fine-tuning Epoch {epoch + 1}/{5}, Loss: {avg_loss}")

        # save the fine-tuned model
        model.save_pretrained(f"{self.save_path}/{model_name.lower()}")
        tokenizer.save_pretrained(f"{self.save_path}/{model_name.lower()}_tokenizer")
        training_time = round(time.time() - init_time, 3)
        print(f"Training done and saved.\nTraining time: {training_time}s")
        self.model_train_time(model_name, training_time)

    def train_fasttext(self):
        """
        Train a FastText model.
        """
        # create a training data file for fasttext
        self.train_data['label'] = '__label__' + self.train_data['label'].astype(str)
        self.train_data[['label', 'sentence']].to_csv(f'{self.save_path}/fasttext_train.txt', sep=' ', index=False,
                                                      header=False)

        # train the fasttext model
        print(f"\nTraining FastText model...")
        init_time = time.time()
        model = fasttext.train_supervised(f'{self.save_path}/fasttext_train.txt')

        # save the model
        model.save_model(f"{self.save_path}/fasttext")
        training_time = round(time.time()-init_time, 3)
        print(f"Training done and saved.\nTraining time: {training_time}s")
        self.model_train_time('FastText', training_time)

    def model_train_time(self, model_type, training_time):
        """
        Saves the model training time to a csv file.
        :param model_type: the name of the model.
        :param training_time: the model training time.
        """
        # read existing CSV file into a DataFrame
        try:
            df = pd.read_csv(f'{self.result_path}/training_time.csv')
        except FileNotFoundError:
            df = pd.DataFrame(columns=['Model', 'Training_time'])

        # check if the model_name already exists in the DataFrame
        model_exists = df['Model'] == model_type

        if model_exists.any():
            # model already exists, update the training time
            df.loc[model_exists, 'Training_time'] = training_time
        else:
            # model doesn't exist, create a new DataFrame with the new row
            new_row = pd.DataFrame({'Model': [model_type], 'Training_time': [training_time]})
            # check if new_row contains any non-NA values
            if not new_row.isnull().values.all():
                df = pd.concat([df, new_row], ignore_index=True)

        # save the DataFrame back to the CSV file
        df.to_csv(f'{self.result_path}/training_time.csv', index=False)
