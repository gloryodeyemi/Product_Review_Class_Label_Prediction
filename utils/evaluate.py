from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
import fasttext
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, \
    DistilBertForSequenceClassification, XLNetTokenizer, XLNetForSequenceClassification, RobertaTokenizer, \
    RobertaForSequenceClassification


class Evaluate:
    def __init__(self, model_path, test_data, result_path):
        self.test_data = test_data
        self.model_path = model_path
        self.result_path = result_path

    def load_saved_model(self, model_name):
        """
        Load the saved model from the model directory.
        :param model_name: The name of the model.
        :return: The saved model and tokenizer.
        """
        print(f"Loading saved {model_name} model...")
        model = None
        tokenizer = None
        model_name = model_name.lower()

        if model_name == 'bert':
            model = BertForSequenceClassification.from_pretrained(f"{self.model_path}/{model_name}")
            tokenizer = BertTokenizer.from_pretrained(f"{self.model_path}/{model_name}_tokenizer")
        elif model_name == 'distilbert':
            model = DistilBertForSequenceClassification.from_pretrained(f"{self.model_path}/{model_name}")
            tokenizer = DistilBertTokenizer.from_pretrained(f"{self.model_path}/{model_name}_tokenizer")
        elif model_name == 'xlnet':
            model = XLNetForSequenceClassification.from_pretrained(f"{self.model_path}/{model_name}")
            tokenizer = XLNetTokenizer.from_pretrained(f"{self.model_path}/{model_name}_tokenizer")
        elif model_name == 'roberta':
            model = RobertaForSequenceClassification.from_pretrained(f"{self.model_path}/{model_name}")
            tokenizer = RobertaTokenizer.from_pretrained(f"{self.model_path}/{model_name}_tokenizer")
        else:
            print("Invalid model name!")
        print("Loading done!\n")
        return model, tokenizer

    def evaluate_fasttext(self):
        print("FastText model making predictions...")
        # load FastText model
        model = fasttext.load_model(f'{self.model_path}/fasttext')

        # predict labels for the test data
        self.test_data['label'] = '__label__' + self.test_data['label'].astype(str)
        self.test_data[['label', 'sentence']].to_csv('fasttext_test.txt', sep=' ', index=False, header=False)
        predictions = model.predict(list(self.test_data['sentence']))

        # extract predicted labels
        predicted_labels = [int(label[0].replace('__label__', '')) for label in predictions[0]]
        print("Prediction done!\n")

        return predicted_labels

    def tokenize_and_encode(self, model_name):
        model, tokenizer = self.load_saved_model(model_name)
        print(f"{model_name} model tokenizing and encoding...")
        test_inputs = tokenizer(list(self.test_data['sentence']), padding=True, truncation=True, return_tensors="pt",
                                max_length=512)
        test_labels = torch.tensor(self.test_data['label'].values)
        test_dataset = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], test_labels)
        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        print("Tokenizing and encoding done!\n")
        return test_dataloader, model

    def evaluate_language_model(self, model_name):
        test_dataloader, model = self.tokenize_and_encode(model_name)
        print(f"{model_name} model making predictions...")

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_dataloader:
                inputs, attention_mask, labels = batch
                outputs = model(inputs, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        print("Prediction done!\n")
        return all_preds, all_labels

    def evaluate_performance(self, model_name):
        if model_name == 'FastText':
            labels = self.test_data['label'].values
            predicted_labels = self.evaluate_fasttext()
            # Convert predicted labels to integers
            predictions = [int(label) for label in predicted_labels]
        else:
            labels, predictions = self.evaluate_language_model(model_name)
        print(f"Evaluating {model_name} performance...")
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        conf_matrix = confusion_matrix(labels, predictions)

        print(f"{model_name} Model:\n----------------")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro Average F1: {f1_macro:.4f}")
        print(f"Weighted Average F1: {f1_weighted:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")

        print("Performance evaluation done!\n")
        return accuracy, f1_macro, f1_weighted, predictions

    def save_predictions(self, bert_predictions, distilbert_predictions, xlnet_predictions, roberta_predictions,
                         fasttext_predictions):
        print("Saving models predictions...")
        # remove the '__label__' prefix and convert labels back to original format
        original_labels = [label.replace('__label__', '') for label in self.test_data['label']]

        # Save the predictions to CSV files
        results_df = pd.DataFrame({
            'Sentence': self.test_data['sentence'],
            'BERT': bert_predictions,
            'DistilBERT': distilbert_predictions,
            'XLNet': xlnet_predictions,
            'RoBERTa': roberta_predictions,
            'FastText': fasttext_predictions,
            'Truth_label': original_labels
        })
        results_df.to_csv(f'{self.result_path}/model_predictions.csv', index=False)
        print("Saving predictions done!\n")
