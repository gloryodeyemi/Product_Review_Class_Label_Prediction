import pandas as pd
from sklearn.model_selection import train_test_split


class Preprocess:
    def __init__(self, data_path):
        self.data_path = data_path

    def preprocess_data(self, positive_file, negative_file):
        """
        Add a class label to the positive and negative files, combine and split them into train and test data.
        :param positive_file: Name of the positive file.
        :param negative_file: Name of the negative file.
        """
        # load positive and negative datasets
        positive_data = pd.read_csv(f"{self.data_path}/{positive_file}", sep='\t', header=None, names=['sentence'])
        negative_data = pd.read_csv(f"{self.data_path}/{negative_file}", sep='\t', header=None, names=['sentence'])

        # assign labels (1 for positive, 0 for negative)
        positive_data['label'] = 1
        negative_data['label'] = 0

        # combine the datasets
        combined_data = pd.concat([positive_data, negative_data], ignore_index=True)

        # split the data into train and test sets
        train_data, test_data = train_test_split(combined_data, test_size=0.2, random_state=42)

        # save the train and test data for later use
        test_data.to_csv(f"{self.data_path}/test_data.csv", index=False)
        train_data.to_csv(f"{self.data_path}/train_data.csv", index=False)

    def load_data(self, data_name):
        """
        Load and return a dataset.
        :param data_name: The name of the dataset.
        :return: The loaded dataset.
        """
        data = pd.read_csv(f"{self.data_path}/{data_name}.csv")
        return data
