from utils import models, preprocess

# initialize a Preprocess class
data_preprocess = preprocess.Preprocess("data/CR")

# preprocess the dataset to get train and test data
# data_preprocess.preprocess_data('custrev.pos', 'custrev.neg')

# load the train data
train_data = data_preprocess.load_data('train_data')
print(f"Train data shape: {train_data.shape}")

# initialize a Models class
models_object = models.Models("models", train_data, "results")

# fine-tune the BERT model
models_object.fine_tune_model("BERT")

# fine-tune the DistilBERT model
models_object.fine_tune_model("DistilBERT")

# fine-tune the RoBERTa model
models_object.fine_tune_model("RoBERTa")

# fine-tune the XLNet model
models_object.fine_tune_model("XLNet")

# train the FastText model
models_object.train_fasttext()
