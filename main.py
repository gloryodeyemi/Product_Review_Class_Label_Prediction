import numpy as np
import matplotlib.pyplot as plt
from utils import preprocess, evaluate

# create an instance of the Preprocess class
preprocess = preprocess.Preprocess('data/CR')
test_data = preprocess.load_data('test_data')
print(f"Test data shape: {test_data.shape}")

# create an instance of the Evaluate class
evaluate_model = evaluate.Evaluate('models', test_data, 'results')

# evaluate the performance for BERT
bert_accuracy, bert_f1_macro, bert_f1_weighted, bert_predictions = evaluate_model.evaluate_performance("BERT")

# evaluate the performance for DistilBERT
(distilbert_accuracy, distilbert_f1_macro, distilbert_f1_weighted,
 distilbert_predictions) = evaluate_model.evaluate_performance("DistilBERT")

# evaluate the performance for RoBERTa
(roberta_accuracy, roberta_f1_macro, roberta_f1_weighted,
 roberta_predictions) = evaluate_model.evaluate_performance("RoBERTa")

# evaluate the performance for XLNet
xlnet_accuracy, xlnet_f1_macro, xlnet_f1_weighted, xlnet_predictions = evaluate_model.evaluate_performance("XLNET")

# evaluate the performance for FastText
(fasttext_accuracy, fasttext_f1_macro, fasttext_f1_weighted,
 fasttext_predictions) = evaluate_model.evaluate_performance("FastText")

# save the predictions
evaluate_model.save_predictions(bert_predictions, distilbert_predictions, xlnet_predictions,
                                roberta_predictions, fasttext_predictions)

# plot the comparison graph with values displayed on the bars
models = ['BERT', 'DistilBERT', 'XLNet', 'RoBERTa', 'FastText']
metrics = ['Accuracy', 'Macro F1', 'Weighted F1']
bert_scores = [bert_accuracy, bert_f1_macro, bert_f1_weighted]
distilbert_scores = [distilbert_accuracy, distilbert_f1_macro, distilbert_f1_weighted]
roberta_scores = [roberta_accuracy, roberta_f1_macro, roberta_f1_weighted]
xlnet_scores = [xlnet_accuracy, xlnet_f1_macro, xlnet_f1_weighted]
fasttext_scores = [fasttext_accuracy, fasttext_f1_macro, fasttext_f1_weighted]

fig, ax = plt.subplots(figsize=(12, 8))
bar_width = 0.25
bar_positions = np.arange(len(models))

# change the bar colors
colors = ['blue', 'green', 'orange']

for i, metric in enumerate(metrics):
    ax.bar(bar_positions + i * bar_width, [bert_scores[i], distilbert_scores[i], xlnet_scores[i],
                                           roberta_scores[i], fasttext_scores[i]], bar_width,
           color=colors[i],
           label=metric)

# add text annotations above each bar
for i, model in enumerate(models):
    for j, metric in enumerate(metrics):
        value = round([bert_scores[j], distilbert_scores[j], xlnet_scores[j], roberta_scores[j],
                       fasttext_scores[j]][i], 3)
        ax.text(bar_positions[i] + j * bar_width, [bert_scores[j], distilbert_scores[j], xlnet_scores[j],
                                                   roberta_scores[j], fasttext_scores[j]][i] + 0.01,
                f'{value}', ha='center', va='bottom', color='black')

ax.set_xticks(bar_positions + bar_width)
ax.set_xticklabels(models)
ax.legend(loc='lower left')
plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Model Performance Comparison')
plt.tight_layout()  # ensure tight layout to avoid overlapping text
plt.savefig('results/class_compare.png')
plt.show()
