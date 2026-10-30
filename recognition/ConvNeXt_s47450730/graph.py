# This file shows how graphs and confusion matrices were made to plot results

# during after training, statistics on loss, training/validation accuracy and predictions were 
# pickle dumped to data.pickle


import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# unpickle the data
with open('data.pickle', 'rb') as file:
    (epochs, losses, test_accuracies, train_accuracies, actual_labels, predicted_probs) = pickle.load(file)

# convert accuracies to percentage
test_accuracies = [x * 100 for x in test_accuracies]
train_accuracies = [x * 100 for x in train_accuracies]
length = len(test_accuracies)

# not every epoch we got to since training is stopped through a timer
epochs = epochs[:length] # truncate so same length as accuracies for plotting

# plot the data
plt.plot(epochs, test_accuracies, label='Validation Accuracy')
plt.plot(epochs, train_accuracies, label='Train Accuracy')


plt.xlabel('Epochs')
plt.ylabel('Accuracy %')
plt.title('Validation Accuracy for ConvNeXt throughout training')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_plot.png', dpi=300, bbox_inches='tight')
plt.close() # close so we can plot the confusion matrix

# Our labels are AD = 1, NC = 0 - therefore process these predictions as such
predicted = (np.array(predicted_probs) >= 0.5).astype(int)

cm = confusion_matrix(actual_labels, predicted, labels=[0, 1])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["NC", "AD"]
)
disp.plot(cmap="Blues")
plt.savefig('accuracy_plot.png', dpi=300, bbox_inches='tight')
