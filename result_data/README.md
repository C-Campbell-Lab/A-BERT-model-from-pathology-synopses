# Data profile

## active_learning_result.csv

This file includes the data used to compare the effectiveness between active learning and random sampling.

Columns:

1. sizes : the training data size
1. is_active : whether it's active learning
1. labs : the validation experiment code
1. f1_scores : the micro-average f1 scores

## review_result.csv

This file includes the results from experts' review.

Columns:

1. experts : the expert code
1. iterations : the iteration number
1. labs : the experiment code
1. validation : is it against the validation set or the experts' review
1. f1_scores : the micro-average f1 scores


## dev_result.csv

This file includes the model performance metrics during development.

Columns:

1. Lab : the validation experiment code
1. Label : the 21 labels
1. F1 Score : the f1 score for a specific label
1. Precision : the precision for a specific label
1. Recall : the precision for a specific label

## upsample_result.csv

This file includes the performance comparison between with upsampling and without upsampling .

Columns:

1. f1_scores : the micro-average f1 scores
1. upsample : whether it's upsampling
1. labs : the validation experiment code

## mat_abbr.csv

This file includes label co-occurrence counts (used to plot chord diagrams).

Columns: abbrivation of labels

## unlabel_tsne.csv & label_tsne.csv

These files include the 2-D projections of the cases' 768-D embeddings.
Columns:

1. tag : the true labels
1. index : the case' index in the dataset
1. tag_num : the label count in a multi-label case
1. from : from training set or validation(test) set
1. pred_tag : the predicted labels
1. D1 : one of the 2-D projection of the 768-D embeddings
1. D2 : one of the 2-D projection of the 768-D embeddings


## kws_influence.csv

This file includes the words' influence scores for each label.

Columns:

1. influence : the influence score
1. reason : the word in a case
1. tag : the individual label
