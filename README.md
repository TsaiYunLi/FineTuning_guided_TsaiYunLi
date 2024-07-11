# FineTuning_guided_TsaiYunLi
This is a guided project, titled "Fine Tune BERT for Text Classification with TensorFlow," taught by Snehan Kekre via the Coursera platform. The data we used is the labelled Quora Insincere Questions Classification Data from Kaggle (https://www.kaggle.com/c/quora-insincere-questions-classification/data). This is an csv file with 1306122 rows and 3 columns, i.e. 1306122 Quora questions collected in a column for question id (qid), a column for the questions (question_text), and a column for the label ('1' for 'toxic' and '0' for 'sincere'). Our goal is to fine tune BERT for predicting whether a question is 'toxic' or 'sincere.' Please visit the Kaggle platform mentioned above for a detailed definition for a 'toxic' or 'sincere' question.

I have followed all the steps taught in the video lectures and built a fine-tuned a bert_en_uncased_L-12_H-768_A-12 BERT model (with 12 layers, 768 hidden units per layer, and 12 attention heads), which has 1094822 + 41 = 1094863 parameters. It is not hard to imagine how such a large language model will overfit on a rather small training data size (9795). I solved this overfitting issue by:

1. lowering the dropout rate from 0.5 to 0.4, i.e., drop = tf.keras.layers.Dropout(0.5)(pooled_output)

2. adding an AdamW optimizer with weight decay, i.e., optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=0.01) 

3. lowering the learning rate from 2e-5 to 1e-5

4. lowering the epoch number from 4 to 1

The model finally turns out to have a  loss: 0.1503, binary_accuracy: 0.9512, val_loss: 0.1451, and val_binary_accuracy: 0.9552.

I also lowered the prediction threshold from 0.5 to 0.015 for capturing nuances of the questions' toxicity. The model predicts well on new, unseen examples. Please see the last few code cells for reference.

** Feeding in more training data would potentially be a solution to this overfitting issue. Worth trying next time.
