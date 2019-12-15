# TweetsByDonald
This project uses sentiment classification to automatically classify tweets as positive or negative.
This classification uses Naive Bayes, trained on a dataset called Sentiment140, which can be downloaded
from kaggle.com. The model can then be used to classify tweets written by Donald Trump. These classified tweets can then
be used to train a model to generate new tweets in the same Trump style, with a positive or negative sentiment.

## Splitting data
To split the dataset Sentiment140, you can use the file split_data.py.
Simply run the file with the argument --file (-f), inputing the path to the file containing the dataset.

Example:

```
python split_data.py -f Data/training_data.csv
```

The script then saves two files, one containing the training data (80 %) and one containing the evaluation data (20 %).
Given the input in the example above, these two files are saved in the same location with
in the format as given below.

```
Data/training_data_t.csv
Data/training_data_v.csv
```

## Training
Training naive Bayes for classification and classifying new tweets are done frome the main.py python file.
To train the model, simply run the file with the argument --train (-t), inputing the datafile (.csv) from which to train the model.
If you wish to save the trained parameters, use the argument --destination (-d), inputing the location for the saved file
(we use the module pickles to save data, so use the extension .p).

Example:

```
python main.py -t Data/training_data_t.csv -d params.p
```

Training naive Bayes with the dataset Sentiment140 usually takes a long time, so it is prefered to train the data once and save
to a pickled file, which can then be loaded very quickly.

## Evaluation
To then load the data, run main.py with the argument --load (-l), inputing the path from which file to load the parameters.
To then evaluate the trained naive Bayes, use the argument --validate (-v), inputing the evaluation dataset.

Example:

```
python main.py -l params.p -d Data/training_data_v.csv
```

## Classification
You can also use the file main.py to classify new tweets. The tweets you want to classify has to follow the following format:

```
text§created_at§is_retweet§id_str
```

Note that this represents 4 columns separated by a section sign.

Simply run the file with the argument --classify (-c), inputing the file containing the tweets you want to classify.

Example:
```
python main.py -c Data/tweet_data.txt
```

The script then saves two files, one containing positive classified tweets, and one with negative classified tweets.
Given the input from the example, the script saves the two files in the same location in the following format:
```
Data/tweet_data_negative.txt
Data/tweet_data_positive.txt
```

## Bigram Trainer
This project also contains a bigram trainer, that you can train with the files containing classified tweets.
To run the bigram trainer, simply run the file with the arguments  --train (-t), inputing the path to the file containing training tweets,
and the argument --destination, inputing the path to where you want to save the bigram probabilities.

Example:
```
python bigram_trainer.py -t Data/tweet_data_negative.txt -d trump_model_negative.txt
```

## Generate
Finally, you can use this bigram model to generate new tweets with a given sentiment.
Simply run the file main.py with the argument --generate (-g), inputing the model with which to generate new tweets.

Example:
```
python main.py -g trump_model_negative.txt
```

This automatically generates 3 new tweets and prints them to the terminal.