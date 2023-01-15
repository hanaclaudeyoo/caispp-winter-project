Claude (Hana) Yoo
hanayoo@usc.edu


Winter Project: News Headlines for Sarcasm Detection


In this project, I built an NLP binary classification model to distinguish satirical and non-satirical news articles by their headlines. An example of a satirical article title would be, “boehner just wants wife to listen, not come up with alternative debt-reduction ideas”, while an example of a non-satirical one would be, “j.k. rowling wishes snape happy birthday in the most magical way”. It is not immediately obvious whether either article is satirical or not, so an artificial intelligence model could help find the difference.


Dataset
The dataset used is “News Headlines Dataset for Sarcasm Detection” from Kaggle. It has 26709 samples in total, with each sample consisting of a link to an article, its headline in lowercase, and a boolean indicating whether it is satirical or not. The satirical articles are taken from The Onion and the non-satirical articles are taken from HuffPost.
From this data, we extracted the headline to be our input and the satirical boolean to be our output, since the link was not conducive to the task of classifying articles by their titles. Each headline was tokenized into words and word-parts, padded to the length of the longest headline, and a corresponding attention mask was built. This preprocessing sets up the data for use in training a transformer architecture. Finally, as the dataset was too large to feasibly train with my current resources and timeframe, I reduced the portion of data used to a random 20% of the original Kaggle dataset. Also, other transfer learning models built with BERT seem to work well on small datasets, such as around 8000.


Model Development and Training
I chose to use transfer learning on the pretrained BERT(Bidirectional Encoder Representations from Transformers) model, since BERT is established as a very effective NLP model. By using BertForSequenceClassification, I added a single linear layer on top of BERT for classification. AdamW, a popular optimizer that performs weight decay more selectively, was used. 90% of the data was used in the forward training step, while the remaining 10% was used in the backward validation step.
A batch size of 16 and 3 epochs were used, as recommended by the original BERT paper. Other papers recommended a low learning rate, so 1e-4 was used.


Model Evaluation/Results
The model produced a 0.77 validation accuracy for all three epochs.
A model using RoBERTa instead of BERT was also tried, but it produced a lower accuracy of 0.653 for all three epochs. A model using 50% of the original Kaggle dataset was also tried, but it produced a lower accuracy of 0.746.


Discussion
1. How well does your dataset, model architecture, training procedures, and chosen metrics fit the task at hand?
The feature space of the dataset should be sufficient, since the goal of the model is to classify articles from just the article headline. The number of samples should also be sufficient since increasing the portion of the dataset used did not improve the model’s accuracy. However, one point the dataset could be better is if it had a greater variation of sources, such as including The Guardian for non-satirical articles and Reductress for satirical articles, for greater model flexibility.
I think that BERT is an appropriate model architecture for this task, since it is a simple NLP classification. Since it is a simple binary classification task, accuracy is an appropriate metric to evaluate the model’s performance.
2. Can your efforts be extended to wider implications, or contribute to social good? Are there any limitations in your methods that should be considered before doing so?
A model that can identify satirical news articles can help fight misinformation. If used right, it can assist in building media literacy. However, one thing that should be considered is the dataset bias. If the dataset was biased so that satire becomes associated with characteristics that are not necessarily mutually inclusive, such as liberalism or gender, it can lead to faulty assumptions and perpetuation of stereotypes.
3. If you were to continue this project, what would be your next steps?
Other BERT-based models produce >90% accuracy, so there is a lot of room for my model to improve. The fact that increasing the dataset size does not improve model performance probably indicates that it is overfitting. Additionally, since the model’s accuracy does not improve over each epoch, there is likely a vanishing gradient problem.
In order to tackle these issues, in the future I would try a dynamic learning rate, as an article concludes: “learning rate is linearly increased for the first 10% of steps and linearly decayed to zero afterward”.


Sources
* https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270 
* https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/bert#transformers.BertForSequenceClassification 
* https://arxiv.org/pdf/1810.04805.pdf
* https://towardsdatascience.com/why-adamw-matters-736223f31b5d
* https://medium.com/distributed-computing-with-ray/hyperparameter-optimization-for-transformers-a-guide-c4e32c6c989b 
* https://arxiv.org/abs/2006.04884