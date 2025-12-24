# HexSoftwares Internship - Week 1
## Sentiment Analysis: Customer Feedback

In this project, two datasets are combined to create a larger and more diverse training set, enhancing the model’s accuracy compared to using a single dataset.
Several preprocessing techniques are used to improve the performance of NLP models and ensuring the data is clean and suitable for training. Some of these techniques include:
- Removing unwanted characters and special symbols
- Eliminating stop words (commonly used words like "and," "the," etc.)
- Applying stemming and lemmatization to reduce words to their root forms
- Encoding textual data
- Padding sequences to ensure uniform input lengths.

The models used for classification are:
- The **Naive Bayes Classifier (NBC)**: Chosen due to the relatively small size of the dataset.
- The **Recurrent Neural Network (RNN)**: Selected for its practicality and scalability, especially with more complex datasets.

The results of both methods are then compared to evaluate their performance.

## Datasets

For training: Both models use the same dataset, each in its required format.
- **NBC**: Full preprocessed text (no truncation).
- **RNN**: Padded to `max_length`.

For training: Both models see the same test samples, but each uses its natural input format to give a fair and realistic comparison.
- **NBC**: Full preprocessed text (no truncation).
- **RNN**: Truncated/padded to the same `max_length` as training.

### Sources for training dataset:
- [given dataset](https://drive.google.com/file/d/1_SHjQJVxZdr_LW2aIHAiOSBPWWGWd7Bs/view?usp=drive_link)
- [open source dataset](https://www.kaggle.com/datasets/kundanbedmutha/customer-sentiment-dataset)
### Sources for testing dataset:
- [open source dataset](https://www.kaggle.com/datasets/ahmedabdulhamid/reviews-dataset)
