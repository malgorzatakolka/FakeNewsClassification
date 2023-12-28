# Classification of Fake and True News <a class="jp-toc-ignore"></a>

The purpose of this notebook is to assess the veracity of news articles—determining whether they align with factual information. The analysis is conducted using a dataset available at https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset. Unfortunately, the dataset lacks a detailed description of its origin. While a forum user suggests that the data may be scraped from https://www.politifact.com/, a non-profit organization dedicated to combating the spread of fake news, this origin is unverified by other sources. Notably, Politifact utilizes a truth-o-meter where independent journalists evaluate the factual accuracy of news.

Confirmation of the dataset's extraction from Politifact would greatly enhance its utility.

In analyzing the true.csv and fake.csv files, we identified patterns that strongly predict the truthfulness of news articles. Consequently, the approach involved eliminating features that might inadvertently leak information, aiming for more universally applicable predictions.

In the notebook the main metric for evaluating classification that was used is weighted f1_score.

## Running the code
To run the code download the data from kaggle (Fake.csv and True.csv) to your drive into folder /content/drive/MyDrive/Colab/FakeNews/.
Run the notebook. If the stored checkpoints with best weights differ from the one stored in the notebook please replace the URL of the checpoints with your own, best performing one on the validation set.

## Structure of the notebook
The notebook is devided into three main parts:
* data familiarization and cleaning
* modeling with classic machine learning models from sklearn library
* modeling with the use of huggingface implementation of pretrained distilbert-cased

## Main conclusions
* The models developed in this analysis heavily rely on the writing style of the author, capturing specific nuances such as certain words, word length, language sophistication, and punctuation. While these models demonstrate impressive performance, a skilled columnist could potentially create fake news that the models would inaccurately classify as true.
* The models vary significantly in size, with the RandomForestClassifier pipeline being only 240KB, while the DistilBERT model weighs 750MB. This size difference is crucial when considering deployment efficiency, and it raises the question of whether a slight drop in performance (e.g., 10%) could be justified to optimize deployment resources.
* To establish which of the models should be deployed we would need to know what risks for a certain business is associated with releasing fake news and how much it costs.
* Preventing the spread of misinformation could have potential positive effect in the feature as this prevents from the use of corrupted data by AI models.
* Although the deployment of better performing model could be costly the business may decide to do it as a means of integrity and building highly ethical business profile.
* The DistilBERT-based model, trained solely on news titles, exhibits the highest accuracy among the models considered. Most likely the model picked up on 'catchy', emotional appealing and less sophisticated headers.
* The uncertainty surrounding the data's collection methodology introduces a potential limitation. Without clarity on how the data was gathered, there is no guarantee that the model would accurately predict false news from other sources. This aspect needs verification to ensure the model's robustness beyond the current dataset and its potential biases.

