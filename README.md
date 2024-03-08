POS(Parts of Speech) tagging is a critical task in NLP(Natural Language Processing) that involves identifying and labeling words in a sentence according to their grammatical roles.
This project is focused on developing a Bi-LSTM(Bidirectional Long Short Term Memory) based POS tagging system for the Nepali language using Python, ReactJS, and Flask.
The model was trained on a corpus comprising of Nepali text from books, articles and web-text, annotated with a tag-set of 112 unique tags.
The trained model achieved an accuracy of 98.6% and performed well on compound words.
The developed POS tagging system can be used to develop various NLP applications in such as translation, text-to-speech , sentiment analysis and more.
For example, in the sentences:
“बाजेलाई नाम भन्नुहोस्”, the word "बाजे" would be labelled as "noun", the word “लाई”
  as a “ postposition” the word " नाम " as  a “noun”,  and  " भन्नुहोस् "  as a " verb " .
“The cat is sleeping on the couch” , the words "cat" and "couch" would be labeled as nouns, "is" as a verb, and "sleeping" as an adjective.

![image](https://github.com/rishabhprasad255/flaskmajor2023/assets/74315210/2ffe7cd1-720c-4688-9b87-c4fdcfb19eed)


IMPLEMENTATION:
A large corpus of Nepali text was retrieved from Nepali National Corpus[4] comprising of approximately 14 million words extracted from books, webtext and news articles in XML format. This corpus is tagged using the Nelralec tagset[5]. 
20% of the corpus was used as our dataset, to train, validate and test the model.
Extracted the data from the  XML files and stored it as a CSV file.
Read the CSV file into a pandas dataframe; removed rows containing erroneous tags.
Tokenized the ‘word’ column of the dataframe and one hot encoded the data in the ‘ctag’ column of the dataframe.
Extracted sentences from the CSV file by splitting at the end of sentence markers(| ,|| , !, ?)  and stored in an array. Each sentence was stored as a sub array where each word was stored in a tuple along with its tag as shown in the below example:
[('यसै', 'DDX'), ('मा', 'II'), ('नै', 'TT'), ('पहिलो', 'MOM'), ('चरण', 'NN'), ('को', 'IKM'), ('वार्ता', 'NN’),    ('समाप्त', 'JX'), ('भयो', 'VVYN1'), ('।', 'YF’)]

Pad the extracted sentences with Dummy values upto 50 words to ensure all inputs to the model are of fixed length.
Split the processed data into training(120981 sentences) , testing(33306 sentences) and validation sets( 13442 sentences).
Define Keras, a Sequential model with an input layer, an embedding layer followed by a dropout layer,  bidirectional LSTM layer and a dense layer as the output layer.
Trained the model on 120981 sentences and achieved training accuracy of 98.61% and validation accuracy of 98.59%.
Tested the trained model on 33306 sentences and achieved testing accuracy of 98.6%,  precision of 98.6%, recall of 98.5% and f1 score of 98.5% .
Designed a user interface using ReactJS.
Designed an API to connect the backend(model and logic) with the frontend(User Interface).
Handled unknown words and misspelt words: 
by comparing the Fasttext Embedding[10] of the word with words already in the model’s vocabulary and replaced the problem word with the closest match.
Tested the model on some manually entered sentences.

![image](https://github.com/rishabhprasad255/flaskmajor2023/assets/74315210/8f952c9d-4d5f-4ec4-99c9-30c02ca70fc0)
![image](https://github.com/rishabhprasad255/flaskmajor2023/assets/74315210/be5eae9a-9ff1-4805-bd2b-ebf448d7228d)
![image](https://github.com/rishabhprasad255/flaskmajor2023/assets/74315210/4e8a3566-a3c4-4ab3-a38b-18789e626127)
![image](https://github.com/rishabhprasad255/flaskmajor2023/assets/74315210/4a3cdf37-00a1-4d72-b541-461350316653)
![image](https://github.com/rishabhprasad255/flaskmajor2023/assets/74315210/c0e555f5-f871-43c1-8686-705c0e72a7d7)

LIMITATIONS:
The model is unable to make prediction on sentences containing non-Devanagari characters other than punctuation marks and numbers. Hence, for such sentences the non-Devanagari characters will have to be manually removed by the user before submitting it for tagging.
Moreover, the input text is required to be separated by whitespace after each token (word/punctuation marks) before submitting for tagging. If the input text is lacking a whitespace between tokens, then they will bet considered to be part of the same token. 
The model does not perform well for ambiguous words i.e., words that can be tagged differently in different contexts. 
For Example: 'साचो' means 'key' as well as 'truth'.
The model can only handle sentences having 50 or less words in it.




