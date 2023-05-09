from flask import Flask, jsonify, request
import pickle
import json
from flask_cors import CORS
import fasttext.util
from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

nepali_model = fasttext.load_model('./cc.ne.70.bin')
import pickle
#Run this while loading model to make predictions
with open("./word2id_1800.pickle", "rb") as f:
    word2id = pickle.load(f)
with open("./tag2id_1800.pickle", "rb") as f:
    tag2id = pickle.load(f)
# with open("./X_1800.pickle", "rb") as f:
#     X = pickle.load(f)
# with open("./y_1800.pickle", "rb") as f:
#     y = pickle.load(f)
with open("./tags_1800.pickle", "rb") as f:
    tags = pickle.load(f)
with open("./words_1800.pickle", "rb") as f:
    words = pickle.load(f)
with open("./word_vectors_1800.pickle", "rb") as f:
    word_vectors_1800 = pickle.load(f)
with open("./tree.pickle", "rb") as f:
    tree = pickle.load(f)


app=Flask(__name__)
CORS(app)


@app.route('/postagger')
def postagger():
    # e:/flask-mini/env/Scripts/Activate.ps1
    text=request.args.get('text')
    print(text)
    pos_tagged=fun(text)
    my_data=pos_tagged
    # print(my_data,type(my_data))
    return jsonify(my_data)
    # return my_data

def fun(text):
    # ! pip install keras
    ctags =['NN','NP','PMX','PMXKM','PMXKF','PMXKO','PTN','PTNKM','PTNKF','PTNKO','PTM','PTMKM','PTMKF','PTMKO','PTH','PXH','PXR','PRF','PRFKM','PRFKF','PRFKO','PMXKX','PTNKX','PTMKX','PRFKX','DDM','DDF','DDO','DDX','DKM','DKF','DKO','DKX','DJM','DJF','DJO','DJX','DGM','DGF','DGO','DGX','QQ','JM','JF','JO','JX','JT','VI','VDM','VDF','VDO','VDX','VE','VN','VQ','VCN','VCM','VCH','VS','VR','VVMX1','VVMX2','VVTN1','VVTX2','VVYN1','VVYX2','VVTN1F','VVTM1F','VVYN1F','VVYM1F','VOMX1','VOMX2','VOTN1','VOTX2','VOYN1','VOYX2','RR','RD','RK','RJ','II','IH','IE','IA','IKM','IKF','IKO','CD','MOM','MOF','MOO','MOX','MLM','MLF','MLO','MLX','CC','CSA','CSB','YF','YM','YQ','YB','TT','FU','FF','FS','FB','FO','FZ','UU','NULL']
    pos_tag_definition = {"NN": "Common Noun","NP": "Proper Noun","PMX": "First Person Pronoun","PMXKM": "First Person Possessive Pronoun with Masculine Agreement","PMXKF": "First Person Possessive Pronoun with Feminine Agreement","PMXKO": "First Person Possessive Pronoun with Other Agreement","PTN": "Non-honorific Second Person Pronoun","PTNKM": "Non-honorific Second Person Possessive Pronoun with Masculine Agreement","PTNKF": "Non-honorific Second Person Possessive Pronoun with Feminine Agreement","PTNKO": "Non-honorific Second Person Possessive Pronoun with Other Agreement","PTM": "Medial-honorific Second Person Pronoun","PTMKM": "Medial-honorific Second Person Possessive Pronoun with Masculine Agreement","PTMKF": "Medial-honorific Second Person Possessive Pronoun with Feminine Agreement","PTMKO": "Medial-honorific Second Person Possessive Pronoun with Other Agreement","PTH": "High-honorific Second Person Pronoun","PXH": "High-honorific Unspecified-person Pronoun","PXR": "Royal-honorific Unspecified-person Pronoun","PRF": "Reflexive Pronoun","PRFKM": "Possessive Reflexive Pronoun with Masculine Agreement","PRFKF": "Possessive Reflexive Pronoun with Feminine Agreement","PRFKO": "Possessive Reflexive Pronoun with Other Agreement","PMXKX": "First Person Possessive Pronoun without Agreement","PTNKX": "Non-honorific Second Person Possessive Pronoun without Agreement","PTMKX": "Medial-honorific Second Person Possessive Pronoun without Agreement","PRFKX": "Possessive Reflexive Pronoun without Agreement","DDM": "Masculine Demonstrative Determiner","DDF": "Feminine Demonstrative Determiner","DDO": "Other-agreement Demonstrative Determiner","DDX": "Unmarked Demonstrative Determiner","DKM": "Masculine Interrogative Determiner","DKF": "Feminine Interrogative Determiner","DKO": "Other-agreement Interrogative Determiner","DKX": "Unmarked Interrogative Determiner","DJM": "Masculine Relative Determiner","DJF": "Feminine Relative Determiner","DJO": "Other-agreement Relative Determiner","DJX": "Unmarked Relative Determiner","DGM": "Masculine General Determiner-pronoun","DGF": "Feminine General Determiner-pronoun","DGO": "Other-agreement General Determiner-pronoun","DGX": "Unmarked General Determiner-pronoun","QQ": "Question Marker","JM": "Masculine Adjective","JF": "Feminine Adjective","JO": "Other-agreement Adjective","JX": "Unmarked Adjective","JT": "Sanskrit-derived Comparative or Superlative Adjective",'VI': 'Infinitive Verb','VDM': 'Masculine d-participle Verb','VDF': 'Feminine d-participle Verb','VDO': 'Other-agreement d-participle Verb','VDX': 'Unmarked d-participle Verb','VE': 'e(ko)-participle Verb','VN': 'ne-participle Verb','VQ': 'Sequential Participle-converb','VCN': 'Command-form Verb, Non-honorific','VCM': 'Command-form Verb, Mid-honorific','VCH': 'Command-form Verb, High-honorific','VS': 'Subjunctive/Conditional e-form Verb','VR': 'i-form Verb','VVMX1': 'First Person Singular Verb','VVMX2': 'First Person Plural Verb','VVTN1': 'Second Person Non-honorific Singular Verb','VVTX2': 'Second Person Plural(or Medial-honorific Singular) Verb','VVYN1': 'Third Person Non-honorific Singular Verb','VVYX2': 'Third Person Plural(or Medial-honorific Singular) Verb','VVTN1F': 'Feminine Second Person Non-honorific Singular Verb','VVTM1F': 'Feminine Second Person Medial-honorific Singular Verb','VVYN1F': 'Feminine Third Person Non-honorific Singular Verb','VVYM1F': 'Feminine Third Person Medial-honorific Singular Verb','VOMX1': 'First Person Singular Optative Verb','VOMX2': 'First Person Plural Optative Verb','VOTN1': 'Second Person Non-honorific Singular Optative Verb','VOTX2': 'Second Person Plural(or Medial-honorific Singular) Optative Verb','VOYN1': 'Third Person Non-honorific Singular Optative Verb','VOYX2': 'Third Person Plural(or Medial-honorific Singular) Optative Verb','RR': 'Adverb','RD': 'Demonstrative Adverb','RK': 'Interrogative Adverb','RJ': 'Relative Adverb','II': 'Postposition','IH': 'Plural-collective Postposition','IE': 'Ergative-instrumental Postposition','IA': 'Accusative-dative Postposition','IKM': 'Masculine Genitive Postposition','IKF': 'Feminine Genitive Postposition','IKO': 'Other-agreement Genitive','CD': 'Cardinal Number','MOM': 'Masculine Ordinal Number','MOF': 'Feminine Ordinal Number','MOO': 'Other-agreement Ordinal Number','MOX': 'Unmarked Ordinal Number','MLM': 'Masculine Numeral Classifier','MLF': 'Feminine Numeral Classifier','MLO': 'Other-agreement Numeral Classifier','MLX': 'Unmarked Numeral Classifier','CC': 'Coordinating Conjunction','CSA': 'Subordinating Conjunction appearing after the clause it subordinates','CSB': 'Subordinating Conjunction appearing before the clause it subordinates','YF': 'Sentence-final Punctuation','YM': 'Sentence-medial Punctuation','YQ': 'Quotation Marks','YB': 'Brackets','TT': 'Particle','UU': 'Interjection','FF': 'Foreign Word in Devnagari','FS': 'Foreign Word not in Devnagari','FB': 'Abbreviation','FO': 'Mathematical Formula','FZ': 'Letter of the Alphabet','Unclassifiable': 'FU','NULL':'NULL TAG'}


# Load the saved model
    model = load_model('./nepali_pos_tagger_new_1800_sents_notred.h5')



# Define the sentence to be tested
    test_sentence = text

    t = test_sentence.split()
    tmp=[]
    act_words=[]
    
    for word in t:
        try:
            tmp.append(word2id[word])
            
        except:
            i = tree.query(nepali_model.get_word_vector(word))
            tmp.append(i[1])
        act_words.append(word)
        
   
    for i in range(len(tmp)+1,51):
        tmp.append(len(words)-1)
    
    
    resultss=""
# Make the prediction
    predicted_probs = model.predict(np.array([tmp]))
    predicted_probs = np.argmax(predicted_probs, axis=-1) # Map softmax back to a POS index
    for w, pred in zip(act_words, predicted_probs[0]): # for every word in the sentence
        if(tags[pred] == '.'):
            break
        resultss+="{:20} -- {}({})".format(w, tags[pred],pos_tag_definition[tags[pred]])+'\n'
        print("{:20} -- {}({})".format(w, tags[pred],pos_tag_definition[tags[pred]])) # Print word and tag
    resultss+='\n-----------------------------------------------------------------------\n'
    print('\n-----------------------------------------------------------------------\n')
    return resultss




if __name__=="__main__":
    app.run(debug=True)