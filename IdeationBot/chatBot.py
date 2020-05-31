import pyttsx3
import speech_recognition as sr
import os
import gensim
import gensim.downloader as api
from gensim.models import KeyedVectors
import random
import nltk
import re
from scipy import spatial
import numpy as np
import simpleaudio as sa
from statistics import mean
import pyemd
from pyemd import emd

#Set working directory 
#BASE_PATH = os.chdir('/Users/lineelgaard/Documents/Human Computer Interaction/Exam_DivergentThinkingInHCI/botScript')

#Define semantic model - only a part of the pretrained vectors (to reduce RAM requirements)
semantic_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=10 ** 5)


# Initializing microphone and speech recognition system
mic = sr.Microphone()

# Using the simplest voice engine we can to do text to speech
engine = pyttsx3.init()
voices = engine.getProperty('voices') # if you get a weird accent, try printing this variable and choose another voice
engine.setProperty('voice', voices[0].id) # a different voice id might be necessary on your system


def build_composite_semantic_vector(word_seq,highDimModel):
    """
    Function for producing vocablist and model is called in the main loop
    If an input word (part of word_seq) contains a space (is a sentence with multiple words), then 
    a composite score is first calculated for this sentence before being added to the overall 
    composite score. 
    """
    ## build composite vector
    getComposite = [0] * 300
    for w1 in word_seq:
        if ' ' in w1:
            newList = w1.split(' ')
            sentenceVector = [0]*300
            for w2 in newList:
                if w2 in highDimModel.vocab: 
                    vector = highDimModel[w2]
                    sentenceVector = sentenceVector + vector
                    getComposite = getComposite + sentenceVector
        else:
            if w1 in highDimModel.vocab:
                semvector = highDimModel[w1]
                getComposite = getComposite + semvector 
    return getComposite


def makeSound():
    frequency = 440  # Our played note will be 440 Hz
    fs = 44100  # 44100 samples per second
    seconds = 1  # Note duration of 3 seconds

    # Generate array with seconds*sample_rate steps, ranging between 0 and seconds
    t = np.linspace(0, seconds, seconds * fs, False)

    # Generate a 440 Hz sine wave
    note = np.sin(frequency * t * 2 * np.pi)

    # Ensure that highest value is in 16-bit range
    audio = note * (2**15 - 1) / np.max(np.abs(note))
    # Convert to 16-bit data
    audio = audio.astype(np.int16)
    return(audio)

def getInput():
    #engine.say("ready")
    #engine.runAndWait()
    audio = makeSound()
    play_obj = sa.play_buffer(audio, 1, 2, 44100)
    play_obj.wait_done()
    with mic as source:
        audio = r.listen(source)
    print("processing")

    try:
        textInput = r.recognize_google(audio)
        print("you said: " + textInput)
    
    # API was unreachable or unresponsive:
    except sr.RequestError:
        #response = "API unavailable"
        textInput = "none"
    # if the speech was unintelligible:
    except sr.UnknownValueError:
        #response = "Sorry! I did not understand that word. "
        textInput = "none" 
    
    #engine.say(response)
    #engine.runAndWait()
    return(textInput)


def getAssociativeCurve(word_seq):
    vectors = getVectors(word_seq)
    distance = [spatial.distance.cosine(targetVector,y) for i,y in enumerate(vectors)]
    meanDistance = mean(distance)
    if meanDistance < 0.8: 
        return("steep")
    else:
        return("flat")

def getVectors(word_seq):
    vectors = [0] * len(word_seq)
    for n in range(len(word_seq)): 
        if ' ' in word_seq[n]:
            vectors[n] = build_composite_semantic_vector(word_seq[n], semantic_model) 
        else:
            if word_seq[n] in  semantic_model.vocab:
                vectors[n] = semantic_model.get_vector(word_seq[n])
    return(vectors)

def getResponseOneInput(inputWord):
    if (inputWord in semantic_model.wv.index2entity):
        #target = []
        #output = []
        #for w in words:
        stop = False
        #print(w)
        #target.append(w)
        regex = re.compile('[@_!#$%^&*()<>?/\|}{~:]')
        while stop == False:
            newWord = random.choice(semantic_model.wv.index2entity)
            if(regex.search(newWord) == None):
                if not newWord[0].isupper():
                    similarity = semantic_model.similarity(inputWord, newWord)
                    distance = 1-similarity
                    if (distance > 0.4 and distance < 0.7):
                        token = nltk.word_tokenize(newWord)
                        POStag = nltk.pos_tag(token)
                        if (POStag[0][1] == "NN" or POStag[0][1]=="VB"):
                            stop=True
                            #output.append(newWord)  
                            return(newWord)
    else: 
        return("I cannot find that word")

def getResponseCompositeInput(word_seq):
#Return response word based on a sequence of inputs 
    stop = False

    regex = re.compile('[@_!#$%^&*()<>?/\|}{~:]')
    while stop == False:
        targetVector = build_composite_semantic_vector(word_seq, semantic_model)
        newWord = random.choice(semantic_model.wv.index2entity)
        if newWord in semantic_model.vocab:
            if(regex.search(newWord) == None):
                newVector = semantic_model.get_vector(newWord)
                distance = spatial.distance.cosine(targetVector, newVector)
                if (distance > 0.4 and distance < 0.7):
                    token = nltk.word_tokenize(newWord)
                    POStag = nltk.pos_tag(token)
                    if (POStag[0][1] == "NN" or POStag[0][1]=="VB"):
                        stop=True
                        return(newWord)


def objectUseCombinationStrategy(word_seq):
    stop = False
    regex = re.compile('[@_!#$%^&*()<>?/\|}{~:]')
    response = [0] * 2
    response[0] = random.choice(word_seq)
    while stop == False:
        targetVector = build_composite_semantic_vector(word_seq, semantic_model)
        newWord = random.choice(semantic_model.wv.index2entity)
        if newWord in semantic_model.vocab:
            if(regex.search(newWord) == None):
                newVector = semantic_model.get_vector(newWord)
                distance = spatial.distance.cosine(targetVector, newVector)
                if (distance > 0.4 and distance < 0.7):
                    token = nltk.word_tokenize(newWord)
                    POStag = nltk.pos_tag(token)
                    if (POStag[0][1] == "NN" or POStag[0][1]=="VB"):
                        stop=True
                        response[1] = newWord
                        return(response)

def broadUseCategoryStrategy(word_seq):
    #stop = False
    regex = re.compile('[@!#$%^&*()<>?/\|}{~:]')
    vectors = getVectors(word_seq)
    similarWords = semantic_model.most_similar(positive=vectors, negative=[targetWord],topn = 200)
    similarWords = [w for w in similarWords if w[0] not in word_seq]
    #while stop == False:
    words = []
    for w in similarWords:
        word = w[0]
        if len(word) > 3:
            if(regex.search(word) == None):
                if not word[0].isupper():
                    token = nltk.word_tokenize(word)
                    POStag = nltk.pos_tag(token)
                    if (POStag[0][1] == "NN" or POStag[0][1]=="VB"):
                        stop=True
                        words += [word]                      
    return(words[0:5])
    #Returns a list of words 

def propertyUseStrategy(targetWord):
    request = 'Name one part of the '+targetWord
    engine.say(request)
    engine.runAndWait()

    textInput = getInput()
    vectors = getVectors(textInput)

    newWord = semantic_model.most_similar(positive=vectors, negative=[targetWord])
    newWord = random.choice(newWord)
    newWord = newWord[0]

    response = textInput + ' is related to ' + newWord

    return(response)


############################################################################
########################### RUN CHATBOT ####################################
############################################################################


targetWord = 'book'
targetVector = semantic_model.get_vector(targetWord)
textInput = "nothing" # we initialise the input for the loop to work
rounds = 0

Introduction = 'Hi. I am ready to assist you in the Alternative Uses task.  ' + '  The target object is            ' + targetWord 
# Introduction: let Paul introduce himself
engine.say(Introduction)
engine.runAndWait()

engine.say('         Say stop when you want the task to end.                  Wait for the tone to say an idea you have.           The task will start now')
engine.runAndWait()

# All interactions happen here
while textInput != "stop":
    # We load the speech to text model
    r = sr.Recognizer()
    # check for background noise and adjust sensitivity accordingly
    with mic as source:
        audio = r.adjust_for_ambient_noise(source)

    # Listen for user input
    #engine.say("I am listening now")
    #engine.runAndWait()

    batch_size = 5
    word_sequence = [0] * batch_size

    for n in range(batch_size):
        word = getInput()
        if word=="stop":
            textInput="stop"
            break
        else:
            word_sequence[n] = word
    
    if textInput == "stop":
        response = "Good bye"

    else:
        word_sequence = [w for w in word_sequence if w != "none"] 
        print(word_sequence)
        if len(word_sequence) > 1:
            assoCurve = getAssociativeCurve(word_sequence)
            if assoCurve == "steep":
                response = getResponseCompositeInput(word_sequence)
            else: 
                num = random.randint(1, 3)
                if num==1: 
                    response = objectUseCombinationStrategy(word_sequence)
                    response = ' '.join(response)
                elif num==2:
                    response = propertyUseStrategy(targetWord)
                else: 
                    response_list = broadUseCategoryStrategy(word_sequence)
                    response_list = ' '.join(response_list)
                    respone = "Other things in the categories " + response_list + " may be relevant"
        else: 
            response = 'I did not catch enough words'
    
    rounds = rounds+1

    print(response)
    engine.say(response)
    engine.runAndWait()


