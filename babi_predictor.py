from __future__ import absolute_import
from __future__ import print_function
from functools import reduce
import re
import os
import json
    
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Merge
from keras.layers import recurrent
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

'''
Trains two recurrent neural networks based upon a story and a question.
The resulting merged vector is then queried to answer a range of bAbI tasks.

For the resources related to the bAbI project, refer to:
https://research.facebook.com/researchers/1543934539189348

Code forked from https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py.

'''


class BabiPreditor:
    
    def __init__(self):
        return

    def predictors_ai_interface(self, **kwargs):

            """
            This is the method used by Predictors.ai to interact with the model.
            It is the only method that needs to be implemented to deploy the model on
            Predictors.ai

            Inputs:

            - pipe_id (integer): id of the pipe that has to be used.

            - input_data (dictionary): dictionary that contains the input data. The keys of the dictionary 
            correspond to the names of the inputs specified in models_definition.json for the selected pipe.
            Each key has an associated value. For the input variables the associated value is the value
            of the variable, whereas for the input files the associated value is its filename. 

            - input_files_dir (string): Relative path of the directory where the input files are stored
            (the algorithm has to read the input files from there).
            - output_files_dir (string): Relative path of the directory where the output files must be stored
            (the algorithm must store the output files in there).

            Outputs:

            - output_data (dictionary): dictionary that contains the output data. The keys of the dictionary 
            correspond to the names of the outputs specified in models_definition.json for the selected pipe. 
            Each key has an associated value. For the output variables the associated value is the value
            of the variable, whereas for the output files the associated value is its filename.  
            """

            pipe_id = kwargs['pipe_id']
            input_data = kwargs['input_data']
            input_files_dir = kwargs['input_files_dir']
            output_files_dir = kwargs['output_files_dir']

            output_data = self.predict(pipe_id, input_data, input_files_dir, output_files_dir)

            return output_data
        
        
    def train(self):
        
        print("training...")
    
        task_from = 16
        task_until = 18
        
        RNN = recurrent.GRU
        EMBED_HIDDEN_SIZE = 50
        SENT_HIDDEN_SIZE = 100
        QUERY_HIDDEN_SIZE = 100
        BATCH_SIZE = 32

        babi_dir = "./tasks_1-20_v1-2/en-10k/"
        train = []
        test = []
        print("Files for training:")
        for filename in sorted(os.listdir(babi_dir))[task_from:task_until]:
            print(filename)
            filepath = babi_dir + filename
            f = open(filepath, 'r')
            if "train" in filename:
                train += get_stories(f)
            else:
                test += get_stories(f)
        print("------------")

        vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train + test)))
        # Reserve 0 for masking via pad_sequences
        self.vocab_size = len(vocab) + 1
        self.word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        self.word_idx_inv = dict(zip(self.word_idx.values(),self.word_idx.keys()))
        self.story_maxlen = max(map(len, (x for x, _, _ in train + test)))
        self.query_maxlen = max(map(len, (x for _, x, _ in train + test)))

        X, Xq, Y = self.vectorize_stories_querys_answers(train)

        sentrnn = Sequential()
        sentrnn.add(Embedding(self.vocab_size, EMBED_HIDDEN_SIZE, mask_zero=True))
        sentrnn.add(RNN(SENT_HIDDEN_SIZE, return_sequences=False))

        qrnn = Sequential()
        qrnn.add(Embedding(self.vocab_size, EMBED_HIDDEN_SIZE))
        qrnn.add(RNN(QUERY_HIDDEN_SIZE, return_sequences=False))

        self.model = Sequential()
        self.model.add(Merge([sentrnn, qrnn], mode='concat'))
        self.model.add(Dense(self.vocab_size, activation='softmax'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', class_mode='categorical')

        best_acc = 0
        keep_training = True
        patience = 5
        patience_counter = 0
        while keep_training:
            history = self.model.fit([X, Xq], Y, batch_size=BATCH_SIZE, nb_epoch=1, validation_split=0.1,
                                     show_accuracy=True)
            current_acc = history.history['val_acc']
            if current_acc > best_acc:
                model_weights = self.model.get_weights()
                best_acc = current_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > patience:
                    self.model.set_weights(model_weights)
                    keep_training = False
            print("current acc: " + str(current_acc))
            print("best acc: " + str(best_acc))
            print("------------")

        test_accs = {}
        for filename in sorted(os.listdir(babi_dir))[task_from:task_until]:
            if "test" in filename:
                filepath = babi_dir + filename
                f = open(filepath, 'r')
                test = get_stories(f)
                tX, tXq, tY = self.vectorize_stories_querys_answers(test)
                loss, acc = self.model.evaluate([tX, tXq], tY, batch_size=BATCH_SIZE, show_accuracy=True)
                test_accs[filename] = acc
                print("test acc: " + str(acc))
        
        self.generate_scores_json(test_accs)
        self.generate_model_definition()
        
        return
    
    
    def vectorize_stories_querys_answers(self, data):
        X = []
        Xq = []
        Y = []
        for story, query, answer in data:
            x = [self.word_idx[w] for w in story]
            xq = [self.word_idx[w] for w in query]
            y = np.zeros(self.vocab_size)
            y[self.word_idx[answer]] = 1
            X.append(x)
            Xq.append(xq)
            Y.append(y)
        return pad_sequences(X, maxlen=self.story_maxlen), pad_sequences(Xq, maxlen=self.query_maxlen), np.array(Y)
    
   
    def vectorize_stories_querys(self, data):
        X = []
        Xq = []
        for story, query in data:
            x = [self.word_idx[w] for w in story]
            xq = [self.word_idx[w] for w in query]
            X.append(x)
            Xq.append(xq)
        return pad_sequences(X, maxlen=self.story_maxlen), pad_sequences(Xq, maxlen=self.query_maxlen)
    
    
    def generate_scores_json(self, test_accs):
        
        """
        Calculate scores.
        
        """

        scores = []

        score = {}
        score['name'] = 'Accuracy'
        score['summary_name'] = 'Average accuracy'
        score['summary_value'] = sum(test_accs.values())/float(len(test_accs))
        score['class_wise'] = {}
        score['class_wise']['names'] = list(test_accs.keys())
        score['class_wise']['values'] = list(test_accs.values())
        scores.append(score)
    
        scores_out = {}
        scores_out["scores"] = scores
        scores_out["schema_version"] = "0.02"

        save_json_file(dict_to_save=scores_out, path="./scores.json")
        
        return
    
    
    def generate_model_definition(self):

        """
        Returns model_definition.json dictionary.

        """

        model_definition = {}
        model_definition["name"] = "Babi predictor"
        model_definition["schema_version"] = "0.02"
        model_definition["environment_name"] = "python2.7.9_October25th2015"
        model_definition["description"] = "<b>This predictor answers questions contained in the bAbI size reasoning task.</b><br />" \
                                          "It is based on the sample code available for the Keras library at https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py<br />" \
                                          "More info about the bAbI project at https://research.facebook.com/researchers/1543934539189348<br />" \
                                          'The following words are allowed (case sensitive): "' + '", "'.join(bp.word_idx.keys()) + '".<br /><br />' \
                                          "<b>Input story example:</b><br />" \
                                          "The box of chocolates fits inside the chest. " \
                                          "The container is bigger than the chest. " \
                                          "The box of chocolates is bigger than the chocolate. " \
                                          "The chest fits inside the container. " \
                                          "The suitcase is bigger than the chocolate. " \
                                          "<br /><br />" \
                                          "<b>Input questions examples:</b><br />" \
                                          "Does the chest fit in the chocolate?<br />" \
                                          "Is the box of chocolates bigger than the container?<br />" \
                                          "Does the chocolate fit in the chest?<br />" \
                                          "Is the chocolate bigger than the chest?<br />" \
                                          "Does the box of chocolates fit in the container?<br />"
        model_definition["retraining_allowed"] = False
        model_definition["base_algorithm"] = "Neural Network"     
        model_definition["score_minimized"] = "accuracy"        

        pipes = self.get_pipes()
        model_definition["pipes"] = pipes

        save_json_file(dict_to_save=model_definition, path="./model_definition.json")

        return


    def get_pipes(self, **kwargs):

        """
        Returns pipes.json dictionary.

        """

        pipes = [ 
                    {
                        "id": 0,
                        "action": "predict",
                        "name":"One by one prediction",
                        "description": "Please input one story and one question.",
                        "inputs": [
                            {
                                "name": "Story",
                                "type": "variable",
                                "variable_type": "string", 
                                "required": True
                            }, 
                            {
                                "name": "Question",
                                "type": "variable",
                                "variable_type": "string", 
                                "required": True
                            }
                        ],
                        "outputs": [
                            {
                                "name": "Answer",
                                "type": "variable",
                                "variable_type": "string"
                            }
                        ]
                    },
                ]

        return pipes

    
    def predict(self, pipe_id, input_data, input_files_dir, output_files_dir):
        
        story = input_data["Story"]
        query = input_data["Question"]

        story = story.replace("\n", " ").replace(".", " . ").replace("?", " ? ").split(" ")
        query = query.replace("\n", " ").replace(".", " . ").replace("?", " ? ").split(" ")
        story = [word for word in story if word != ""]
        query = [word for word in query if word != ""]
        
        print(story)
        print("......")
        print(query)
        print("......")
        
        if len(story) > self.story_maxlen:
            output = {"Answer": "The story is too long."}
            
        elif len(story) > self.query_maxlen:
            output = {"Answer": "The question is too long."}
            
        try:
            X, Xq = self.vectorize_stories_querys([(story, query)])
            preds = self.model.predict([X, Xq])
            answer = self.word_idx_inv[np.argmax(preds[0])]
            output = {"Answer": answer}
        except:
            output = {"Answer": "Please use allowed words only."}
        
        return output
    
    
def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format

    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.

    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


def save_json_file(**kwargs):


    """
    Saves dictionary in path.

    """

    dict_to_save = kwargs["dict_to_save"]
    path = kwargs["path"]
    with open(path,'wb') as fp:
        json.dump(dict_to_save, fp)

    return

