
#basics
import pandas as pd
import torch
import nltk
nltk.download('averaged_perceptron_tagger') # needed for pos tags



def extract_features(data:pd.DataFrame, max_sample_length:int, id2w, device):
    # this function should extract features for all samples and 
    # return a features for each split. The dimensions for each split
    # should be (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH, FEATURE_DIM)
    # NOTE! Tensors returned should be on GPU
    #
    # NOTE! Feel free to add any additional arguments to this function. If so
    # document these well and make sure you dont forget to add them in run.ipynb

    # extract data from df for quicker access
    sent_ids = [s for s in data["sentence_id"]]
    token_ids = [s for s in data["token_id"]]
    start_ids = [s for s in data["char_start_id"]]
    end_ids = [s for s in data["char_end_id"]]
    split = [s for s in data["split"]]
    all_rows = list(zip(sent_ids, token_ids, start_ids, end_ids, split))                
    
    train_features = [] # initialize feature lists
    val_features = []
    test_features = []
    
    train = data[data.split == "train"]
    tokens_in_train = train.token_id.unique()  # get list of all tokens in training data
    test_val = data[data.split != "train"]
    tokens_test_val = test_val.token_id.unique() # get list of tokens in test + val data
    tokens_unknown = [i for i in tokens_test_val if i not in tokens_in_train] # get list of "unknown" tokens
    
    for row in all_rows:
        if row[4] != "train":
            if row[1] in tokens_unknown:
                row = (row[0], -1, row[2], row[3], row[4])  # label unseen data as unknown (-1)
    
    
    
    pos2id = {}  # dictionary for mapping pos-tags to ids
    pos_ids = [] # initialize list of all pos ids
    
    sentences = []
    tokens_for_pos = token_ids.copy()
    for i in range(int(len(tokens_for_pos)/max_sample_length)):  # divide up into sentences
        token_ids = tokens_for_pos[:max_sample_length]
        tokens = [id2w[i] for i in token_ids if not id2w[i] == "padding"]  # get tokens instead of ids
        del tokens_for_pos[:max_sample_length] 
        sentences.append(tokens)
    for sentence in sentences:
        for i, token in enumerate(sentence):
            if len(token) == 0:
                sentence[i] = "x"  # dummy to avoid problems with empty tokens
        pos_list = nltk.pos_tag(sentence)  # tag all tokens
        for token, tag in pos_list:  # fill and access mapping dictionary + list of pos tags
            if tag in pos2id:
                pos_ids.append(int(pos2id[tag]))
            else:
                pos2id[tag] = len(pos2id) + 1  # start ids at 1 (because 0 reserved for padding)
                pos_ids.append(int(pos2id[tag]))
        pos_ids = pos_ids + ([0] * (max_sample_length - len(sentence))) # add label 0 for paddings
    
    
    for i in range(len(all_rows)):
        row = all_rows[i]
        if (i == 0) or (all_rows[i-1][0] != row[0]): # check whether token is first word of a sentence
            at_start = 1
            prior_word = len(id2w)  # additional id meaning that no prior token in same sentence
        else:
            at_start = 0
            prior_word = all_rows[i-1][1]
        if (i == len(all_rows)-1) or (all_rows[i+1][0] != row[0]): # check whether token is last word of a sentence
            at_end = 1
            next_word = len(id2w)  # additional id, as above
        else:
            at_end = 0
            next_word = all_rows[i+1][1]
        word_length = row[3] - row[2]  # get word length from character on- and offset
        if id2w[row[1]].isalpha:  # check for punctuation in the words (remember, 1-methyl-4-phenyl-1,2,3,6-tetrahydropyridine)
            has_punct = 0
        else:
            has_punct = 1
        pos = pos_ids[i]
        features = [at_start, at_end, prior_word, next_word, word_length, has_punct, pos] # create feature list and append to correct split
        if row[4] == "train":
            train_features.append(features)
        elif row[4] == "val":
            val_features.append(features)
        elif row[4] == "test":
            test_features.append(features)
    
    # convert lists to tensors and reshape to correct dimensions
    train_f_tensor = torch.LongTensor(train_features)
    train_f_tensor = train_f_tensor.reshape([(len(train_f_tensor)//max_sample_length),max_sample_length, 7]).to(device) # number features hardcoded: 7
    val_f_tensor = torch.LongTensor(val_features)
    val_f_tensor = val_f_tensor.reshape([(len(val_f_tensor)//max_sample_length),max_sample_length, 7]).to(device)
    test_f_tensor = torch.LongTensor(test_features)
    test_f_tensor = test_f_tensor.reshape([(len(test_f_tensor)//max_sample_length),max_sample_length, 7]).to(device)

    return train_f_tensor, val_f_tensor, test_f_tensor
