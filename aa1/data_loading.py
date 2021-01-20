
#basics
import random
import pandas as pd
import torch
import xml.etree.ElementTree as ET
from pathlib import Path
import random
from collections import Counter
import matplotlib.pyplot as plt
#from matplotlib_venn import venn3, venn3_circles
from venn import venn

class DataLoaderBase:

    #### DO NOT CHANGE ANYTHING IN THIS CLASS ### !!!!

    def __init__(self, data_dir:str, device=None):
        self._parse_data(data_dir)
        assert list(self.data_df.columns) == [
                                                "sentence_id",
                                                "token_id",
                                                "char_start_id",
                                                "char_end_id",
                                                "split"
                                                ]

        assert list(self.ner_df.columns) == [
                                                "sentence_id",
                                                "ner_id",
                                                "char_start_id",
                                                "char_end_id",
                                                ]
        self.device = device
        

    def get_random_sample(self):
        # DO NOT TOUCH THIS
        # simply picks a random sample from the dataset, labels and formats it.
        # Meant to be used as a naive check to see if the data looks ok
        sentence_id = random.choice(list(self.data_df["sentence_id"].unique()))
        sample_ners = self.ner_df[self.ner_df["sentence_id"]==sentence_id]
        sample_tokens = self.data_df[self.data_df["sentence_id"]==sentence_id]

        decode_word = lambda x: self.id2word[x]
        sample_tokens["token"] = sample_tokens.loc[:,"token_id"].apply(decode_word)

        sample = ""
        for i,t_row in sample_tokens.iterrows():

            is_ner = False
            for i, l_row in sample_ners.iterrows():
                 if t_row["char_start_id"] >= l_row["char_start_id"] and t_row["char_start_id"] <= l_row["char_end_id"]:
                    sample += f'{self.id2ner[l_row["ner_id"]].upper()}:{t_row["token"]} '
                    is_ner = True
            
            if not is_ner:
                sample += t_row["token"] + " "

        return sample.rstrip()



class DataLoader(DataLoaderBase):


    def __init__(self, data_dir:str, device=None):
        super().__init__(data_dir=data_dir, device=device)


    def fill_dfs(self, file_list):  # read data from xml into lists of lists (for df later)
        self.file_list = file_list
        self.data_list = []
        self.ner_list = []
        i_data_list = [] # intermediate list for storage
        self.max_sample_length = 0
        for file in file_list:
            if str(file).split("/")[2] == "Test":
                split = "test"
            else:
                split = random.choices(["train", "val"], weights = (80, 20), k = 1)[0]  # split train into train and eval 
            tree = ET.parse(file)
            root = tree.getroot()
            for elem in root:
                sent_id = elem.get("id")
                sentence = elem.get("text")                
                tokens = self.get_tokens(sentence)  # tokenize sentence, see helper function below
                sentence_token_list = []
                if len(tokens) > self.max_sample_length:  # get length of longest sentence (max_sample_length)
                    self.max_sample_length = len(tokens)
                for subelem in elem:
                    if subelem.tag == "entity": # add entities to ner_df:
                        ner_id = self.get_id(subelem.get("type"), self.ner2id)
                        if len(subelem.get("charOffset").split("-")) == 2: # for single and continuous compound words:
                            start = int(subelem.get("charOffset").split("-")[0])
                            for word in subelem.get("text").split(" "):  # separate at spaces
                                if word != "":
                                    last_char = word[-1]
                                    start_char = int(start)
                                    if last_char.isalnum():  # condition to remove trailing punctuation (mainly "-")
                                        end_char = int(start) + len(word) - 1
                                        start += len(word) + 1
                                    else: 
                                        j = 0
                                        while not last_char.isalnum(): 
                                            word = word.replace(last_char, "")
                                            j += 1
                                            last_char = word[-1]
                                        end_char = int(start) + len(word)-1
                                        start += len(word) + (1+j)
                                    self.ner_list.append([sent_id, ner_id, int(start_char), int(end_char)])
                               
                        else:
                            for word in subelem.get("charOffset").split(";"):  # for interrupted compound words
                                start_char, end_char = word.split("-")
                                self.ner_list.append([sent_id, ner_id, int(start_char), int(end_char)])
                for token in tokens:  # add tokens to data_df:
                    token_id = self.get_id(token[0], self.word2id)
                    start_char = token[1]
                    end_char = token[2]
                    sentence_token_list.append([sent_id, token_id, start_char, end_char, split])
                    
            i_data_list.append(sentence_token_list)           
        for sent_l in i_data_list:
            if len(sent_l) > 0:
                split = sent_l[-1][4]
                if len(sent_l) < self.max_sample_length:  # add padding so that all samples have same length
                    for i in range(self.max_sample_length - len(sent_l)):
                        sent_l.append([0, 0, 0, 0, split]) 
                self.data_list.extend(sent_l)   
        pass                   
                        
                
                
    def get_id(self, token, dic):  # helper function to access and update the word2id and ner2id dictionaries
        self.token = token
        self.dic = dic
        if token in dic:  
            return dic[token]
        else:
            dic[token] = len(dic) +1  # id starts at 1, since 0 is reserved for paddings
            return dic[token]
        
    
    def get_tokens(self, sentence):  # split the sentence into tokens with start and end characters
        self.sentence = sentence
        sentence = sentence.replace(";", " ")
        sentence = sentence.replace("/", " ")
        tokens = sentence.split(" ") 
        tokens_with_numbers = []
        i = 0
        for token in tokens: 
            if token != "":
                token = token.lower() 
                start_char = i
                last_char = token[-1]  # actual last character, vs. end_char is position of that character in sentence
                if last_char.isalnum():  # condition to remove trailing punctuation in words
                    end_char = i + len(token)-1
                    i += len(token) + 1
                elif any(i.isalnum() for i in token): 
                    j = 0
                    while not last_char.isalnum():
                        token = token.replace(last_char, "")
                        j += 1
                        last_char = token[-1]
                    end_char = i + len(token)-1
                    i += len(token) + (1+j)
                else:
                    i += len(token) +1
                    continue
                tokens_with_numbers.append((token, start_char, end_char))
            #else:
            #    i += 1
        return tokens_with_numbers
    
    
    def _parse_data(self,data_dir):
        # Should parse data in the data_dir, create two dataframes with the format specified in
        # __init__(), and set all the variables so that run.ipynb run as it is.
        #
        # NOTE! I strongly suggest that you create multiple functions for taking care
        # of the parsing needed here. Avoid create a huge block of code here and try instead to 
        # identify the seperate functions needed.
        
        # initialize dictionaries
        self.word2id = {}
        self.ner2id = {}        
        
        # read in the files
        allfiles = Path(data_dir)
        file_list = [f for f in allfiles.glob('**/*.xml') if f.is_file()]
        self.fill_dfs(file_list)
        
        # make DataFrames
        self.data_df = pd.DataFrame(self.data_list, columns = ["sentence_id", "token_id", "char_start_id", "char_end_id", "split"])
        self.ner_df = pd.DataFrame(self.ner_list, columns = ["sentence_id", "ner_id", "char_start_id", "char_end_id"]) 
        
        # transpose token-id dicts to id-token and get vocab:
        self.id2word = {y:x for x,y in self.word2id.items()}
        self.id2word[0] = "padding"  # special token for artificially added "tokens"
        self.id2ner = {y:x for x,y in self.ner2id.items()}
        self.vocab = list(self.word2id.keys())


    def get_y(self):
        # Should return a tensor containing the ner labels for all samples in each split.
        # the tensors should have the following following dimensions:
        # (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH)
        # NOTE! the labels for each split should be on the GPU
        
        # divide df by splits
        df_train = self.data_df[self.data_df.split=="train"]
        df_val = self.data_df[self.data_df.split=="val"]
        df_test = self.data_df[self.data_df.split=="test"]
        
        #get labels
        self.train_labels = self.get_labels(df_train)
        self.val_labels = self.get_labels(df_val)
        self.test_labels = self.get_labels(df_test)          
        
        device = torch.device('cuda:0')
        # put labels into tensors and reshape
        train_tensor = torch.LongTensor(self.train_labels)
        self.train_tensor = train_tensor.reshape([(len(self.train_labels)//self.max_sample_length),self.max_sample_length]).to(device)
        val_tensor = torch.LongTensor(self.val_labels)
        self.val_tensor = val_tensor.reshape([(len(self.val_labels)//self.max_sample_length),self.max_sample_length]).to(device)
        test_tensor = torch.LongTensor(self.test_labels)
        self.test_tensor = test_tensor.reshape([(len(self.test_labels)//self.max_sample_length),self.max_sample_length]).to(device)
        
        return self.train_tensor, self.val_tensor, self.test_tensor
                       
    def get_labels(self, df):  # helper function to extract ner labels from dataframe
        self.df = df
        label_list = []
        
        # extract relevant info from df
        sent_ids = [s for s in df["sentence_id"]] 
        start_ids = [s for s in df["char_start_id"]]
        end_ids = [s for s in df["char_end_id"]]
        id_tuples = list(zip(sent_ids, start_ids, end_ids))
        
        label_sent_ids = [s for s in self.ner_df["sentence_id"]]
        label_start_ids = [s for s in self.ner_df["char_start_id"]]
        label_end_ids = [s for s in self.ner_df["char_end_id"]]
        labels = [s for s in self.ner_df["ner_id"]]
        label_tuples = list(zip(label_sent_ids, label_start_ids, label_end_ids))
        
        # compare whether token is an entity. If not, assign label 0
        for t in id_tuples:
            if t[0] == 0:
                label = 5
            elif t in label_tuples:
                label = labels[label_tuples.index(t)]
            else:
                label = 0
            label_list.append(label)            
        
        return label_list              
                       
    def plot_split_ner_distribution(self):
        # should plot a histogram displaying ner label counts for each split
        self.get_y()
        
        # get label counts
        train_counts = Counter(self.train_labels)
        val_counts = Counter(self.val_labels)
        test_counts = Counter(self.test_labels)
        
        # put counts into a dataframe:
        counts_df = pd.DataFrame([train_counts, val_counts, test_counts], index=['train', 'val', 'test'])
        counts_df.plot(kind='bar') 
        plt.show()
        pass


    def plot_sample_length_distribution(self):
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of sample lengths in number tokens
        sent_ids = [s for s in self.data_df["sentence_id"]]
        token_counts = Counter(sent_ids)  # count sentence length (= number of times that sent id appears in df)
        del token_counts[0]  # get rid of padding sentence_id
        token_dict = dict(token_counts)
       
        counts_list = [i for i in token_dict.values()] 
        
        plt.hist(counts_list, 50) # plot how many sentences have 1 tokens, 2 tokens, etc., bin size set to 50
        plt.show()
        pass


    def plot_ner_per_sample_distribution(self):        
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of number of NERs in sentences
        # e.g. how many sentences has 1 ner, 2 ner and so on
        
        count_list = []
        for sent_id in self.ner_df["sentence_id"]:
            count = len(self.ner_df[self.ner_df["sentence_id"] == sent_id])  # get number of ner's per sentence 
            count_list.append(count)
            
        plt.hist(count_list, 50)
        plt.show()
        pass


    def plot_ner_cooccurence_venndiagram(self):
        # FOR BONUS PART!!
        # Should plot a ven-diagram displaying how the ner labels co-occur
        
        all_counts = []
        for ner in [0, 1, 2, 3, 4]:
            n_df = self.ner_df[self.ner_df["ner_id"] == ner]
            sents = [i for i in n_df["sentence_id"]] 
            all_counts.append(sents)
            
        list0, list1, list2, list3, list4 = all_counts
        #list2 = list2 + list3
        venn({"group": set(list1), "drug_n": set(list2), "drug": set(list3), "brand": set(list4)})
        plt.show()
        pass



