import os
import pandas as pd

#reset indeces in the data set
def reindex(data):
    index = range(len(data))
    data['index'] = index
    data.set_index(['index'], inplace=True)
    return data

def get_data(path):
    data = []
    files = os.listdir(path)
    for file in files:
        f = open(path+file, encoding = "ISO-8859-1")
        words_list = f.read()
        data.append(words_list)
        f.close()
    return data

def create_df_spam_assasin(path):
    easy_ham = get_data(path + "easy_ham/")
    hard_ham = get_data(path + "hard_ham/")
    ham = easy_ham + hard_ham
    spam_one = get_data(path + "spam/")
    spam_two = get_data(path + "spam_2/")
    spam = spam_one + spam_two

    ham_label = []
    for i in range(len(ham)):
        ham_label.append(0)
    
    spam_label = []
    for i in range(len(spam)):
        spam_label.append(1)
    
    ham_df = pd.DataFrame({'text' : ham,'label' : ham_label})
    spam_df = pd.DataFrame({'text' : spam, 'label' : spam_label})
    spam_assasin = pd.concat([ham_df, spam_df], axis=0)
    spam_assasin_final = reindex(spam_assasin)
    return spam_assasin_final


def create_df_enron(path):
    enron_df = pd.DataFrame()
    for i in range(1,7):
        ham_path = path + "/enron" + str(i) + "/ham/"
        spam_path = path + "/enron" + str(i) + "/spam/"
        
        ham = get_data(ham_path)
        spam = get_data(spam_path)
        
        ham_label = []
        for i in range(len(ham)):
            ham_label.append(0)
    
        spam_label = []
        for i in range(len(spam)):
            spam_label.append(1)

        ham_df = pd.DataFrame({'text' : ham,'label' : ham_label})
        spam_df = pd.DataFrame({'text' : spam, 'label' : spam_label})
        enron_df = pd.concat([enron_df,ham_df,spam_df],axis=0)
    final = reindex(enron_df)
    return final

#replace empty subject entries with "Subject"
#important for merging subject with message and it doesn't change the content of the data, because "Subject" is a stop word in context of mails
def fill_empty_subject_entries(input, col_1, col_2):
    temp = pd.DataFrame()
    temp = input[input[col_1].notnull() | input[col_2].notnull()]
    temp.loc[temp[col_1].astype(str)=='nan',col_1] = 'Subject'
    temp.loc[temp[col_2].astype(str)=='nan',col_2] = 'Subject'
    return temp

def create_df_from_csv(path,col_1='subject',col_2='message',col_3='label'):
    df = pd.read_csv(path)
    df = fill_empty_subject_entries(df, col_1, col_2)
    final_df = pd.DataFrame()
    final_df['text'] = df[col_1] + df[col_2]
    final_df['label'] = df[col_3]
    return final_df

def create_trec_from_csv(path):
    col_1='subject'
    col_2='email_to'
    col_3='email_from'
    col_4='message'
    label_col = 'label'
    df = pd.read_csv(path)
    df = fill_empty_subject_entries(df, col_1, col_2)
    df = fill_empty_subject_entries(df, col_3, col_4)
    final_df = pd.DataFrame()
    final_df['text'] = df[col_1] + df[col_2] + df[col_3] + df[col_4]
    final_df['label'] = df[label_col]
    return final_df