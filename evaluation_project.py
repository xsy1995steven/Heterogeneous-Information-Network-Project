import numpy as np
from sklearn.externals import joblib
import scipy.sparse as sp

clf = joblib.load('FM_model_nor.pkl')#change model
recall = 0.0

files = []
for i in range(29):#change number of test file
    files.append("test_vec_%s.txt"%i)


dict_like_song = {}
for line in open("test_hidden_triplets.txt"):#should liked song
    line = line.strip().split()
    line[0] = line[0].strip()
    line[1] = line[1].strip()
    if line[0] not in dict_like_song:
        dict_like_song[line[0]] = set()
    dict_like_song[line[0]].add(line[1])

for filename in files:
    dict_predict_song = {}
    dict_score_to_song = {}
    recall_sub = 0.0
    index = 0
    for line in open(filename):#test file
        line = line.strip().split(':')
        feature_list = []
        feature_string = line[2].strip().strip('[').strip(']')
        for feature in feature_string.split(','):
            feature = feature.strip()
            feature_list.append(float(feature))
        feature_list = np.asarray(feature_list)
        feature_list = np.reshape(feature_list, (1,len(feature_list)))
        feature_list = sp.csc_matrix(feature_list)
        score = clf.predict(feature_list)
        index+=1
        # print(index)
        line[0] = line[0].strip()
        if line[0] not in dict_predict_song:
            dict_predict_song[line[0]] = []
        dict_predict_song[line[0]].append(score[0])
        if line[0] not in dict_score_to_song:
            dict_score_to_song[line[0]] = {}
        dict_score_to_song[line[0]][score[0]] = line[1].strip()


    for user in dict_predict_song.keys():
        list = sorted(dict_predict_song[user], reverse=True)
        sub_recall = 0.0
        for i in range(len(dict_like_song[user])):
            if dict_score_to_song[user][list[i]] in dict_like_song[user]:
                sub_recall += 1
        sub_recall = sub_recall/len(dict_like_song[user])
        print(sub_recall)
        recall_sub += sub_recall
        recall += sub_recall

    print(filename)
    print(recall_sub/len(dict_predict_song.keys()))

recall = recall/len(dict_like_song.keys())
print(recall)
