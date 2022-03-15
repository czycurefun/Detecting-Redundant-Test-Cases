import requests
import json
from jieba import lcut
from gensim.similarities import SparseMatrixSimilarity
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import Levenshtein
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import jieba
from scipy.linalg import norm
import gensim
import pandas as pd
from sklearn.metrics import cluster
from numpy import mean
from sklearn.metrics.cluster import entropy, mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import re
NMI = lambda x, y: normalized_mutual_info_score(x, y, average_method='arithmetic')

def labels_to_original(labels, original_corpus):
        assert len(labels) == len(original_corpus)
        max_label = max(labels)
        number_label = [i for i in range(0, max_label + 1, 1)]
        number_label.append(-1)
        result = [[] for i in range(len(number_label))]
        for i in range(len(labels)):
            index = number_label.index(labels[i])
            result[index].append(original_corpus[i])
        return result
def judge_contains(objone,objtwo):
    objone_list = list(objone)
    objtwo_list = list(objtwo)
    flagone = True
    flagtwo = True
    for each in objone_list:
        if each not in objtwo_list:
            flagone = False
            break
    for each in objtwo_list:
        if each not in objone_list:
            flagtwo = False
            break
    if flagone or flagtwo:
        return True
    else:
        return False



def jaccard_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))
    
    # 将字中间加入空格
    s1, s2 = add_space(s1), add_space(s2)
    # 转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 求交集
    numerator = np.sum(np.min(vectors, axis=0))
    # 求并集
    denominator = np.sum(np.max(vectors, axis=0))
    # 计算杰卡德系数
    return 1.0 * numerator / denominator

data_path ='D:/czy/OneDrive/UCAS/VQA/测试推荐/公共数据集构建/data/word2vec.vector'
with open(data_path, 'r', encoding='UTF-8') as inp_vec:
    emb_vec = inp_vec.readlines()
    word_vectors = {}
    for vec in emb_vec:
        result = vec.strip().split(' ',1)
        word_vectors[result[0]] = np.array(list((map(float, result[1].split()))))

url = "http://192.168.15.183:8000//api/EntityRelationEx/"
datalist = []

def vector_similarity(s1, s2):
        def sentence_vector(s):
            words = jieba.lcut(s)
            v = np.zeros(48)
            for word in words:
                v += word_vectors[word]
            v /= len(words)

            return v
        v1, v2 = sentence_vector(s1), sentence_vector(s2)
        return np.dot(v1, v2) / (norm(v1) * norm(v2))
testnumlist = [[4,10],[10,18],[26,32],[261,269],[326,334],[401,409],[591,598],[807,813],[868,876],[1000,1006]]
# object_path = "D:/czy/OneDrive/UCAS/VQA/测试推荐/公共数据集构建/data/project_splitdata/project10_new.xlsx"
# object_path = "D:/czy/OneDrive/UCAS/VQA/测试推荐/公共数据集构建/数据标注部分抽取1170.xlsx"
# object_path = "D:/czy/OneDrive/UCAS/VQA/测试推荐/公共数据集构建/数据标注2000.xlsx"
object_path = "D:/czy/OneDrive/UCAS/VQA/测试推荐/公共数据集构建/data/project_splitdata/project5_new.xlsx"
# object_path = "D:/czy/OneDrive/UCAS/VQA/测试推荐/公共数据集构建/data/testcaseredundancy_result/testcase5.xlsx"
try:
    pd_sheets = pd.ExcelFile(object_path)
except Exception as e:
    print("读取 {} 文件失败".format(object_path), e)

df = pd.read_excel(pd_sheets, "Sheet1", header=[0])
temp = "最小内存余量测试"
testname = "文件/文件夹视图测试"
testcase_name = []
testcase_des = []
test_relation = []
testcasename_list = []
testcasedes_list = []
testrelation_list = []
for row in df.itertuples(index=True):
    row_list = list(row)
    testcasename = row_list[5:6]
    #测试说明
    testdes = row_list[7:8]
    #预置条件对象
    preconobject = row_list[8:9]
    #输入操作
    test_input = row_list[9:10]
    #期望测试结果
    test_expectedresult = row_list[10:11]
    testrelation = row_list[4:5]
    testitem = row_list[3:4]
    if pd.isnull(test_input):
        test_input = [""]
    if pd.isnull(test_expectedresult):
        test_expectedresult = [""]
    if pd.isnull(preconobject):
        preconobject = [""]
    #只考虑一个测试项
    # testcase_name.append(testcasename[0])
    # testcase_des.append(testdes[0] )
    # test_relation.append(testrelation[0])
    if testname == testitem[0].strip() and testname != "命令":
        testcase_name.append(testcasename[0])
        testcase_des.append(testdes[0] )
        test_relation.append(testrelation[0])
    elif testname != "命令":
        testcasename_list.append(testcase_name)
        testcasedes_list.append(testcase_des)
        testrelation_list.append(test_relation)
        testcase_name = []
        testcase_des = []
        test_relation = []
        testcase_name.append(testcasename[0])
        testcase_des.append(testdes[0] )
        test_relation.append(testrelation[0])
        testname = testitem[0].strip()
    else:
        testname = testitem[0].strip()
testcasename_list.append(testcase_name)
testcasedes_list.append(testcase_des)
testrelation_list.append(test_relation)

predictlist = []
nmiscorelist = []
redundancy_TP = 0
redundancy_FP = 0
redundancy_FN = 0
unredundancy_TP = 0
unredundancy_FP = 0
unredundancy_FN = 0
kongnum = 0
log_path = "D:/czy/OneDrive/UCAS/VQA/测试推荐/公共数据集构建/data/result/project10.txt"
sameitemnum = 0
nosameitemnum= 0
# obj_list = []
# allresult_path = "D:/czy/OneDrive/UCAS/VQA/测试推荐/公共数据集构建/data/result/allresult.json"
# with open(allresult_path, 'r', encoding="utf-8") as f:
#     all_resultlist = json.loads(f.read())

#     for resultnew in all_resultlist:
#         for entity in resultnew["entities"]:
#             if entity["type"] == "测试对象":
#                 obj_list.append(entity["value"].replace(" ","") )

#     text_file = " ".join([" ".join(jieba.cut(c)) for c in obj_list])
#     word_count = Counter(text_file.split())
#     total_count = sum(word_count.values())
#     unigram = {}
#     for item in word_count.items():
#         unigram[item[0]] = item[1] / total_count

for k,datalist in enumerate(testcasedes_list):

# for num in range(1000,1006):
#         path = "D:/czy/OneDrive/UCAS/VQA/测试推荐/公共数据集构建/data/rawdata/" + str(num) + ".txt" 
#         with open(path, 'r', encoding="utf-8") as f:
#             temp = f.read()
#             datalist.append(temp.replace("\n",""))
# datalist = ["测试在高负载的情况下，系统的稳定性。1.操作系统安装成功，且可正常工作"]
    resultlist = []
    for num, each in enumerate(datalist):
        if len(each) <= 200:
            datas = {"text":each}
            data = json.dumps(datas)
            headers = {'Content-Type': 'application/json'}
            response = requests.post(url, data=data, headers=headers)
            result = response.json()
            # if result[0]["entities"] != []:
            result[0]["tokens"] = each
            resultlist.append(result[0])
        else:
            each = each[0:200]
            datas = {"text":each}
            data = json.dumps(datas)
            headers = {'Content-Type': 'application/json'}
            response = requests.post(url, data=data, headers=headers)
            result = response.json()
            # if result[0]["entities"] != []:
            result[0]["tokens"] = each
            resultlist.append(result[0])

    # print(resultlist)
    entitylist = []
    for result in resultlist:
        mergeentity = ""
        testobject = ""
        testoperate = ""
        teststate = ""
        testprecon = ""
        entitydic = {}
        final_list = []
        judgeobject = True
        judgeoperate = True
        judgestate = True
        judgeprecon = True
        obj_flag = 0
        for entity in result["entities"]:
            if entity["type"] == "测试对象":
                obj_flag += 1
        #两个以上的对象做查找关系操作才有意义
        if obj_flag >= 2:
            for entity in result["entities"]:
                if entity["type"] == "测试对象":
                    obj_id = entity["id"]
                    obj_value = entity["value"].replace(" ","")
                    testcasetemp_dic = {}
                    testcasetemp_dic["测试对象"] = obj_value
                    #考虑多元组全是对象，没有关系的情况
                    if result["relations"] != []:
                        for relation in result["relations"]:
                            if relation["head"].strip() != "" and relation["tail"].strip() != "":
                                if relation["tailIndex"] == obj_id:
                                    headIndex = relation["headIndex"]
                                    #找与对象相连的另一个实体
                                    for each in result["entities"]:
                                        if each["id"] == headIndex:
                                            #考虑一个对象有多种状态等情况
                                            if each["type"] in testcasetemp_dic:
                                                testcasetemp_dic[each["type"]] += "," + each["value"].replace(" ","") 
                                            else:
                                                testcasetemp_dic[each["type"]] = each["value"].replace(" ","")
                    final_list.append(testcasetemp_dic)
        else:
            testcasetemp_dic = {}
            if result["entities"] != [] :
                for entity in result["entities"]:
                    if entity["type"] != "预置条件对象":
                        if entity["type"] in testcasetemp_dic:
                            testcasetemp_dic[entity["type"]] += "," + entity["value"].replace(" ","") 
                        else:
                            testcasetemp_dic[entity["type"]] = entity["value"].replace(" ","")
                    else:
                        testcasetemp_dic["测试对象"] = result["sentence"]
            else:
                if "是否支持" in result["sentence"]:
                    objentity = result["sentence"][result["sentence"].find("是否支持") + 4:len(result["sentence"])] 
                    testcasetemp_dic["测试对象"] = objentity
                elif  "是否可以" in result["sentence"]:
                    objentity = result["sentence"][result["sentence"].find("是否可以") + 4:len(result["sentence"])] 
                    testcasetemp_dic["测试对象"] = objentity
                elif "支持" in result["sentence"]:
                    objentity = result["sentence"][result["sentence"].find("支持") + 2:len(result["sentence"])] 
                    testcasetemp_dic["测试对象"] = objentity
                elif "可以" in result["sentence"]:
                    objentity = result["sentence"][result["sentence"].find("可以") + 2:len(result["sentence"])] 
                    testcasetemp_dic["测试对象"] = objentity
                else:
                    testcasetemp_dic["测试对象"] = result["sentence"]
            final_list.append(testcasetemp_dic)
        entitylist.append(final_list)

    
    #只取测试对象查看
    watch_list = []
    obj_list = []
    operate_list = []
    state_list = []
    tool_list = []
    condition_list = []
    tuple_list = []
    #初始化元组的全集
    alltuple_list = []
    num_id = 0
    for entity in entitylist:
        templist = []
        #这里就算上了一个测试用例多个对象的情况
        for entity_dic in entity:
            #删除特性
            if "软件特性" in entity_dic:
                del entity_dic["软件特性"]
            alltuple_list.append(entity_dic)
            if "测试对象" in entity_dic:
                # tempdic = {}
                # tempdic["测试对象"] = entity_dic["测试对象"]
                # tempdic["id"] = num_id
                # watch_list.append(tempdic)
                tuple_list.append(num_id)
                obj_list.append(entity_dic["测试对象"].lower())
            else:
                tuple_list.append(num_id)
                obj_list.append(" ")
            if "测试操作" in entity_dic:
                tempdic = {}
                tempdic["测试操作"] = entity_dic["测试操作"]
                tempdic["id"] = num_id
                operate_list.append(entity_dic["测试操作"].lower())
            else:
                operate_list.append(" ")
            if "测试状态" in entity_dic:
                tempdic = {}
                tempdic["测试状态"] = entity_dic["测试状态"]
                tempdic["id"] = num_id
                state_list.append(entity_dic["测试状态"].lower())
            else:
                state_list.append(" ")
            if "测试工具" in entity_dic:
                tempdic = {}
                tempdic["测试工具"] = entity_dic["测试工具"]
                tempdic["id"] = num_id
                tool_list.append(entity_dic["测试工具"].lower())
            else:
                tool_list.append(" ")
            if "满足条件" in entity_dic:
                tempdic = {}
                tempdic["满足条件"] = entity_dic["满足条件"]
                tempdic["id"] = num_id
                condition_list.append(entity_dic["满足条件"].lower())
            else:
                condition_list.append(" ")
        num_id += 1
        # watch_list.append(templist)
    #w2v+sif+hdbscan
    if len(obj_list) != 1 :
        # for k,each in enumerate(testcase_name):
        #     if each == True or each == False:
        #         testcase_name[k] = ""
        text_file = " ".join([" ".join(jieba.cut(c)) for c in obj_list])
        word_count = Counter(text_file.split())
        total_count = sum(word_count.values())
        unigram = {}
        for item in word_count.items():
            unigram[item[0]] = item[1] / total_count
        #对象初始化，用SIF
        all_vector_representation = np.zeros(shape=(len(obj_list), 48))
        for i, line in enumerate(obj_list):
            word_sentence = jieba.cut(line)
            sent_rep = np.zeros(shape=[48, ])
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue
                try:
                    weight = 0.1 / (0.1 + unigram[word])
                except KeyError:
                    weight = 1
                sent_rep += wv * weight
                # sent_rep += wv
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep

        pca = PCA(n_components=1)
        pca.fit(all_vector_representation)
        pca = pca.components_

        XXobj = all_vector_representation - all_vector_representation.dot(pca.transpose()) * pca
        # XXobj = all_vector_representation
        # #归一到[0,1]
        # scaler = StandardScaler()
        # XXobj = scaler.fit_transform(XXobj)

        
         #操作初始化,不用SIF方法
        all_vector_representation = np.zeros(shape=(len(operate_list), 48))
        for i, line in enumerate(operate_list):
            word_sentence = jieba.cut(line)
            sent_rep = np.zeros(shape=[48, ])
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue
                
                sent_rep += wv 
                # sent_rep += wv
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep
        #归一到[0,1]
        # scaler = StandardScaler()
        # XXoperate = scaler.fit_transform(all_vector_representation)
        
        XXoperate = all_vector_representation

        
        #状态初始化,不用SIF方法
        all_vector_representation = np.zeros(shape=(len(state_list), 48))
        for i, line in enumerate(state_list):
            word_sentence = jieba.cut(line)
            sent_rep = np.zeros(shape=[48, ])
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue
                
                sent_rep += wv 
                # sent_rep += wv
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep
        #归一到[0,1]
        # scaler = StandardScaler()
        # XXstate = scaler.fit_transform(all_vector_representation)
        XXstate = all_vector_representation

        # text_file = " ".join([" ".join(jieba.cut(c)) for c in tool_list])
        # word_count = Counter(text_file.split())
        # total_count = sum(word_count.values())
        # unigram = {}
        # for item in word_count.items():
        #     unigram[item[0]] = item[1] / total_count
        #工具初始化,不用SIF方法
        all_vector_representation = np.zeros(shape=(len(tool_list), 48))
        for i, line in enumerate(tool_list):
            word_sentence = jieba.cut(line)
            sent_rep = np.zeros(shape=[48, ])
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue
                # weight = 0.1 / (0.1 + unigram[word])
                sent_rep += wv 
                # sent_rep += wv
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep
        #归一到[0,1]
        # scaler = StandardScaler()
        # XXtool = scaler.fit_transform(all_vector_representation)
        # pca = PCA(n_components=1)
        # pca.fit(all_vector_representation)
        # pca = pca.components_
        # XXtool = all_vector_representation - all_vector_representation.dot(pca.transpose()) * pca
        XXtool = all_vector_representation

        # text_file = " ".join([" ".join(jieba.cut(c)) for c in condition_list])
        # word_count = Counter(text_file.split())
        # total_count = sum(word_count.values())
        # unigram = {}
        # for item in word_count.items():
        #     unigram[item[0]] = item[1] / total_count

        #条件初始化,不用SIF方法
        all_vector_representation = np.zeros(shape=(len(condition_list), 48))
        for i, line in enumerate(condition_list):
            word_sentence = jieba.cut(line)
            sent_rep = np.zeros(shape=[48, ])
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue
                # weight = 0.1 / (0.1 + unigram[word])
                sent_rep += wv 
                # sent_rep += wv
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep
        #归一到[0,1]
        # scaler = StandardScaler()
        # XXcondition = scaler.fit_transform(all_vector_representation)
        # pca = PCA(n_components=1)
        # pca.fit(all_vector_representation)
        # pca = pca.components_
        # XXcondition = all_vector_representation - all_vector_representation.dot(pca.transpose()) * pca
        XXcondition = all_vector_representation
        score_num = 0
        XXzero = np.zeros(shape=(len(obj_list), 48))
        #初始化赋值
        totalsum_list = cosine_similarity(XXzero)
        objcos_simlist = cosine_similarity(XXobj)
        objcos_simlist_oral = cosine_similarity(XXobj)
        # scaler = MinMaxScaler()
        # objcos_simlist = scaler.fit_transform(objcos_simlist)
        for numone,obj in enumerate(obj_list):
            for numtwo in range(numone+1,len(obj_list)):
                if judge_contains(obj_list[numone],obj_list[numtwo]) and objcos_simlist[numone][numtwo] >= 0.95:
                    objcos_simlist[numone][numtwo] = 1
                elif judge_contains(obj_list[numone],obj_list[numtwo]) and (len(obj_list[numone]) <= 4 and len(obj_list[numtwo]) <= 4):
                    objcos_simlist[numone][numtwo] = 1
                elif obj_list[numone] == obj_list[numtwo]:
                    objcos_simlist[numone][numtwo] = 1
        if not np.all(objcos_simlist == 0):
            totalsum_list += objcos_simlist
            score_num += 1
        operatecos_simlist = cosine_similarity(XXoperate)
        operatecos_simlist_oral = cosine_similarity(XXoperate)
        for numone,obj in enumerate(operate_list):
            for numtwo in range(numone+1,len(operate_list)):
                if judge_contains(operate_list[numone],operate_list[numtwo]) and operatecos_simlist[numone][numtwo] >= 0.85:
                # if judge_contains(operate_list[numone],operate_list[numtwo]):
                    operatecos_simlist[numone][numtwo] = 1
        if not np.all(operatecos_simlist == 0):
            totalsum_list += operatecos_simlist
            score_num += 1
        statecos_simlist = cosine_similarity(XXstate)
        # for numone,obj in enumerate(state_list):
        #     for numtwo in range(numone+1,len(state_list)):
        #         if judge_contains(state_list[numone],state_list[numtwo]) and statecos_simlist[numone][numtwo] >= 0.85:
        #             statecos_simlist[numone][numtwo] = 1
        if not np.all(statecos_simlist == 0):
            totalsum_list += statecos_simlist
            score_num += 1
        toolcos_simlist = cosine_similarity(XXtool)
        # for numone,obj in enumerate(tool_list):
        #     for numtwo in range(numone+1,len(tool_list)):
        #         if judge_contains(tool_list[numone],tool_list[numtwo]) and toolcos_simlist[numone][numtwo] >= 0.85:
        #             toolcos_simlist[numone][numtwo] = 1
        if not np.all(toolcos_simlist == 0):
            totalsum_list += toolcos_simlist
            score_num += 1
        conditioncos_simlist = cosine_similarity(XXcondition)
        for numone,obj in enumerate(condition_list):
            for numtwo in range(numone+1,len(condition_list)):
                if judge_contains(condition_list[numone],condition_list[numtwo])  and conditioncos_simlist[numone][numtwo] >= 0.85:
                    conditioncos_simlist[numone][numtwo] = 1
        if not np.all(conditioncos_simlist == 0):
            totalsum_list += conditioncos_simlist
            score_num += 1
        totalsum_list = totalsum_list/score_num
        # nonsense_operatelist = ["支持","实现","具有","存在","具备"]
        clusterlist = []
        alpha = 0.5
        threshold = 0.99
        # totalsum_list = (alpha/2)*(objcos_simlist + operatecos_simlist) + ((1-alpha)/3)*(statecos_simlist + toolcos_simlist + conditioncos_simlist  )
        if not (totalsum_list.min() == 1 and totalsum_list.max() == 1):
            scaler = MinMaxScaler()
            totalsum_list = scaler.fit_transform(totalsum_list)
        # totalsum_list = (totalsum_list + 1)/2
        # print(totalsum_list)
        #先初始化簇，每个测试用例都不一样,从2开始，0和1用作冗余标注
        for num in range(2,len(totalsum_list) + 2):
            clusterlist.append(num)
        
        for numone in range(0,len(totalsum_list)):
            for numtwo in range(numone + 1,len(totalsum_list)):

                if alltuple_list[numone].keys() == alltuple_list[numtwo].keys():
                    sameitemnum += 1
                    #对操作小于3的单独处理
                    if (len(operate_list[numone]) <= 3 and len(operate_list[numtwo]) <= 3) and (operate_list[numone] != operate_list[numtwo]):
                        continue
                    #抽取只有操作
                    elif operate_list[numtwo] != operate_list[numone]  and len(alltuple_list[numone]) == 1:
                        continue
                    #考虑对象出现英文的情况
                    elif bool(re.search('[a-zA-Z]', obj_list[numtwo])) or bool(re.search('[a-zA-Z]', obj_list[numone])):
                        if re.sub('[\u4e00-\u9fa5]', '', obj_list[numone]) != re.sub('[\u4e00-\u9fa5]', '', obj_list[numtwo]):
                            continue
                        if bool(re.search('[a-zA-Z]', tool_list[numtwo])) or bool(re.search('[a-zA-Z]', tool_list[numone])):
                            if re.sub("[\u4e00-\u9fa5\0-9\,\。]", "", tool_list[numtwo]) != re.sub("[\u4e00-\u9fa5\0-9\,\。]", "", tool_list[numone]):
                                continue
                        elif tool_list[numtwo] != tool_list[numone]:
                            continue 
                        # if bool(re.search('[1-9]', condition_list[numtwo])) or bool(re.search('[1-9]', condition_list[numone])):
                        #     if "".join(list(filter(str.isdigit, condition_list[numtwo]))) != "".join(list(filter(str.isdigit, condition_list[numone]))):
                        #         continue
                        if condition_list[numtwo] != condition_list[numone]:
                            continue
                        if operatecos_simlist_oral[numone][numtwo] < 0.999 and operatecos_simlist_oral[numone][numtwo] !=0: 
                            continue
                        # if operate_list[numtwo] != operate_list[numone]:
                        #     continue
                        # if state_list[numtwo] != state_list[numone]:
                        #     continue
                        if  totalsum_list[numone][numtwo] > threshold or (obj_list[numone] == obj_list[numtwo] and len(alltuple_list[numone]) == 1 ):
                            temp = clusterlist[numone]
                            for index,each in enumerate(clusterlist):
                                if each == temp:
                                    clusterlist[index] = clusterlist[numtwo]
                    #比较方法用w2v+sif+cos
                    elif objcos_simlist_oral[numone][numtwo] < 0.999 and objcos_simlist_oral[numone][numtwo] !=0:
                        continue
                    # elif obj_list[numtwo] != obj_list[numone]:
                    #     continue
                    else:
                    #考虑满足条件出现数字的情况
                        # if bool(re.search('[1-9]', condition_list[numtwo])) or bool(re.search('[1-9]', condition_list[numone])):
                        #     if "".join(list(filter(str.isdigit, condition_list[numtwo]))) != "".join(list(filter(str.isdigit, condition_list[numone]))):
                        #         continue
                        if condition_list[numtwo] != condition_list[numone]:
                            continue
                        #考虑工具出现英文的情况
                        if bool(re.search('[a-zA-Z]', tool_list[numtwo])) or bool(re.search('[a-zA-Z]', tool_list[numone])):
                            if re.sub("[\u4e00-\u9fa5\0-9\,\。]", "", tool_list[numtwo]) != re.sub("[\u4e00-\u9fa5\0-9\,\。]", "", tool_list[numone]):
                                continue
                        elif tool_list[numtwo] != tool_list[numone]:
                            continue 
                        if operatecos_simlist_oral[numone][numtwo] < 0.999 and operatecos_simlist_oral[numone][numtwo] != 0:
                            continue
                        # if operate_list[numtwo] != operate_list[numone]:
                        #     continue
                        # if state_list[numtwo] != state_list[numone]:
                        #     continue
                        #多加一种因为minmax出错的情况
                        if totalsum_list[numone][numtwo] > threshold or (obj_list[numone] == obj_list[numtwo] and len(alltuple_list[numone]) == 1 ):
                            temp = clusterlist[numone]
                            for index,each in enumerate(clusterlist):
                                if each == temp:
                                    clusterlist[index] = clusterlist[numtwo]
                else:
                    nosameitemnum += 1
        
                # clusternumone = 0
                # clusternumtwo = 0
                # for each in clusterlist:
                #     if each == clusterlist[numone]:
                #         clusternumone += 1
                #     if each == clusterlist[numtwo]:
                #         clusternumone += 1
                # if clusternumone >= clusternumone:
                #     #所有clustertwo的数值要与clusterone一致
                #     for index,each in enumerate(clusterlist):
                #         if each == clusterlist[numtwo]:
                #             clusterlist[index] = clusterlist[numone]
                # else:
                #     for index,each in enumerate(clusterlist):
                #         if each == clusterlist[numone]:
                #             clusterlist[index] = clusterlist[numtwo]
    #直接相似度比较方法
    # for numone,entity in enumerate(entitylist):
    #     for numtwo in range(numone+1,len(entitylist)):
    #         sum_scorellist = []
    #         #对第一个测试对象的元组做操作
    #         for entitydic in entity:
    #             for otherentitydic in entitylist[numtwo]:
    #                 #杰卡德系数计算
    #                 if ("测试对象" not in entitydic) and ("测试对象" not in otherentitydic):
    #                     #2是标记
    #                     resobject = 2
    #                 elif ("测试对象" not in entitydic) or ("测试对象" not in otherentitydic):
    #                     resobject = 0
    #                 #涵盖规则
    #                 elif judge_contains(entitydic["测试对象"],otherentitydic["测试对象"]):
    #                     resobject = 1
    #                 else:
    #                     resobject = jaccard_similarity(entitydic["测试对象"], otherentitydic["测试对象"])
                    # if ("测试操作" not in entitydic) and ("测试操作" not in otherentitydic):
                    #     #2是标记
                    #     resoperate = 2
                    # elif ("测试操作" not in entitydic) or ("测试操作" not in otherentitydic):
                    #     resoperate = 0
                    # else:
                    #     resoperate = jaccard_similarity(entitydic["测试操作"], otherentitydic["测试操作"])
                    # if ("测试状态" not in entitydic) and ("测试状态" not in otherentitydic):
                    #     #2是标记
                    #     resstate = 2
                    # elif ("测试状态" not in entitydic) or ("测试状态" not in otherentitydic):
                    #     resstate = 0
                    # else:
                    #     resstate = jaccard_similarity(entitydic["测试状态"], otherentitydic["测试状态"])
                    # if ("测试工具" not in entitydic) and ("测试工具" not in otherentitydic):
                    #     #2是标记
                    #     restool = 2
                    # elif ("测试工具" not in entitydic) or ("测试工具" not in otherentitydic):
                    #     restool = 0
                    # else:
                    #     restool = jaccard_similarity(entitydic["测试工具"], otherentitydic["测试工具"])
                    # if ("软件特性" not in entitydic) and ("软件特性" not in otherentitydic):
                    #     #2是标记
                    #     restexing = 2
                    # elif ("软件特性" not in entitydic) or ("软件特性" not in otherentitydic):
                    #     restexing = 0
                    # else:
                    #     restexing = jaccard_similarity(entitydic["软件特性"], otherentitydic["软件特性"])
                    # if ("满足条件" not in entitydic) and ("满足条件" not in otherentitydic):
                    #     #2是标记
                    #     rescondition = 2
                    # elif ("满足条件" not in entitydic) or ("满足条件" not in otherentitydic):
                    #     rescondition = 0
                    # else:
                    #     rescondition = jaccard_similarity(entitydic["满足条件"], otherentitydic["满足条件"])
                    
            #         sumlist = []
            #         sumlist.append(resobject)
            #         # sumlist.append(resoperate)
            #         # sumlist.append(resstate)
            #         # sumlist.append(restool)
            #         # sumlist.append(restexing)
            #         # sumlist.append(rescondition)
            #         resnum = 0
            #         sum_score = 0
            #         for res in sumlist:
            #             if res != 2:
            #                 sum_score += res
            #                 resnum += 1
            #         #相似度得分
            #         if resnum == 0:
            #             sum_score = 1
            #         else:
            #             sum_score = float(sum_score)/resnum
            #         sum_scorellist.append(sum_score)
            # for score in sum_scorellist:
            #     if score > 0.7 :
            #         clusternumone = 0
            #         clusternumtwo = 0
            #         for each in clusterlist:
            #             if each == clusterlist[numone]:
            #                 clusternumone += 1
            #             if each == clusterlist[numtwo]:
            #                 clusternumone += 1
            #         if clusternumone >= clusternumone:
            #             #所有clustertwo的数值要与clusterone一致
            #             for index,each in enumerate(clusterlist):
            #                 if each == clusterlist[numtwo]:
            #                     clusterlist[index] = clusterlist[numone]
            #         else:
            #             for index,each in enumerate(clusterlist):
            #                 if each == clusterlist[numone]:
            #                     clusterlist[index] = clusterlist[numtwo]
            #         break
            # testobjectone = entity["测试对象"]
            # testobjecttwo = entitylist[numtwo]["测试对象"]
            # testoperateone = entity["测试操作"]
            # testoperatetwo = entitylist[numtwo]["测试操作"]
            # teststateone = entity["测试状态"]
            # teststatetwo = entitylist[numtwo]["测试状态"]
            # # testpreconone = entity["预置条件对象"]
            # # testprecontwo = entitylist[numtwo]["预置条件对象"]
            # #对测试操作做处理，把无意义词去掉
            # for each in nonsense_operatelist:
            #     if each == testoperateone:
            #         testoperateone = ""
            #     if each == testoperatetwo:
            #         testoperatetwo = ""
            # #word2vec编码，余弦值计算
            # # if testobjectone == "" and testobjecttwo=="":
            # #     #2是标记
            # #     resobject = 2
            # # elif testobjectone == "" or testobjecttwo=="":
            # #     resobject = 0
            # # else:
            # #     resobject = vector_similarity(testobjectone, testobjecttwo)
            # # if testoperateone == "" and testoperatetwo=="":
            # #     resoperate = 2
            # # elif testoperateone == "" or testoperatetwo=="":
            # #     resoperate = 0
            # # else:
            # #     resoperate = vector_similarity(testoperateone, testoperatetwo)

            # # if teststateone == "" and teststatetwo=="":
            # #     resstate = 2
            # # elif teststateone == "" or teststatetwo=="":
            # #     resstate = 0
            # # else:
            # #     resstate = vector_similarity(teststateone, teststatetwo)

            # # if testpreconone == "" and testprecontwo=="":
            # #     resprecon = 2
            # # elif testpreconone == "" or testprecontwo=="":
            # #     resprecon = 0
            # # else:
            # #     resprecon = vector_similarity(testpreconone, testprecontwo)

            # #杰卡德系数计算
            # if testobjectone == "" and testobjecttwo=="":
            #     #2是标记
            #     resobject = 2
            # else:
            #     resobject = jaccard_similarity(testobjectone, testobjecttwo)
            # if testoperateone == "" and testoperatetwo=="":
            #     resoperate = 2
            # else:
            #     resoperate = jaccard_similarity(testoperateone, testoperatetwo)
            # if teststateone == "" and teststatetwo=="":
            #     resstate = 2
            # else:
            #     resstate = jaccard_similarity(teststateone, teststatetwo)
            # # if testpreconone == "" and testpreconone=="":
            # #     resprecon = 2
            # # else:
            # #     resprecon = jaccard_similarity(testpreconone, testprecontwo)
            # sumlist = []
            # sumlist.append(resobject)
            # sumlist.append(resoperate)
            # sumlist.append(resstate)
            # resnum = 0
            # sum_score = 0
            # for res in sumlist:
            #     if res != 2:
            #         sum_score += res
            #         resnum += 1
            # #相似度得分
            # if resnum == 0:
            #     sum_score = 1
            # else:
            #     sum_score = sum_score/resnum
            # if sum_score > 0.9 :
            #     clusternumone = 0
            #     clusternumtwo = 0
            #     for each in clusterlist:
            #         if each == clusterlist[numone]:
            #             clusternumone += 1
            #         if each == clusterlist[numtwo]:
            #             clusternumone += 1
            #     if clusternumone >= clusternumone:
            #         #所有clustertwo的数值要与clusterone一致
            #         for index,each in enumerate(clusterlist):
            #             if each == clusterlist[numtwo]:
            #                 clusterlist[index] = clusterlist[numone]
            #     else:
            #         for index,each in enumerate(clusterlist):
            #             if each == clusterlist[numone]:
            #                 clusterlist[index] = clusterlist[numtwo]
        #初始化冗余列表
        redundancy_list = []
        for each in testrelation_list[k]:
            redundancy_list.append(0)
        if len(testrelation_list[k]) == len(clusterlist):
            #做标注转换
            for numone,clusternumone in enumerate(clusterlist):
                if redundancy_list[numone] == 0:
                    for numtwo in range(numone+1,len(clusterlist)):
                        #如果后面有和前面标签一致的
                        if clusternumone == clusterlist[numtwo]:
                            redundancy_list[numtwo] = 1
            # nmiscore = NMI(testrelation_list[k], clusterlist)
            # nmiscorelist.append(nmiscore)
        else:
            new_clusterlist = []
            flag_num = 0
            #将多元组转换为list
            for num,t in enumerate(tuple_list):
                if flag_num ==  num:
                    #不考虑列表最后一个
                    if (num != len(tuple_list) - 1) and (t == tuple_list[num + 1]):
                        sametuple_list = []
                        for numtwo in range(num ,len(tuple_list)):
                            if tuple_list[numtwo] == t and (numtwo != len(tuple_list) - 1):
                                sametuple_list.append(clusterlist[numtwo])
                            #考虑最后一个是多元组情况
                            elif tuple_list[numtwo] == t and (numtwo == len(tuple_list) - 1):
                                sametuple_list.append(clusterlist[numtwo])
                                new_clusterlist.append(sametuple_list)
                                flag_num = len(tuple_list)
                            else:
                                new_clusterlist.append(sametuple_list)
                                flag_num = numtwo
                                break
                    else:
                        new_clusterlist.append(clusterlist[num])
                        flag_num += 1
            #将多元组包含单个元组情况的冗余解决
            for cluster in new_clusterlist:
                if type(cluster) == list:
                    for num,each in enumerate(new_clusterlist):
                        if type(each) == int and (each in cluster):
                            redundancy_list[num] = 1
            #将多元组包含多个元组情况的冗余解决
            for index,cluster in enumerate(new_clusterlist):
                if type(cluster) == list and (redundancy_list[index] == 0):
                    for indextwo in range(index + 1,len(new_clusterlist)):
                        if type(new_clusterlist[indextwo]) == list :
                            if (set(new_clusterlist[indextwo]) == set(cluster)) or set(new_clusterlist[indextwo]).issubset(set(cluster)):
                                redundancy_list[indextwo] = 1
            #将单元组包含单元组情况的冗余解决
            for index,cluster in enumerate(new_clusterlist):
                if type(cluster) == int and (redundancy_list[index] == 0):
                    for indextwo in range(index + 1,len(new_clusterlist)):
                        if type(new_clusterlist[indextwo]) == int:
                            if cluster == new_clusterlist[indextwo]:
                                redundancy_list[indextwo] = 1
        print(redundancy_list)

    
        # # list转dataframe
        # df = pd.DataFrame(redundancy_list, columns=['relation'])
        
        # # 保存到本地excel
        # df.to_excel("D:/czy/OneDrive/UCAS/VQA/测试推荐/公共数据集构建/data/project_splitdata/project8_redundanctextract.xlsx", index=False)

        #计算指标
        for num,each in enumerate(testrelation_list[k]):
            if each == 0 and redundancy_list[num] == 0:
                unredundancy_TP += 1
            elif each == 0 and redundancy_list[num] == 1:
                unredundancy_FN += 1
                redundancy_FP += 1
            elif each == 1 and redundancy_list[num] == 1:
                redundancy_TP += 1
            elif each == 1 and redundancy_list[num] == 0:
                unredundancy_FP += 1
                redundancy_FN += 1

            
                            
            #这里就不把这个值加入列表中
            # nmiscore = 0


        # predictlist.append(predictlist)
        # if len(clusterlist) == len(entitylist):
        #     labels_original = labels_to_original(clusterlist, entitylist)
        # else:
        #     labels_original = entitylist
        # if len(testrelation_list[k]) == len(entitylist):
        #     ground_truth = labels_to_original(testrelation_list[k], entitylist)
        # else:
        #     ground_truth = entitylist
        # for i in range(len(labels_original)):
        #     print(labels_original[i])
        # for i in range(len(ground_truth)):
        #     print(ground_truth[i])
        # print(" ".join(str(testrelation_list[k])))
        #日志
        with open(log_path, "a",encoding="utf-8") as fout:
            fout.write("********************************\n")
            fout.write("groudtruth:\n")
            for each in entitylist:
                fout.write((str(each)))
                fout.write("\n")
            fout.write(str(testrelation_list[k]))
            
            fout.write("\n")
            # for i in range(len(ground_truth)-1):
            #     fout.write((str(ground_truth[i])))
            #     print(str(ground_truth[i]))
            #     fout.write("\n")
            fout.write("\n")
            fout.write("predict:\n") 
            fout.write(str(redundancy_list))
            fout.write("\n")
            # for i in range(len(labels_original)-1):
            #     fout.write((str(labels_original[i])))
            #     fout.write("\n")
            # fout.write(str(nmiscore))
            fout.write("\n")
# with open("D:/czy/OneDrive/UCAS/VQA/测试推荐/公共数据集构建/data/entityexample.json", "w", encoding="utf-8") as w:
#         json.dump(entitylist, w, ensure_ascii=False, indent=4)
    # print(mergeentity)
redundancy_P = redundancy_TP/(redundancy_TP + redundancy_FP)
redundancy_R = redundancy_TP/(redundancy_TP + redundancy_FN)
redundancy_F1 = 2*redundancy_P*redundancy_R/(redundancy_P + redundancy_R)
unredundancy_P = unredundancy_TP/(unredundancy_TP + unredundancy_FP)
unredundancy_R = unredundancy_TP/(unredundancy_TP + unredundancy_FN)
unredundancy_F1 = 2*unredundancy_P*unredundancy_R/(unredundancy_P + unredundancy_R)
print(redundancy_TP + redundancy_FP)
print("sameitemnum: " + str(sameitemnum)) 
print("nosameitemnum: " + str(nosameitemnum)) 
print("redundancy_P: " + str(redundancy_P))
print("redundancy_R: " + str(redundancy_R))
print("redundancy_F1: " + str(redundancy_F1))
print("unredundancy_P: " + str(unredundancy_P))
print("unredundancy_R: " + str(unredundancy_R))
print("unredundancy_F1: " + str(unredundancy_F1))
# print(mean(nmiscorelist))



