'''
Created on 2020年9月7日

@author: xiaoyw
'''
import numpy as np
#import numpy.matlib
import pandas as pd
import warnings


class AHP:
    def __init__(self, criteria, factors):
        self.RI = (0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49)
        self.criteria = criteria               #准则
        self.factors = factors                 #因素
        self.num_criteria = criteria.shape[0]
        self.num_factors = factors[0].shape[0]

    def cal_weights(self, input_matrix):
        input_matrix = np.array(input_matrix)
        n, n1 = input_matrix.shape
        
        assert n == n1, '不是一个方阵'
        for i in range(n):
            for j in range(n):
                if np.abs(input_matrix[i, j] * input_matrix[j, i] - 1) > 1e-7:
                    raise ValueError('不是反互对称矩阵')

        eigenvalues, eigenvectors = np.linalg.eig(input_matrix)

        max_idx = np.argmax(eigenvalues)
        max_eigen = eigenvalues[max_idx].real
        eigen = eigenvectors[:, max_idx].real
        eigen = eigen / eigen.sum()

        if n > 9:
            CR = None
            warnings.warn('无法判断一致性')
        else:
            CI = (max_eigen - n) / (n - 1)
            CR = CI / self.RI[n]
        return max_eigen, CR, eigen

    def run(self):
        max_eigen, CR, criteria_eigen = self.cal_weights(self.criteria)
        print('准则层：最大特征值{:<5f},CR={:<5f},检验{}通过'.format(max_eigen, CR, '' if CR < 0.1 else '不'))
        print('准则层权重={}\n'.format(criteria_eigen))

        max_eigen_list, CR_list, eigen_list = [], [], []
        k = 1
        for i in self.factors:
            max_eigen, CR, eigen = self.cal_weights(i)
            max_eigen_list.append(max_eigen)
            CR_list.append(CR)
            eigen_list.append(eigen)
            print('准则 {} 因素层：最大特征值{:<5f},CR={:<5f},检验{}通过'.format(k,max_eigen, CR, '' if CR < 0.1 else '不'))
            print('因素层权重={}\n'.format(eigen))

            k = k + 1

        return criteria_eigen ,eigen_list

#模糊综合评价法(FCE)
def fuzzy_eval(criteria, eigen,df,score):
    print('单因素模糊综合评价：{}\n'.format(df))
    #把单因素评价数据，拆解到5个准则中
    v1 = df.iloc[0:2,:].values
    v2 = df.iloc[2:5,:].values
    v3 = df.iloc[5:9,:].values
    v4 = df.iloc[9:12,:].values
    v5 = df.iloc[12:16,:].values
   
    vv = [v1,v2,v3,v4,v5]
   
    val = []
    num = len(eigen)
    for i in range(num):
        v = np.dot(np.array(eigen[i]),vv[i])
        print('准则{} , 矩阵积为：{}'.format(i+1,v))
        val.append(v)
       
    # 目标层
    obj = np.dot(criteria, np.array(val))
    print('目标层模糊综合评价：{}\n'.format(obj))
    #综合评分
    eval = np.dot(np.array(obj),np.array(score).T)
    print('综合评价：{}'.format(eval*100))
       
   
    return obj,eval
#从Excel读取数据
def get_DataFromExcel():
    df = pd.read_excel('FCE1.xlsx') 

    return df

# 随机生产16*5评价矩阵，每行打分值合计为1
def get_DataFromRandom():
    #v0 = np.random.randint(0,10,(16,5))
    # 构造随机矩阵
    #v0 = np.round(np.random.dirichlet(np.ones(5),size=16), decimals=1)
    v0 = []
    for i in range(16):
        v0.append(get_rand(1,9,5))
        i = i+ 1
        #print(i)
    #创建df，把numpy转换为Dataframe
    df = pd.DataFrame(v0,columns=('优秀','良好','一般','较差','非常差'))
        
    return df
#取5个随机数，合计为1
def get_rand(maxRange=1,randrange=9,num=5):
    k = []
    for i in range(num):
        k0 = np.random.randint(randrange)
        k.append(k0)
        
    k1 = sum(k)
    #print(k)
    k0 = 0.0
    for i in range(num-1):
        k[i] = round(k[i]/k1*maxRange,1)
        k0 = k0 + k[i]
    
    k[num-1] = round(maxRange- k0,1)    
        
    #print(k)
    
    return k
        


def main():
    #量化评语（优秀、    良好、    一般、    较差、   非常差）
    score = [1,0.8,0.6,0.4,0.2]
    
    df = get_DataFromExcel()
    # 准则判断矩阵（重要性矩阵）
    criteria = np.array([[1, 7, 5, 7, 5],
                         [1 / 7, 1, 2, 3, 3],
                         [1 / 5, 1 / 2, 1,  2,  3],
                         [1 / 7, 1 / 3, 1 / 2, 1, 3],
                         [1 / 5, 1 / 3, 1 / 3, 1 / 3, 1]])

    # 对每个准则，准则下的因素判断矩阵
    b1 = np.array([[1, 5], [1 / 5, 1]])
    b2 = np.array([[1, 2, 5], [1 / 2, 1, 2], [1 / 5, 1 / 2, 1]])
    b3 = np.array([[1, 5, 6, 8], [1 / 5, 1 ,2, 7], [1 / 6, 1 / 2, 1 ,4],[1 / 8, 1 / 7, 1 / 4, 1]])
    b4 = np.array([[1, 3, 4], [1 / 3, 1, 1], [1 / 4, 1, 1]])
    b5 = np.array([[1, 4, 5, 5], [1 / 4, 1, 2, 4], [1 /5 , 1 / 2, 1, 2], [1 / 5,1 /4,1 / 2, 1]])

    b = [b1, b2, b3, b4, b5]
    a,c = AHP(criteria, b).run()
   
    fuzzy_eval(a,c,df,score)

# 自动构造训练数据
def main2():
    #量化评语（优秀、    良好、    一般、    较差、   非常差）
    score = [1,0.8,0.6,0.4,0.2]
    
    #eval = np.dot(v0,np.array(score).T)
    # 准则判断矩阵（重要性矩阵）
    criteria = np.array([[1, 7, 5, 7, 5],
                         [1 / 7, 1, 2, 3, 3],
                         [1 / 5, 1 / 2, 1,  2,  3],
                         [1 / 7, 1 / 3, 1 / 2, 1, 3],
                         [1 / 5, 1 / 3, 1 / 3, 1 / 3, 1]])

    # 对每个准则，准则下的因素判断矩阵
    b1 = np.array([[1, 5], [1 / 5, 1]])
    b2 = np.array([[1, 2, 5], [1 / 2, 1, 2], [1 / 5, 1 / 2, 1]])
    b3 = np.array([[1, 5, 6, 8], [1 / 5, 1 ,2, 7], [1 / 6, 1 / 2, 1 ,4],[1 / 8, 1 / 7, 1 / 4, 1]])
    b4 = np.array([[1, 3, 4], [1 / 3, 1, 1], [1 / 4, 1, 1]])
    b5 = np.array([[1, 4, 5, 5], [1 / 4, 1, 2, 4], [1 /5 , 1 / 2, 1, 2], [1 / 5,1 /4,1 / 2, 1]])

    b = [b1, b2, b3, b4, b5]
    a,c = AHP(criteria, b).run()
    
    train_df = pd.DataFrame(columns=('总评价','持证上岗','组织赋能','合规管理','员工赋能','绩效管理','人员专业化 ','人员持证率',
                                     '团队建设','领导赋能下属 ','提高员工动力 ','制度化率','领导重视 ','合规监督',
                                     '持续改进','辅导与培训',' 激励','明确责权利','绩效目标','绩效沟通','诊断提升 ','绩效评价'))
    for i in range(10):
        df = get_DataFromRandom()
        v = df.values;
        row = []
        #评分矩阵分值
        v1 = np.dot(np.array(v),np.array(score).T)
        obj,eval=fuzzy_eval(a,c,df,score)
    
        row.append(eval)
        row.extend(obj.tolist())
        row.extend(v1.tolist())
        
        train_df.loc[len(train_df)]=row
        '''
        # 合并方式
        df_new = pd.DataFrame([row],columns=('总评价','持证上岗','组织赋能','合规管理','员工赋能','绩效管理','人员专业化 ','人员持证率',
                                         '团队建设','领导赋能下属 ','提高员工动力 ','制度化率','领导重视 ','合规监督',
                                         '持续改进','辅导与培训',' 激励','明确责权利','绩效目标','绩效沟通','诊断提升 ','绩效评价'))   
        print(df_new)
        
        train_df = pd.concat([train_df,df_new],ignore_index=True)
        '''
    
    print(train_df)
    train_df.to_excel("BP_random.xlsx")   
    
    
if __name__ == '__main__':
    #get_rand(1,9,5)
    #print(get_DataFromRandom())
    #main()
    main2()
    