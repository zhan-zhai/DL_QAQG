import numpy as np
import time

import torch
from torch.nn import functional as F
import torch.nn as nn
import torch.utils.data as Data
from torch import optim

from models.QA import QuestionAnswerer
from models.QG import QuestionGenerator


class DualTaskQAGSystem():
    def __init__(self, max_len, pair_len, feat_dim, max_iter,
                 QA_d_model, QA_n_heads, QA_num_layers, QA_dim_feedforward, QA_dropout, QA_act, QA_epochs, QA_lr, QA_weight_l2,
                 QG_d_model, QG_n_heads, QG_num_layers, QG_dim_feedforward, QG_dropout, QG_act, QG_epochs, QG_lr, QG_weight_l2,
                 batch_size, device, shuffle=True, num_workers=4):
        
        self.max_len = max_len # 最长句子长度
        self.pair_len = pair_len # 元组长度
        self.feat_dim = feat_dim # 嵌入向量维度
        self.max_iter = max_iter # 最多训练几轮
        
        self.QA_d_model = QA_d_model # Transformer隐藏参数
        self.QA_n_heads = QA_n_heads
        self.QA_num_layers = QA_num_layers # encoder数量
        self.QA_dim_feedforward = QA_dim_feedforward
        self.QA_dropout = QA_dropout
        self.QA_act = QA_act
        self.QA_epochs = QA_epochs
        self.QA_lr = QA_lr
        self.QA_weight_l2 = QA_weight_l2
        
        self.QG_d_model = QA_d_model
        self.QG_n_heads = QA_n_heads
        self.QG_num_layers = QA_num_layers
        self.QG_dim_feedforward = QA_dim_feedforward
        self.QG_dropout = QG_dropout
        self.QG_act = QG_act
        self.QG_epochs = QG_epochs
        self.QG_lr = QG_lr
        self.QG_weight_l2 = QG_weight_l2
        
        self.batch_size = batch_size
        self.device = device
        
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        self.QA_model = QuestionAnswerer(feat_dim, 
                                         max_len, 
                                         QA_d_model, 
                                         QA_n_heads, 
                                         QA_num_layers, 
                                         QA_dim_feedforward, 
                                         QA_dropout, 
                                         QA_act).to(device)
        self.QG_model = QuestionGenerator(feat_dim, 
                                          pair_len, 
                                          max_len,
                                          QG_d_model, 
                                          QG_n_heads, 
                                          QG_num_layers, 
                                          QG_dim_feedforward, 
                                          QG_dropout, 
                                          QG_act).to(device)
        
    
    @staticmethod
    def _tensor2numpy(data):
        return torch.Tensor.cpu(data.detach()).numpy() 

    
    def batch_load(self, qvecs, pvecs):
        dataset = Data.TensorDataset(qvecs, pvecs)
        loader = Data.DataLoader(dataset=dataset, batch_size=self.batch_size, 
                                 shuffle=self.shuffle, num_workers=self.num_workers)
        batch_num = -1
        for step, (batch_x, batch_y) in enumerate(loader):
            print('Step:%d' % step, ' question vectors shape:', batch_x.shape, ', pair vectors shape:', batch_y.shape)   
            batch_num = step + 1 if step > batch_num else batch_num
        print('%d batches in total' % batch_num)
        self.loader = loader
        self.batch_num = batch_num
        return loader, batch_num
    
    
    def preprocess_data(self, qvecs, pvecs):
        print("Question vectors shape:", qvecs.shape)
        pvecs = torch.tensor(pvecs, dtype=torch.float32)
        print("Entity-Relation pairs shape:", pvecs.shape)
        
        loader, batch_num = self.batch_load(qvecs, pvecs)
        return loader, batch_num
    
    
    def fit_QA(self):
        log_loss = []
        criterion = nn.MSELoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.QA_model.parameters()), 
                               lr=self.QA_lr, 
                               weight_decay=self.QA_weight_l2)
        min_loss = np.inf
        
        print("-- trainning QA model --")
        for epoch in range(self.QA_epochs):
            t_start = time.time()
            running_loss = 0
            for step, (qvecs, pvecs) in enumerate(self.loader):
                question = qvecs.to(self.device)
                ans = pvecs.to(self.device)
                
                ans_en, ans_re = self.QA_model(question)
                ans_en = ans_en.reshape(ans_en.shape[0], 1, ans_en.shape[1])
                ans_re = ans_re.reshape(ans_re.shape[0], 1, ans_re.shape[1])
                ans_hat = torch.cat([ans_en, ans_re], 1)
                question_hat = self.QG_model(ans_hat)
                
                loss = criterion(question, question_hat)
                
                # 更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            if running_loss < min_loss:
                min_loss = running_loss
                torch.save(self, './checkpoints/best_QA.net')
                
            running_loss /= self.batch_num
            log_loss.append(running_loss)
            t_end = time.time()
            print('epoch=%d, loss=%.6f, time=%.2fs per epoch' \
               % (epoch+1, running_loss, t_end-t_start), end='\n')
            torch.cuda.empty_cache()
        return log_loss
    
    
    def fit_QG(self):
        log_loss_en, log_loss_re = [], []
        criterion = nn.MSELoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.QG_model.parameters()), 
                               lr=self.QG_lr, 
                               weight_decay=self.QG_weight_l2)
        min_loss = np.inf
        
        print("-- trainning QG model --")
        for epoch in range(self.QG_epochs):
            t_start = time.time()
            running_loss_en, running_loss_re = 0, 0
            for step, (qvecs, pvecs) in enumerate(self.loader):
                question = qvecs.to(self.device)
                ans = pvecs.to(self.device)
                ans_en = ans[:, 0, :]
                ans_re = ans[:, 1, :]
                
                question_hat = self.QG_model(ans)
                ans_en_hat, ans_re_hat = self.QA_model(question_hat)
                
                loss_en = criterion(ans_en, ans_en_hat)
                loss_re = criterion(ans_re, ans_re_hat)
                loss = loss_en + loss_re
                
                # 更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss_en += loss_en.item()
                running_loss_re += loss_re.item()
                running_loss = running_loss_en + running_loss_re
                
            if running_loss < min_loss:
                min_loss = running_loss
                torch.save(self, './checkpoints/best_QG.net')
                
            running_loss_en /= self.batch_num
            log_loss_en.append(running_loss_en)
            running_loss_re /= self.batch_num
            log_loss_re.append(running_loss_re)
            running_loss /= self.batch_num
            
            t_end = time.time()
            print('epoch=%d, loss=%.6f, time=%.2fs per epoch' \
               % (epoch+1, running_loss, t_end-t_start), end='\n')
            torch.cuda.empty_cache()
        return log_loss_en, log_loss_re
    
    
    def predict_answer(self, question):
        ans_en, ans_re = self.QA_model(question.to(self.device))
        return ans_en, ans_re
    
    
    def generate_question(self, pair):
        que = self.QG_model(pair.to(self.device))
        return que
    
    
    def load_checkpoints(self, path="./checkpoints/"):
        self.QA_model = torch.load(path + "best_QA.net")
        self.QG_model = torch.load(path + "best_QG.net")
    
    
    def train(self):
        loss_QA, loss_QG_en, loss_QG_re = [], [], []
        for it in range(self.max_iter):
            print("--------------- Iteration %d ---------------" % (it + 1))
            self.QA_model.restore_parameters()
            self.QG_model.freeze_parameters()
            log_QA = self.fit_QA()
            loss_QA += log_QA
            
            self.QG_model.restore_parameters()
            self.QA_model.freeze_parameters()
            log_en, log_re = self.fit_QG()
            loss_QG_en += log_en
            loss_QG_re += log_re
            
        return loss_QA, loss_QG_en, loss_QG_re