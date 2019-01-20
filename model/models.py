import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models


def model_1():
    model = models.vgg16(pretrained=True)
    model.classifier = nn.Sequential(
                                        nn.Linear(25088,5005),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(5005,5005),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(5005,5005)
                                    )
    return model

class model_2(nn.Module):
    def __init__(self):
        super(model_2, self).__init__()

        self.d = feature_dim
        self.char_emb = nn.Embedding(char_vocab_size, char_emb_size, padding_idx=1)
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)
        self.word_emb = nn.Embedding.from_pretrained(pretrained_word_emb, freeze=True)
        
        
        self.dropout = nn.Dropout(p=0.2)
                
        # char CNN
        self.char_conv1 = nn.Conv2d(1 ,self.d ,(char_emb_size,3))
        self.char_conv2 = nn.Conv2d(1 ,self.d ,(char_emb_size,5))
        self.char_conv3 = nn.Conv2d(1 ,self.d ,(char_emb_size,7))
        
        # highway network
        self.highway_num_layers = 2
        highway_size = 2 * self.d
        self.highway_1 = nn.ModuleList([nn.Linear(highway_size, highway_size) for i in range(self.highway_num_layers)])
        self.highway_2 = nn.ModuleList([nn.Linear(highway_size, highway_size) for i in range(self.highway_num_layers)])

        # contextual embedding
        self.context_lstm = nn.LSTM(input_size=2 * self.d,
                                    hidden_size=self.d,
                                    batch_first=True,
                                    bidirectional=True)

        # attention flow layer
        self.similarity1 = nn.Linear(2*self.d,1)
        self.similarity2 = nn.Linear(2*self.d,1)
        self.similarity3 = nn.Linear(2*self.d,1)
        # modeling layer
        self.modeling_lstm = nn.LSTM(input_size=8 * self.d,
                                    hidden_size=self.d,
                                    num_layers=2,
                                    batch_first=True,
                                    bidirectional=True,
                                    dropout=0.2)

        # output layer
        self.linear_p1 = nn.Linear(10 * self.d, 1)
        self.linear_p2 = nn.Linear(10 * self.d, 1)
        self.output_lstm = nn.LSTM(input_size=2 * self.d,
                                    hidden_size=self.d,
                                    batch_first=True,
                                    bidirectional=True)

    def init_LSTM(self,batch_size,hidden_size,num_layers):
        h0 = Variable(torch.zeros(batch_size,hidden_size,num_layers)).cuda()
        c0 = Variable(torch.zeros(batch_size,hidden_size,num_layers)).cuda()
        return h0, c0


    def init_BiLSTM(self, batch_size, hidden_size, num_layers):
        h0 = Variable(torch.zeros(num_layers * 2, batch_size, hidden_size)).cuda()
        c0 = Variable(torch.zeros(num_layers * 2, batch_size, hidden_size)).cuda()
        return h0, c0

    def forward(self,batch):

        def char_embedding(data):
            x = self.char_emb(data)  # (batch, seq_len, char_len, char_emb_size)
            size = x.size()
            x = x.view(size[0] * size[1], size[3], size[2]).unsqueeze(1)  # (batch*seq,1,char_emb_size,char_len)
            out = torch.cat( ( self.char_conv1(self.dropout(x)),self.char_conv2(self.dropout(x)),self.char_conv3(self.dropout(x))),dim=3).squeeze(2)  # (batch*seq,100,1,char_conv_dim) -> squeeze(2)
            out = F.max_pool1d(out, (out.size(2))).squeeze(2).view(size[0],size[1],self.d)
            return out


        def highway_network(x):  # (batch, seq_len, 2*100)
            for layer in range(self.highway_num_layers):
                gate = torch.sigmoid(self.highway_1[layer](x))
                linear = F.relu(self.highway_2[layer](x))

                x = gate * linear + (1 - gate) * x
            return x


        def attention_flow(H,U): # (batch, t ,2d), (batch, j, 2d)
            batch,t,d = H.size()
            j = U.size(1)
            S_size = (batch, t, j)
            S = self.similarity1(H).expand(S_size)+\
                self.similarity2(U).permute(0,2,1).expand(S_size)+\
                self.similarity3(H.unsqueeze(2).expand(batch,t,j,d)*U.unsqueeze(1).expand(batch,t,j,d)).squeeze() # (batch,t,j,2d) elementwise
                
         
            a = F.softmax(S, dim=2)  # j 열의 모든 t를 합치면 1 , (batch, t, j)
            U = torch.bmm(a, U)  # (batch,t,2d)

            b = torch.max(S, 2)[0]  # (batch, t)
            b = F.softmax(b,dim=1).unsqueeze(1)  # (batch,1,t)
            H_tilde = torch.bmm(b,H)  # (batch,1,2d)
            H_tilde = H_tilde.expand(H.size())  # (batch,t,2d)

            G = torch.cat((H, U, H * U, H * H_tilde), 2)  # (batch,t,8d) --- 2d 4개 concat
            return G

        ############### forward ##########
        # input :  (c_char , q_char , c_word, q_word)
#        h0, c0 = self.init_BiLSTM(batch_size=c_char.size(0), hidden_size=100, num_layers=2)
        # 1.char embedding layer
        c = char_embedding(batch.c_char)  # (batch, seq_len, word_len, 5) -> (batch, seq_len, d)
        q = char_embedding(batch.q_char)


        # 2.word embedding layer
        c = torch.cat((c, self.word_emb(batch.c_word)), 2)
        q = torch.cat((q, self.word_emb(batch.q_word)), 2)
        c = highway_network(c)  # (batch, seq_len, d) # 여기서 d는 200임, because : word embedding 이 옵션이기 때문
        q = highway_network(q)
        
        # 3.contextual embedding layer
        self.context_lstm.flatten_parameters()
        c = self.context_lstm(self.dropout(c))[0]  # (batch_size, context_seq_len, 2*d)
        q = self.context_lstm(self.dropout(q))[0]  # (batch_size, query_seq_len, 2*d)
        
        # 4.attention flow layer
        G = attention_flow(c,q)

        # 5. modeling layer
        self.modeling_lstm.flatten_parameters()
        M1 = self.modeling_lstm(self.dropout(G))[0]
        # 6. output layer
        p1 = self.linear_p1(self.dropout(torch.cat((G, M1), 2))).squeeze(1)
        self.output_lstm.flatten_parameters()
        M2 = self.output_lstm(self.dropout(M1))[0]
        p2 = self.linear_p2(self.dropout(torch.cat((G, M2), 2))).squeeze(1)
        return p1, p2