import torch
from torch import nn, Tensor
import numpy as np
import torch.nn.functional as F
import random
import tensorflow as tf


seed=333
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def _estimate_alpha(feature_reps, targets):

    X_T, X = feature_reps, feature_reps.permute(0,2,1)

    # Compute matrix inversion
    c=torch.bmm(X_T, X)
    X_TX_inv = torch.linalg.inv(torch.bmm(X_T, X))
    X_Ty = torch.bmm(X_T, targets.unsqueeze(-1))

    # Compute likely scores
    alpha_hat = torch.bmm(X_TX_inv, X_Ty)
    return alpha_hat


##############proposed model DeLELSTM #########################

class decompose_Explain_LSTM_pertime(torch.jit.ScriptModule):
    __constants__ = ["n_units", "input_dim",'time_depth']

    def __init__(self, input_dim, output_dim, n_units, N_units,time_depth,init_std=0.02):
        super().__init__()
        #tensor LSTM parameter
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_i = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_f = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_o = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_i = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_f = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_o = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.B_j = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.B_i = nn.Parameter(torch.Tensor(input_dim, n_units) * init_std)
        self.B_f = nn.Parameter(torch.Tensor(input_dim, n_units) * init_std)
        self.B_o = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.n_units = n_units
        self.input_dim = input_dim
        self.depth= time_depth
        self.output_dim=output_dim
        self.N_units=N_units


        #standard LSTM parameter
        self.u_j = nn.Parameter(torch.randn(input_dim, N_units) * init_std)
        self.w_j = nn.Parameter(torch.randn(N_units, N_units) * init_std)
        self.b_j = nn.Parameter(torch.zeros(N_units))
        self.w_i = nn.Parameter(torch.randn(N_units, N_units) * init_std)
        self.u_i = nn.Parameter(torch.randn(input_dim, N_units) * init_std)
        self.b_i = nn.Parameter(torch.zeros(N_units))
        self.w_f = nn.Parameter(torch.randn(N_units, N_units) * init_std)
        self.u_f = nn.Parameter(torch.randn(input_dim, N_units) * init_std)
        self.b_f = nn.Parameter(torch.zeros(N_units))
        self.w_o = nn.Parameter(torch.randn(N_units, N_units) * init_std)
        self.u_o = nn.Parameter(torch.randn(input_dim, N_units) * init_std)
        self.b_o = nn.Parameter(torch.zeros(N_units))
        self.w_p = nn.Parameter(torch.randn(N_units, 1))
        self.b_p = nn.Parameter(torch.zeros(1))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if 'B' in name  :
                nn.init.constant_(param.data, 0.01)
            elif 'U' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'W' in name:
                nn.init.orthogonal_(param.data)
            elif 'b' in name:
                nn.init.constant_(param.data, 0.01)
            elif 'u' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'w' in name:
                nn.init.orthogonal_(param.data)

    def forward(self, x):
        #tensorized LSTM initial hidden state
        H_tilda_t = torch.randn(x.shape[0], self.input_dim, self.n_units).cuda()
        C_tilda_t = torch.randn(x.shape[0], self.input_dim, self.n_units).cuda()

        #standard LSTM initial hidden state
        h_tilda_t = torch.randn(x.shape[0],  self.N_units).cuda()
        c_tilda_t = torch.randn(x.shape[0],  self.N_units).cuda()
        unorm_list = torch.jit.annotate(list[Tensor], [])
        pred_list = torch.jit.annotate(list[Tensor], [])


        for t in range(self.depth-1):
            #tensorized LSTM　
            temp=H_tilda_t
            J_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", H_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_j) + self.B_j)

            I_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", H_tilda_t, self.W_i) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_i) + self.B_i)
            F_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", H_tilda_t, self.W_f) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_f) + self.B_f)
            O_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", H_tilda_t, self.W_o) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_o) + self.B_o)

            C_tilda_t = C_tilda_t * F_tilda_t + I_tilda_t * J_tilda_t

            H_tilda_t = (O_tilda_t * torch.tanh(C_tilda_t)) #shape batch, feature, dim

            #standard LSTM
            i_t = torch.sigmoid(((x[:, t, :] @ self.u_i) + (h_tilda_t @ self.w_i) + self.b_i))
            f_t = torch.sigmoid(((x[:, t, :] @ self.u_f) + (h_tilda_t @ self.w_f) + self.b_f))
            o_t = torch.sigmoid(((x[:, t, :] @ self.u_o) + (h_tilda_t @ self.w_o) + self.b_o))
            j_t = torch.tanh(((x[:, t, :] @ self.u_j) + (h_tilda_t @ self.w_j) + self.b_j))
            c_tilda_t = f_t * c_tilda_t + i_t * j_t
            h_tilda_t = o_t * torch.tanh(c_tilda_t)
            diff=H_tilda_t-temp #dynamic change of hidden state
            newH_tilda_t=torch.concat([temp, diff],dim=1)
            unnorm_weight= _estimate_alpha(newH_tilda_t, targets=h_tilda_t) # best solution
            h_tilda_t=torch.bmm(newH_tilda_t.permute(0,2,1),unnorm_weight).squeeze(-1) #approximate the hidden state of standard LSTM


            #prediction
            pred_y=(h_tilda_t @ self.w_p) + self.b_p
            pred_list+=[pred_y]
            unorm_list += [unnorm_weight]
        pred = torch.stack(pred_list).permute(1,0,2)
        unorm = torch.stack(unorm_list)


        return pred, unorm




##############baseline models #########################


###LSTM ####

class normalLSTMpertime(torch.jit.ScriptModule):
    __constants__ = ["n_units", "input_dim"]

    def __init__(self, input_dim, output_dim, n_units, init_std=0.01):
        super().__init__()
        self.U_j = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.W_j = nn.Parameter(torch.randn(n_units, n_units)*init_std)
        self.b_j = nn.Parameter(torch.zeros(n_units))
        self.W_i = nn.Parameter(torch.randn(n_units, n_units)*init_std)
        self.U_i=nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_i = nn.Parameter(torch.zeros(n_units))
        self.W_f = nn.Parameter(torch.randn(n_units, n_units) * init_std)
        self.U_f = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.b_f = nn.Parameter(torch.zeros(n_units))
        self.W_o = nn.Parameter(torch.randn(n_units, n_units) * init_std)
        self.U_o = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.b_o = nn.Parameter(torch.zeros(n_units))
        self.W_p=nn.Parameter(torch.randn(n_units, 1))
        self.b_p= nn.Parameter(torch.zeros(1))
        self.n_units = n_units
        self.input_dim = input_dim

        self.reset_parameters()

    def reset_parameters(self) -> None:
            for name, param in self.named_parameters():
                if 'B' in name:
                    nn.init.constant_(param.data, 0.01)
                elif 'U' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'W' in name:
                    nn.init.orthogonal_(param.data)
                elif 'b' in name:
                    nn.init.constant_(param.data, 0.01)
                elif 'u' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'w' in name:
                    nn.init.orthogonal_(param.data)
    def forward(self, x):
        h_tilda_t = torch.zeros(x.shape[0], self.n_units).cuda()
        c_tilda_t = torch.zeros(x.shape[0], self.n_units).cuda()
        pred_list = torch.jit.annotate(list[Tensor], [])

        for t in range(x.shape[1]-1):

            i_t=torch.sigmoid(((x[:,t,:]@self.U_i)+(h_tilda_t@self.W_i)+self.b_i))
            f_t = torch.sigmoid(((x[:, t, :] @ self.U_f) + (h_tilda_t @ self.W_f) + self.b_f))
            o_t = torch.sigmoid(((x[:, t, :] @ self.U_o) + (h_tilda_t @ self.W_o) + self.b_o))
            j_t=torch.tanh(((x[:,t,:]@self.U_j)+(h_tilda_t@self.W_j)+self.b_j))
            c_tilda_t=f_t*c_tilda_t+i_t*j_t
            h_tilda_t=o_t*torch.tanh(c_tilda_t)

            pred_y = (h_tilda_t @ self.W_p) + self.b_p
            pred_list += [pred_y]

        pred = torch.stack(pred_list).permute(1, 0, 2)



        return pred




###IMV-Tensor LSTM ####
class IMVTensorLSTM_pertime(torch.jit.ScriptModule):
    __constants__ = ["n_units", "input_dim"]

    def __init__(self, input_dim, output_dim, n_units, init_std=0.02):
        super().__init__()
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_i = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_f = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_o = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_i = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_f = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_o = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.b_j = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.b_i = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.b_f = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.b_o = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1) * init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1) * init_std)
        self.F_beta = nn.Linear(2 * n_units, 1)
        self.Phi = nn.Linear(2 * n_units, output_dim)
        self.n_units = n_units
        self.input_dim = input_dim

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():

            if 'b' in name  :
                nn.init.constant_(param.data, 0.01)
            elif 'U' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'W' in name:
                nn.init.orthogonal_(param.data)

    #@torch.jit.script_method
    def forward(self, x):
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).cuda()
        c_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).cuda()
        outputs = torch.jit.annotate(list[Tensor], [])
        pred_list = torch.jit.annotate(list[Tensor], [])
        beta_list = torch.jit.annotate(list[Tensor], [])
        alpha_list = torch.jit.annotate(list[Tensor], [])

        for t in range(x.shape[1]-1):
            outputs += [h_tilda_t]
            # eq 1
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_j) + self.b_j)
            # eq 5
            i_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_i) + self.b_i)
            f_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_f) + self.b_f)
            o_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_o) + self.b_o)
            # eq 6
            c_tilda_t = c_tilda_t * f_tilda_t + i_tilda_t * j_tilda_t
            # eq 7
            h_tilda_t = (o_tilda_t * torch.tanh(c_tilda_t))
            newoutputs = torch.stack(outputs)
            newoutputs = newoutputs.permute(1, 0, 2, 3)
            # eq 8
            alphas = torch.tanh(torch.einsum("btij,ijk->btik", newoutputs, self.F_alpha_n) + self.F_alpha_n_b)
            alphas = torch.exp(alphas)
            alphas = alphas / torch.sum(alphas, dim=1, keepdim=True)
            g_n = torch.sum(alphas * newoutputs, dim=1)
            hg = torch.cat([g_n, h_tilda_t], dim=2)
            mu = self.Phi(hg)
            betas = torch.tanh(self.F_beta(hg))
            betas = torch.exp(betas)
            betas = betas / torch.sum(betas, dim=1, keepdim=True)
            mean = torch.sum(betas * mu, dim=1)
            pred_list += [mean]
            alpha_list+=[alphas]
            beta_list+=[betas]

        pred = torch.stack(pred_list).permute(1, 0, 2)
        alpha_list=torch.cat(alpha_list, dim=1).squeeze(-1)
        beta_list=torch.stack(beta_list).squeeze(-1).permute(1,0,2)

        return pred, alpha_list, beta_list




###IMV-Full LSTM ####
class IMVFullLSTM_pertime(torch.jit.ScriptModule):
    __constants__ = ["n_units", "input_dim"]
    def __init__(self, input_dim, output_dim, n_units, init_std=0.02):
        super().__init__()
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.b_j = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.W_i = nn.Linear(input_dim*(n_units+1), input_dim*n_units)
        self.W_f = nn.Linear(input_dim*(n_units+1), input_dim*n_units)
        self.W_o = nn.Linear(input_dim*(n_units+1), input_dim*n_units)
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1)*init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1)*init_std)
        self.F_beta = nn.Linear(2*n_units, 1)
        self.Phi = nn.Linear(2*n_units, output_dim)
        self.n_units = n_units
        self.input_dim = input_dim

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if 'b' in name:
                nn.init.constant_(param.data, 0.01)
            elif 'U' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'W' in name:
                nn.init.orthogonal_(param.data)

    @torch.jit.script_method
    def forward(self, x):
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).cuda()
        c_t = torch.zeros(x.shape[0], self.input_dim*self.n_units).cuda()
        outputs = torch.jit.annotate(list[Tensor], [])
        pred_list = torch.jit.annotate(list[Tensor], [])
        beta_list = torch.jit.annotate(list[Tensor], [])
        alpha_list = torch.jit.annotate(list[Tensor], [])
        for t in range(x.shape[1]-1):
            outputs += [h_tilda_t]
            # eq 1
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_j) + self.b_j)
            inp =  torch.cat([x[:, t, :], h_tilda_t.reshape(h_tilda_t.shape[0], -1)], dim=1)
            # eq 2
            i_t = torch.sigmoid(self.W_i(inp))
            f_t = torch.sigmoid(self.W_f(inp))
            o_t = torch.sigmoid(self.W_o(inp))
            # eq 3
            c_t = c_t*f_t + i_t*j_tilda_t.reshape(j_tilda_t.shape[0], -1)
            # eq 4
            h_tilda_t = (o_t*torch.tanh(c_t)).reshape(h_tilda_t.shape[0], self.input_dim, self.n_units)

            newoutputs = torch.stack(outputs)
            newoutputs = newoutputs.permute(1, 0, 2, 3)
        # eq 8
            alphas = torch.tanh(torch.einsum("btij,ijk->btik", newoutputs, self.F_alpha_n) +self.F_alpha_n_b)
            alphas = torch.exp(alphas)
            alphas = alphas/torch.sum(alphas, dim=1, keepdim=True)
            g_n = torch.sum(alphas*newoutputs, dim=1)
            hg = torch.cat([g_n, h_tilda_t], dim=2)
            mu = self.Phi(hg)
            betas = torch.tanh(self.F_beta(hg))
            betas = torch.exp(betas)
            betas = betas/torch.sum(betas, dim=1, keepdim=True)
            mean = torch.sum(betas*mu, dim=1)
            pred_list += [mean]
            alpha_list += [alphas]
            beta_list += [betas]

        pred = torch.stack(pred_list).permute(1, 0, 2)
        alpha_list = torch.cat(alpha_list, dim=1).squeeze(-1)
        beta_list = torch.stack(beta_list).squeeze(-1).permute(1, 0, 2)

        return pred, alpha_list, beta_list




###RETAIN LSTM ####
class Retain_pertime(nn.Module):
    def __init__(self, inputDimSize, embDimSize, alphaHiddenDimSize, betaHiddenDimSize, outputDimSize, keep_prob=1.0):
        super(Retain_pertime, self).__init__()
        self.inputDimSize = inputDimSize
        self.embDimSize = embDimSize
        self.alphaHiddenDimSize = alphaHiddenDimSize
        self.betaHiddenDimSize = betaHiddenDimSize
        self.outputDimSize = outputDimSize
        self.keep_prob = keep_prob

        self.embedding = nn.Linear(self.inputDimSize, self.embDimSize)
        self.dropout = nn.Dropout(self.keep_prob)
        self.gru_alpha = nn.GRU(self.embDimSize, self.alphaHiddenDimSize)
        self.gru_beta = nn.GRU(self.embDimSize, self.betaHiddenDimSize)
        self.alpha_att = nn.Linear(self.alphaHiddenDimSize, 1)
        self.beta_att = nn.Linear(self.betaHiddenDimSize, self.embDimSize)
        self.out = nn.Linear(self.embDimSize, self.outputDimSize)

    def initHidden_alpha(self, batch_size):
        return torch.zeros(1, batch_size, self.alphaHiddenDimSize, device=torch.device('cuda:0'))

    def initHidden_beta(self, batch_size):
        return torch.zeros(1, batch_size, self.betaHiddenDimSize, device=torch.device('cuda:0'))


    def attentionStep(self, h_a, h_b, att_timesteps):
        reverse_emb_t = self.emb[:att_timesteps].flip(dims=[0])
        reverse_h_a = self.gru_alpha(reverse_emb_t, h_a)[0].flip(dims=[0]) * 0.5
        reverse_h_b = self.gru_beta(reverse_emb_t, h_b)[0].flip(dims=[0]) * 0.5

        preAlpha = self.alpha_att(reverse_h_a)
        preAlpha = torch.squeeze(preAlpha, dim=2)
        alpha = torch.transpose(F.softmax(torch.transpose(preAlpha, 0, 1)), 0, 1)
        beta = torch.tanh(self.beta_att(reverse_h_b))

        c_t = torch.sum((alpha.unsqueeze(2) * beta * self.emb[:att_timesteps]), dim=0)
        return c_t, alpha, beta

    def forward(self, x):
        temp=x.permute(1,0,2)
        first_h_a = self.initHidden_alpha(temp.shape[1])
        first_h_b = self.initHidden_beta(temp.shape[1])

        self.emb = self.embedding(temp)
        w_emb=self.embedding.weight.data
        if self.keep_prob < 1:
            self.emb = self.dropout(self.emb)

        count = np.arange(temp.shape[0]-1)+1
        pred_list = torch.jit.annotate(list[Tensor], [])
        weight_list = torch.jit.annotate(list[Tensor], [])
        for i, att_timesteps in enumerate(count):

             c_t, alpha, beta = self.attentionStep(first_h_a, first_h_b, att_timesteps)
             y_hat=self.out(c_t)
             w_out=self.out.weight.data
             pred_list += [y_hat]

             #compute variable importance for each time prediction
             new_beta = beta.permute(1, 0, 2).unsqueeze(-1)
             d = torch.mul(new_beta, w_emb)
             e = torch.matmul(w_out, d) .squeeze(2)
             new_alpha = alpha.permute(1, 0).unsqueeze(-1)
             f = torch.mul(new_alpha, e)
             f=torch.mul(f,x[:,:att_timesteps,:])
             g = torch.mean(f, dim=1)
             weight_list +=[g]


        pred = torch.stack(pred_list).permute(1, 0, 2)
        weight_list=torch.stack(weight_list).permute(1,0,2)
        return pred, weight_list




