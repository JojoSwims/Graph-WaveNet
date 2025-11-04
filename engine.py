import torch.optim as optim
from model import *
import util
class trainer():
    def __init__(
        self,
        scaler,
        in_dim,
        seq_length,
        num_nodes,
        nhid,
        dropout,
        lrate,
        wdecay,
        device,
        supports,
        gcn_bool,
        addaptadj,
        aptinit,
        loss_fn='mae',
        use_lr_scheduler=False,
        lr_t_max=100,
        lr_eta_min=1e-5,
    ):
        self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.lr_scheduler = None
        if use_lr_scheduler:
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=lr_t_max,
                eta_min=lr_eta_min,
            )
        self.loss = self._select_loss(loss_fn)
        self.scaler = scaler
        self.clip = 5

    def _select_loss(self, loss_name):
        losses = {
            'mae': util.masked_mae,
            'huber': util.masked_huber,
        }
        if loss_name not in losses:
            raise ValueError("Unsupported loss function '{}'. Available options are: {}".format(loss_name, ', '.join(losses.keys())))
        return losses[loss_name]

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def step_lr_scheduler(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']
