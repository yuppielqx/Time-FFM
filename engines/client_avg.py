import os
import torch
import warnings
import numpy as np
from copy import deepcopy
from models.model import ClientEncoder, ClientHead
warnings.filterwarnings('ignore')

from utils.metrics import metric

class Engine_Forecasting(object):
    def __init__(self, args, model, client_encoder, client_head=None):
        self.args = args
        self.data_id = args.data_id + '_' + str(args.seq_len) + '_'
        self.info = [self.data_id, args.seq_len, args.stride]
        self.criterion = torch.nn.MSELoss()
        self.train_loaders = None
        self.valid_loaders = None
        self.test_loaders = []
        self.train_batches = 0
        self.train_iter = None
        self.seen_batches = 0
        self.word_embeddings = model.backbone.get_input_embeddings().weight.clone().detach().requires_grad_(True)

        if client_encoder is None:
            self.client_encoder = ClientEncoder(args, self.word_embeddings).to(args.device)
        else:
            self.client_encoder = client_encoder

        if client_head is None:
            self.client_head = None
        else:
            self.client_head = client_head
            self.optimizer_client_head = torch.optim.AdamW(self.client_head.parameters(), lr=self.args.learning_rate,
                                                           weight_decay=self.args.weight_decay)
            self.scheduler_c2 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_client_head, T_max=20,
                                                                           eta_min=1e-6)
            self._print_trainable_parameters(self.client_head)

        self._print_trainable_parameters(self.client_encoder)

        self.optimizer_client_encoder = torch.optim.AdamW(self.client_encoder.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.optimizer_server = torch.optim.AdamW(model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.scheduler_c1 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_client_encoder, T_max=20, eta_min=1e-6)
        self.scheduler_s = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_server, T_max=20, eta_min=1e-6)

    def _print_trainable_parameters(self, model):
        freeze = 0
        trainable = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable += param.nelement()
            else:
                freeze += param.nelement()
        self.args.logger.info('Trainable Params: {}, All Params: {}, Percent: {}'.format(
                              trainable, freeze + trainable, trainable / (freeze + trainable)))

    def _client_train_head(self, x_enc, batch_x, batch_y, mean, std, channels):
        self.client_head.train()
        self.optimizer_client_head.zero_grad()
        outputs = self.client_head(x_enc, mean, std, channels)
        f_dim = -1 if self.args.features == 'MS' else 0
        if self.args.max_backcast_len == 0:
            outputs = outputs[:, :self.args.pred_len, f_dim:]
            batch_y = batch_y[..., f_dim:]
        elif self.args.max_forecast_len == 0:
            outputs = outputs[:, self.args.max_backcast_len - self.args.seq_len:, f_dim:]
            batch_y = batch_x[..., f_dim:]
        else:
            outputs = outputs[:, self.args.max_backcast_len - self.args.seq_len:
                                 self.args.max_backcast_len + self.args.pred_len, f_dim:]
            batch_y = torch.cat((batch_x, batch_y), dim=1)  # bs, seq_len+pred_len, channels
            batch_y = batch_y[..., f_dim:]

        loss = self.criterion(outputs, batch_y)
        loss.backward()
        x_enc_hat = x_enc.grad.clone().detach()

        if self.args.clip != 0:
            torch.nn.utils.clip_grad_norm_(self.client_head.parameters(), self.args.clip)
        self.optimizer_client_head.step()

        return x_enc_hat, loss.item()

    def train_batch_split(self, model, embed_state):
        if self.seen_batches % self.train_batches == 0:
            self.train_iter = iter(self.train_loaders)
        batch = next(self.train_iter)
        self.client_encoder.train()
        model.train()

        batch_x, batch_y = batch
        batch_x = batch_x.float().to(self.args.device)  # batch_size * seq_len * channels
        batch_y = batch_y.float().to(self.args.device)  # batch_size * len_pred * channels

        b, t, n = batch_x.shape
        mask = torch.rand((b, t, n)).to(self.args.device)
        mask[mask < self.args.mask_rate] = 0
        mask[mask >= self.args.mask_rate] = 1
        inp = batch_x.masked_fill(mask == 0, 0)

        # client embedding forward
        self.optimizer_client_encoder.zero_grad()
        embeds, mean, std, channels = self.client_encoder(self.info, inp, mask)
        embeds1 = embeds.clone().detach().requires_grad_(True)

        # server encoder forward
        self.optimizer_server.zero_grad()
        x_enc = model(self.info, embeds1)
        x_enc1 = x_enc.clone().detach().requires_grad_(True)

        if self.client_head is None:
            self.client_head = ClientHead(self.args, x_enc.shape[1]).to(self.args.device)
            self.optimizer_client_head = torch.optim.AdamW(self.client_head.parameters(), lr=self.args.learning_rate,
                                                           weight_decay=self.args.weight_decay)
            self.scheduler_c2 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_client_head, T_max=20,
                                                                           eta_min=1e-6)
            self._print_trainable_parameters(self.client_head)

        # client head training
        self.client_head.train()
        x_enc_hat, loss = self._client_train_head(x_enc1, batch_x, batch_y, mean, std, channels)

        # server encoder backward
        x_enc.backward(x_enc_hat)
        embeds_hat = embeds1.grad.clone().detach()
        if self.args.clip != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
        self.optimizer_server.step()

        # client embed backward
        embeds.backward(embeds_hat)
        if self.args.clip != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
        self.optimizer_client_encoder.step()

        self.seen_batches = (self.seen_batches+1) % self.train_batches

        return loss, deepcopy(self.client_encoder.state_dict())

    def train_split(self, model, set_batches, embed_state):
        print(self.data_id)
        self.client_encoder.load_state_dict(embed_state)
        epoch_loss = []
        batch = 0
        client_encoder_state = None
        while batch < set_batches:
            loss, client_encoder_state = self.train_batch_split(model, embed_state)
            epoch_loss.append(loss)
            batch += 1
        loss = np.mean(epoch_loss)
        return loss, client_encoder_state


    def valid_split(self, server_model, embed_state):
        valid_loss = []
        self.client_encoder.eval()
        server_model.eval()
        self.client_head.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.valid_loaders):
                batch_x = batch_x.float().to(self.args.device)
                batch_y = batch_y.float().to(self.args.device)

                b, t, n = batch_x.shape
                mask = torch.ones((b, t, n)).to(self.args.device)
                # client_embed
                self.client_encoder.load_state_dict(embed_state)
                embeds, mean, std, channels = self.client_encoder(self.info, batch_x, mask)
                # server_encoder
                x_enc = server_model(self.info, embeds)
                # client_head
                outputs = self.client_head(x_enc, mean, std, channels)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, self.args.max_backcast_len:
                                  self.args.max_backcast_len+self.args.pred_len, f_dim:]
                batch_y = batch_y[..., f_dim:]

                loss = self.criterion(outputs, batch_y)
                valid_loss.append(loss.item())

        valid_loss = np.average(valid_loss)
        return valid_loss

    def test_split(self, server_model, path_head):
        for test_loader in self.test_loaders:
            preds = []
            trues = []
            self.client_encoder.eval()#existed encoder
            server_model.eval()#existed server
            with torch.no_grad():
                for i, (batch_x, batch_y) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.args.device)
                    batch_y = batch_y.float().to(self.args.device)

                    b, t, n = batch_x.shape
                    mask = torch.ones((b, t, n)).to(self.args.device)

                    # client_embed
                    embeds, mean, std, channels = self.client_encoder(self.info, batch_x, mask)
                    # server_encoder
                    x_enc = server_model(self.info, embeds)
                    if self.client_head is None:
                        self.client_head = ClientHead(self.args, x_enc.shape[1]).to(self.args.device)
                    self.client_head.load_state_dict(torch.load(path_head))
                    self.client_head.eval()
                    # client_head
                    outputs = self.client_head(x_enc, mean, std, channels)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, self.args.max_backcast_len:
                                      self.args.max_backcast_len+batch_y.shape[1], f_dim:]
                    batch_y = batch_y[..., f_dim:]

                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()
                    preds.append(outputs)
                    trues.append(batch_y)

            preds = np.concatenate(preds, 0)
            trues = np.concatenate(trues, 0)

            mae, mse, rmse, mape, mspe = metric(preds, trues)
            self.args.logger.info('Setting: {}, MSE: {:.6f}, MAE: {:.6f}'.format(self.data_id+str(batch_y.shape[1]), mse, mae))

            f = open(os.path.join(self.args.checkpoint, 'result_s' + str(self.args.seed) + '.txt'), 'a')
            f.write(self.data_id + '\n')
            f.write('MSE: {}, MAE: {}'.format(mse, mae))
            f.write('\n')
            f.write('\n')
            f.close()

    def update_lr(self):
        self.scheduler_c1.step()
        self.args.logger.info('Update client_encoder learning rate to {}'.format(self.scheduler_c1.get_last_lr()[0]))
        self.scheduler_s.step()
        self.args.logger.info('Update server_learning rate to {}'.format(self.scheduler_s.get_last_lr()[0]))
        self.scheduler_c2.step()
        self.args.logger.info('Update client_head learning rate to {}'.format(self.scheduler_c2.get_last_lr()[0]))
