import os
import json
import copy
import time
import torch
import numpy as np
import pandas as pd
import configparser
from copy import deepcopy

from models.model import ServerEncoder, ClientEncoder
from data_provider.data_factory import data_provider
from engines.client_avg import Engine_Forecasting

class Engine(object):
    '''
    server
    '''
    def __init__(self, args):
        args.device = torch.device('cuda:{}'.format(args.gpu))
        model_map_dict = {
            'gpt2': 'gpt2'
        }
        self.data_name_map = {
            'ETTh1': 0,
            'ETTh2': 1,
            'ETTm1': 2,
            'ETTm2': 3,
            'Electricity': 4,
            'Weather': 5,
            'Exchange': 6,
            'Illness': 7
        }
        args.model_path = model_map_dict[args.lm_pretrain_model]
        if args.lm_pretrain_model == 'gpt2':
            self.server_encoder = ServerEncoder(args).to(args.device)
            args.d_model = self.server_encoder.d_model
            self.word_embeddings = self.server_encoder.backbone.get_input_embeddings().weight.clone().detach().requires_grad_(True)

        self.client_encoder = ClientEncoder(args, self.word_embeddings).to(args.device)
        self.client_head = None

        self.args = args

        self._print_trainable_parameters(self.server_encoder)
        self._construct_unified_dataloaders()

    def _print_trainable_parameters(self, server_encoder):
        freeze = 0
        trainable = 0
        for name, param in server_encoder.named_parameters():
            if param.requires_grad:
                trainable += param.nelement()
            else:
                freeze += param.nelement()
        self.args.logger.info('Trainable Params: {}, All Params: {}, Percent: {}'.format(
                              trainable, freeze + trainable, trainable / (freeze + trainable)))


    def _construct_unified_dataloaders(self):
        self.clients = [None] * len(self.data_name_map)
        self.client_train_selection = [0] * len(self.data_name_map)
        self.client_test_selection  = [0] * len(self.data_name_map)
        self.client_valid_set = []
        self.client_test_set = []
        self.train_batches = 0
        self.client_train_set = []
        if self.args.is_training:
            df = pd.read_csv(self.args.training_list)
        else:
            df = pd.read_csv(self.args.inference_list)
        for i, row in df.iterrows():
            args = copy.deepcopy(self.args)
            data_name = row['Data']
            train_flag = row['Train']
            valid_flag = row['Valid']
            test_flag = row['Test']

            config = configparser.ConfigParser()
            config.read('data_configs/' + data_name + '.conf')
            data_config = config['config']

            args.data_path = data_config['data_path']
            args.data_reader = data_config['data_reader']
            args.data_id = data_config['data_id']
            args.features = data_config['features']
            args.seq_len = int(data_config['seq_len'])
            args.stride = int(data_config['stride'])
            args.batch_size = int(data_config['batch_size'])

            args.pred_len = int(row['Prediction'])
            args.mask_rate = self.args.mask_rate

            client_idx = self.data_name_map[data_name]
            if self.clients[client_idx] is None:
                client = Engine_Forecasting(args, self.server_encoder, self.client_encoder, self.client_head)
                self.clients[client_idx] = client

            setting = '{}_{}_{}_{}_{}_{}'.format(args.data_id, args.features, args.seq_len, args.pred_len, args.stride, args.batch_size, args.learning_rate)
            self.args.logger.info('***** Task: {} *****'.format(setting))

            if self.args.is_training:
                if train_flag:
                    _, train_loader = data_provider(args, 'train')
                    self.clients[client_idx].train_loaders = train_loader
                    self.clients[client_idx].train_batches = len(train_loader)
                    self.client_train_selection[client_idx] += 1
                    if self.client_train_selection[client_idx] <= 1:
                        self.train_batches += len(train_loader)
                        self.client_train_set.append(client_idx)
                if valid_flag:
                    _, valid_loader = data_provider(args, 'val')
                    self.clients[client_idx].valid_loaders = valid_loader
                    self.client_valid_set.append(client_idx)

                if test_flag:
                    _, test_loader = data_provider(args, 'test')
                    self.clients[client_idx].test_loaders.append(test_loader)
                    self.client_test_selection[client_idx] += 1
                    if self.client_test_selection[client_idx] <= 1:
                        self.client_test_set.append(client_idx)

            else:
                _, test_loader = data_provider(args, 'test')
                self.clients[client_idx].test_loaders.append(test_loader)
                self.client_test_selection[client_idx] += 1
                if self.client_test_selection[client_idx] <= 1:
                    self.client_test_set.append(client_idx)

    def train(self):
        self.args.logger.info('Start training!')

        self.client_train_selection = np.array(self.client_train_selection) / np.sum(self.client_train_selection)
        wait = 0
        best_valid_loss = np.array([5] * len(self.client_valid_set))
        for e in range(self.args.train_epochs):
            batch_cnt = [0] * len(self.clients)

            #train with one batch
            t1 = time.time()
            train_loss = []
            client_states = []
            set_batches = np.int32(self.train_batches * self.client_train_selection)
            np.random.shuffle(self.client_train_set)
            for idx in self.client_train_set:
                if self.args.local_batches == 'set':
                    loss, client_state = self.clients[idx].train_split(self.server_encoder, set_batches[idx], deepcopy(self.client_encoder.state_dict()))
                else:
                    loss, client_state = self.clients[idx].train_split(self.server_encoder,
                                                                       self.clients[idx].train_batches,
                                                                       deepcopy(self.client_encoder.state_dict()))
                train_loss.append(loss)
                client_states.append(client_state)

            mtrain_loss = np.mean(train_loss)

            t2 = time.time()
            self.args.logger.info('Epoch: {}, Train Time: {:.6f}, Train Loss: {:.6f}'.format(e + 1, t2 - t1, mtrain_loss))

            self.client_encoder.load_state_dict(self.average_weights(client_states))


            # valid
            v1 = time.time()
            valid_loss = []
            for client_idx in self.client_valid_set:
                loss = self.clients[client_idx].valid_split(self.server_encoder, self.client_encoder.state_dict())
                valid_loss.append(loss)

            valid_loss = np.array(valid_loss)
            mvalid_loss = np.mean(valid_loss)
            improve = np.sum((best_valid_loss - valid_loss) / best_valid_loss)
            v2 = time.time()
            self.args.logger.info(
                'Epoch: {}, Valid Time: {:.6f}, Valid Loss: {:.6f}, Valid Loss Improve: {:.6f}'.format(e + 1, v2 - v1, mvalid_loss, improve))

            if improve >= 0:
                torch.save(self.server_encoder.state_dict(),
                           os.path.join(self.args.checkpoint, 'server_model_s' + str(self.args.seed) + '.pth'))

                for idx in self.client_train_set:
                    torch.save(self.clients[idx].client_encoder.state_dict(),
                               os.path.join(self.args.checkpoint, 'client_{}_encoder'.format(idx) + str(self.args.seed) + '.pth'))
                    torch.save(self.clients[idx].client_head.state_dict(),
                               os.path.join(self.args.checkpoint, 'client_{}_head'.format(idx) + str(self.args.seed) + '.pth'))
                self.args.logger.info('Saving best model!')
                best_valid_loss = valid_loss
                wait = 0
            else:
                torch.save(self.server_encoder.state_dict(), os.path.join(self.args.checkpoint,
                                                                          'server_model_s' + str(self.args.seed) + '_e' + str(
                                                                              e + 1) + '.pth'))
                for idx in self.client_train_set:
                    torch.save(self.clients[idx].client_encoder.state_dict(), os.path.join(self.args.checkpoint,
                                                                              'client_{}_encoder'.format(idx) + str(
                                                                                  self.args.seed) + '_e' + str(
                                                                                  e + 1) + '.pth'))
                    torch.save(self.clients[idx].client_head.state_dict(), os.path.join(self.args.checkpoint,
                                                                                'client_{}_head'.format(idx) + str(
                                                                                    self.args.seed) + '_e' + str(
                                                                                    e + 1) + '.pth'))
                wait += 1
                if wait == self.args.patience:
                    self.args.logger.info('Early stop at epoch {}'.format(e + 1))
                    break

            for idx in self.client_train_set:
                self.args.logger.info('update lr for client_{}'.format(idx))
                self.clients[idx].update_lr()

        self.test()


    def test(self):
        self.args.logger.info('Start testing!')
        if self.args.eval_model_path != '':
            path_server = self.args.eval_model_path
            self.server_encoder.load_state_dict(torch.load(path_server))
        else:
            path_server = os.path.join(self.args.checkpoint, 'server_model_s' + str(self.args.seed) + '.pth')
            for idx in self.client_test_set:
                path_encoder = os.path.join(self.args.checkpoint,
                                            'client_{}_encoder'.format(idx) + str(self.args.seed) + '.pth')
                path_head = os.path.join(self.args.checkpoint,
                                         'client_{}_head'.format(idx) + str(self.args.seed) + '.pth')
                self.clients[idx].client_encoder.load_state_dict(torch.load(path_encoder))
                self.server_encoder.load_state_dict(torch.load(path_server))
                self.clients[idx].test_split(self.server_encoder, path_head)

    def average_weights(self, w):
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))

        return w_avg