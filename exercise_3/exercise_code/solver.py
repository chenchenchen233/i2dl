from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################
        for epoch in range(num_epochs):
            # TRAINING
            for i, (inputs, targets) in enumerate(train_loader, 1):
                inputs, targets = Variable(inputs), Variable(targets)
                if model.is_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                    
                optim.zero_grad()
                outputs = model(inputs)
                loss = self.loss_func(outputs, targets)
                loss.backward()
                optim.step()
                
                self.train_loss_history.append(loss.data.cpu().numpy())
                if log_nth and i % log_nth == 0:
                    last_log_nth_losses = self.train_loss_history[-log_nth:]
                    train_loss = np.mean(last_log_nth_losses)
# =============================================================================
#                     print('[Iteration %d/%d] TRAIN loss: %.3f' % \
#                          (i + epoch * iter_per_epoch,
#                           iter_per_epoch * num_epochs,
#                           train_loss))
# ==========================================================================
            _, preds = torch.max(outputs, 1)
            targets_mask = targets >= 0
            train_acc = np.mean((preds == targets)[targets_mask].data.cpu().numpy())
            self.train_acc_history.append(train_acc)
            if log_nth:
                 print('[Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   train_acc,
                                                                   train_loss))
            
            # VALIDATION
            val_losses = []
            val_scores = []
            model.eval()
            for inputs, targets in val_loader:
                inputs, targets = Variable(inputs), Variable(targets)
                
                if model.is_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model.forward(inputs)
                loss = self.loss_func(outputs, targets)
                val_losses.append(loss.data.cpu().numpy())
                
                _, preds = torch.max(outputs, 1)
                # Only allow images/pixels with target >= 0 e.g. for segmentation
                targets_mask = targets >= 0
                scores = np.mean((preds == targets)[targets_mask].data.cpu().numpy())
                val_scores.append(scores)
            model.train()
            val_acc, val_loss = np.mean(val_scores), np.mean(val_losses)
            self.val_acc_history.append(val_acc)
            self.val_loss_history.append(val_loss)
            if log_nth:
                print('[Epoch %d/%d] VAL   acc/loss: %.3f/%.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   val_acc,
                                                                   val_loss))
                 
                 

# =============================================================================
#         num_iterations=iter_per_epoch*num_epochs
#         best_val_acc = -1
#         loss_func=self.loss_func
#         best_params={}
#         
#         train_acc = 0
#         val_accu = 0
#         for epoch in range(num_epochs):
#             ##Train
#             model.train()
#             for i, data in enumerate(train_loader, 0):
#                 inputs, labels = data
#                 inputs, labels= Variable(input),Variable(labels)
#                 
#                 if model.is_cuda:
#                     inputs = inputs.cuda()
#                     labels = labels.cuda()
#                 optim.zero_grad()
#                 outputs = model(inputs)
#                 train_loss = loss_func(outputs, labels)
#                 loss.backward()
#                 optim.step()
# =============================================================================
                
# =============================================================================
#                 self.train_loss_history.append(train_loss.data[0])
# =============================================================================
# =============================================================================
#                 self.train_loss_history.append(loss.data.cpu().numpy())
#                 
#                 t=epoch*(iter_per_epoch-1) +i
#                 if i % log_nth == 0:
#                     print('[Iteration %d / %d] TRAIN loss: %f' % (t, num_iterations, train_loss.data[0])
#                 if (i+1) % iter_per_epoch == 0:
#                     _,pred_train=torch.max(outputs)
#                     acc_train=np.mean(pred_train == labels)
#                     self.train_acc_history.append(acc_train)
#                     print('Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f' % (epoch + 1,num_epochs,train_acc,train_loss.data[0]))
#             model.eval()
#             ##Validation
#             for i,data_val in enumerate(val_loader, 0):
#                 input_val,labels_val = data_val
#                 input_val, labels_val = Variable(input_val), Variable(labels_val)
#     
# =============================================================================
# =============================================================================
#                 t=epoch*(iter_per_epoch-1) +i
#                 if t%log_nth ==0 :
#                     print('[Iteration %d / %d] TRAIN loss: %f' % (t, num_iterations, train_loss.data[0])
#                 if i == len(train_loader)
#                     y_pred = np.argmax(outputs.data.numpy(), axis = 1)
#                     train_acc = np.mean(y_pred == labels)
#                     self.train_acc_history.append(train_acc)
#                     
#             model.eval()
#             for i,data_val in enumerate(val_loader, 0):
#                 input_val,labels_val = data
#                 input_val, labels_val = Variable(input_val), Variable(labels_val)
#                 
#                 if model.is_cuda:
#                     inputs= inputs.cuda()
#                     labels = labels.cuda()
#                 output_val=model(input_val)
#                 loss_val=self.loss_func(output_val,labels_val)
#                 val_pred=torch.max(X_val,1)
#                 val_loss=loss_func(val_pred, y_val)
#                 
#                 self.val_loss_history.append(val_loss.data[0])
#                 if i == len(val_loader):
#                     y_pred = np.argmax(outputs.data.numpy(), axis = 1)
#                     val_acc = np.mean(outputs == labels)
#                     self.train_acc_history.append(val_acc)
#             print('[Epoch %d / %d] train acc: %f; val_acc: %f' % (epoch, num_epochs, train_acc, val_acc))
#                     
# =============================================================================
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
        return val_acc
