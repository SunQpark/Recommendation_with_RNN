import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        Modify __init__() if you have additional arguments to pass.
    """
    def __init__(self, model, loss, metrics, data_loader, optimizer, epochs,
                 save_dir, save_freq, resume, with_cuda, verbosity, training_name='',
                 valid_data_loader=None, train_logger=None, monitor='loss', monitor_mode='min'):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, epochs,
                                      save_dir, save_freq, resume, verbosity, training_name,
                                      with_cuda, train_logger, monitor, monitor_mode)
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False

    # def _to_variable(self, data, target):
    #     # data, target = torch.FloatTensor(data), torch.LongTensor(target)
    #     data, target = Variable(data), Variable(target)
    #     if self.with_cuda:
    #         data, target = data.cuda(), target.cuda()
    #     return data, target

    def _sess2data_target(self, session):
        """ method to transform packed sequence session into data and target """
        padded_sess, lengths = pad_packed_sequence(session)
        padded_sess = padded_sess.float()
        padded_data = padded_sess[:-1]
        padded_target = padded_sess[1:]
        lengths -= 1
        data = pack_padded_sequence(padded_data, lengths)
        target = pack_padded_sequence(padded_target, lengths)
        return data, target

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
        """
        self.model.train()
        if self.with_cuda:
            self.model.cuda()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, session in enumerate(self.data_loader):
            data, target = self._sess2data_target(session)

            self.optimizer.zero_grad()
            output, hidden = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            for i, metric in enumerate(self.metrics):
                y_output = output[0].data.cpu().numpy()
                y_output = np.argmax(y_output, axis=1)
                y_target = target[0].data.cpu().numpy()
                total_metrics[i] += metric(y_output, y_target)

            total_loss += loss.data[0]
            log_step = int(np.sqrt(self.batch_size))
            if self.verbosity >= 2 and batch_idx % log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.data_loader) * len(data),
                    100.0 * batch_idx / len(self.data_loader), loss.data[0]))

        avg_loss = total_loss / len(self.data_loader)
        avg_metrics = (total_metrics / len(self.data_loader)).tolist()
        log = {'loss': avg_loss, 'metrics': avg_metrics}

        if self.valid:
            val_log = self._valid_epoch()
            log = {**log, **val_log}

        return log

    def _valid_epoch(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        for batch_idx, session in enumerate(self.valid_data_loader):
            data, target = self._sess2data_target(session)

            output, hidden = self.model(data)
            loss = self.loss(output, target)
            total_val_loss += loss.data[0]

            for i, metric in enumerate(self.metrics):
                y_output = output.data.cpu().numpy()
                y_output = np.argmax(y_output, axis=1)
                y_target = target.data.cpu().numpy()
                total_val_metrics[i] += metric(y_output, y_target)

        avg_val_loss = total_val_loss / len(self.valid_data_loader)
        avg_val_metrics = (total_val_metrics / len(self.valid_data_loader)).tolist()
        return {'val_loss': avg_val_loss, 'val_metrics': avg_val_metrics}

