import torch
import torch.nn.functional as F
from torch.autograd import Variable as Var
from tqdm import tqdm
from . import utils

from . import Constants


class Trainer(object):
    def __init__(self, args, model, embeddings, vocabs, criterion, optimizer):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.embeddings = embeddings
        self.vocabs = vocabs
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = 0

    def train_embeddings(self):
        for key in self.embeddings.keys():
            self.embeddings[key].train()
            self.embeddings[key].zero_grad()

    def set_embeddings_step(self):
        for key in self.embeddings.keys():
            if key == 'chars':
                continue

            for f in self.embeddings[key].parameters():
                f.data.sub_(f.grad.data * self.args.emblr)

            self.embeddings[key].zero_grad()

    def test_embeddings(self):
        for key in self.embeddings.keys():
            self.embeddings[key].eval()

    def get_char_vector(self, toks_sent):
        words = self.vocabs['toks'].convertToLabels(toks_sent.numpy(), None)
        char_vectors = []

        for word in words:
            if word == Constants.UNK_WORD:
                char_vectors.append(F.torch.zeros(self.vocabs['chars'].size()))
            else:
                char_vector = []
                for char in word:
                    if self.vocabs['chars'].getIndex(char) != None:
                        char_vector.append(self.vocabs['chars'].getIndex(char) )
                    else:
                        char_vector.append(0)

                char_vector = torch.tensor(char_vector)
                char_vectors.append(F.torch.sum(self.embeddings['chars'](char_vector), 0) / len(char_vector))

        return F.torch.unsqueeze(torch.stack(char_vectors), 1)

    def get_data(self, data, num_classes):
        tree, toks_sent, pos_sent, rels_sent, label = data
        toks_sent = Var(toks_sent)
        pos_sent = Var(pos_sent)
        rels_sent = Var(rels_sent)


        target = Var(utils.map_label_to_target(label, num_classes, self.vocabs['output']))

        toks_emb = F.torch.unsqueeze(self.embeddings['toks'](toks_sent), 1)
        pos_emb = F.torch.unsqueeze(self.embeddings['pos'](pos_sent), 1)
        rels_emb = F.torch.unsqueeze(self.embeddings['rels'](rels_sent), 1)
        chars_emb = self.get_char_vector(toks_sent)

        return tree, torch.cat((toks_emb, pos_emb, rels_emb, chars_emb), 2), target
        # return tree, torch.cat((toks_emb, pos_emb, rels_emb), 2), target

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.train_embeddings()

        self.optimizer.zero_grad()
        total_loss, k = 0.0, 0
        indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')

        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            tree, emb, target = self.get_data(dataset[indices[idx]], dataset.num_classes)

            output = self.model.forward(tree, emb, training=True)
            err = self.criterion(output, target.cuda())

            total_loss += err.item()
            err.backward()
            k += 1

            if k == self.args.batchsize:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.set_embeddings_step()
                k = 0

        self.epoch += 1
        return total_loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        self.test_embeddings()

        total_loss = 0
        predictions = torch.zeros(len(dataset))

        for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
            torch.no_grad()
            tree, emb, target = self.get_data(dataset[idx], dataset.num_classes)

            output = self.model.forward(tree, emb, training=True)
            err = self.criterion(output, target.cuda())
            total_loss += err.item()

            _, pred = torch.max(output, 1)

            predictions[idx] = pred.data.cpu()[0]
        return total_loss / len(dataset), predictions


class TrainernoChars(object):
    def __init__(self, args, model, embeddings, vocabs, criterion, optimizer):
        super(TrainernoChars, self).__init__()
        self.args = args
        self.model = model
        self.embeddings = embeddings
        self.vocabs = vocabs
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = 0

    def train_embeddings(self):
        for key in self.embeddings.keys():
            self.embeddings[key].train()
            self.embeddings[key].zero_grad()

    def set_embeddings_step(self):
        for key in self.embeddings.keys():
            if key == 'chars':
                continue

            for f in self.embeddings[key].parameters():
                f.data.sub_(f.grad.data * self.args.emblr)

            self.embeddings[key].zero_grad()

    def test_embeddings(self):
        for key in self.embeddings.keys():
            self.embeddings[key].eval()

    def get_char_vector(self, toks_sent):
        words = self.vocabs['toks'].convertToLabels(toks_sent.numpy(), None)
        char_vectors = []

        for word in words:
            if word == Constants.UNK_WORD:
                char_vectors.append(F.torch.zeros(self.vocabs['chars'].size()))
            else:
                char_vector = []
                for char in word:
                    if self.vocabs['chars'].getIndex(char) != None:
                        char_vector.append(self.vocabs['chars'].getIndex(char) )
                    else:
                        char_vector.append(0)

                char_vector = torch.tensor(char_vector)
                char_vectors.append(F.torch.sum(self.embeddings['chars'](char_vector), 0) / len(char_vector))

        return F.torch.unsqueeze(torch.stack(char_vectors), 1)

    def get_data(self, data, num_classes):
        tree, toks_sent, pos_sent, rels_sent, label = data
        toks_sent = Var(toks_sent)
        pos_sent = Var(pos_sent)
        rels_sent = Var(rels_sent)


        target = Var(utils.map_label_to_target(label, num_classes, self.vocabs['output']))

        toks_emb = F.torch.unsqueeze(self.embeddings['toks'](toks_sent), 1)
        pos_emb = F.torch.unsqueeze(self.embeddings['pos'](pos_sent), 1)
        rels_emb = F.torch.unsqueeze(self.embeddings['rels'](rels_sent), 1)

        # return tree, torch.cat((toks_emb, pos_emb, rels_emb, chars_emb), 2), target
        return tree, torch.cat((toks_emb, pos_emb, rels_emb), 2), target

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.train_embeddings()

        self.optimizer.zero_grad()
        total_loss, k = 0.0, 0
        indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')

        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            tree, emb, target = self.get_data(dataset[indices[idx]], dataset.num_classes)

            output = self.model.forward(tree, emb, training=True)
            err = self.criterion(output, target.cuda())

            total_loss += err.item()
            err.backward()
            k += 1

            if k == self.args.batchsize:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.set_embeddings_step()
                k = 0

        self.epoch += 1
        return total_loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        self.test_embeddings()

        total_loss = 0
        predictions = torch.zeros(len(dataset))

        for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
            torch.no_grad()
            tree, emb, target = self.get_data(dataset[idx], dataset.num_classes)

            output = self.model.forward(tree, emb, training=True)
            err = self.criterion(output, target.cuda())
            total_loss += err.item()

            _, pred = torch.max(output, 1)

            predictions[idx] = pred.data.cpu()[0]
        return total_loss / len(dataset), predictions


class TrainerPOS(object):
    def __init__(self, args, model, embeddings, vocabs, criterion, optimizer):
        super(TrainerPOS, self).__init__()
        self.args = args
        self.model = model
        self.embeddings = embeddings
        self.vocabs = vocabs
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = 0

    def train_embeddings(self):
        for key in self.embeddings.keys():
            self.embeddings[key].train()
            self.embeddings[key].zero_grad()

    def set_embeddings_step(self):
        for key in self.embeddings.keys():
            if key == 'chars':
                continue

            for f in self.embeddings[key].parameters():
                f.data.sub_(f.grad.data * self.args.emblr)

            self.embeddings[key].zero_grad()

    def test_embeddings(self):
        for key in self.embeddings.keys():
            self.embeddings[key].eval()

    def get_char_vector(self, toks_sent):
        words = self.vocabs['toks'].convertToLabels(toks_sent.numpy(), None)
        char_vectors = []

        for word in words:
            if word == Constants.UNK_WORD:
                char_vectors.append(F.torch.zeros(self.vocabs['chars'].size()))
            else:
                char_vector = []
                for char in word:
                    if self.vocabs['chars'].getIndex(char) != None:
                        char_vector.append(self.vocabs['chars'].getIndex(char) )
                    else:
                        char_vector.append(0)

                char_vector = torch.tensor(char_vector)
                char_vectors.append(F.torch.sum(self.embeddings['chars'](char_vector), 0) / len(char_vector))

        return F.torch.unsqueeze(torch.stack(char_vectors), 1)

    def get_data(self, data, num_classes):
        tree, toks_sent, pos_sent, rels_sent, label = data
        toks_sent = Var(toks_sent)
        pos_sent = Var(pos_sent)


        target = Var(utils.map_label_to_target(label, num_classes, self.vocabs['output']))

        toks_emb = F.torch.unsqueeze(self.embeddings['toks'](toks_sent), 1)
        pos_emb = F.torch.unsqueeze(self.embeddings['pos'](pos_sent), 1)

        # return tree, torch.cat((toks_emb, pos_emb, rels_emb, chars_emb), 2), target
        return tree, torch.cat((toks_emb, pos_emb), 2), target

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.train_embeddings()

        self.optimizer.zero_grad()
        total_loss, k = 0.0, 0
        indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')

        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            tree, emb, target = self.get_data(dataset[indices[idx]], dataset.num_classes)

            output = self.model.forward(tree, emb, training=True)
            err = self.criterion(output, target.cuda())

            total_loss += err.item()
            err.backward()
            k += 1

            if k == self.args.batchsize:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.set_embeddings_step()
                k = 0

        self.epoch += 1
        return total_loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        self.test_embeddings()

        total_loss = 0
        predictions = torch.zeros(len(dataset))

        for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
            torch.no_grad()
            tree, emb, target = self.get_data(dataset[idx], dataset.num_classes)

            output = self.model.forward(tree, emb, training=True)
            err = self.criterion(output, target.cuda())
            total_loss += err.item()

            _, pred = torch.max(output, 1)

            predictions[idx] = pred.data.cpu()[0]
        return total_loss / len(dataset), predictions


class TrainerRELS(object):
    def __init__(self, args, model, embeddings, vocabs, criterion, optimizer):
        super(TrainerRELS, self).__init__()
        self.args = args
        self.model = model
        self.embeddings = embeddings
        self.vocabs = vocabs
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = 0

    def train_embeddings(self):
        for key in self.embeddings.keys():
            self.embeddings[key].train()
            self.embeddings[key].zero_grad()

    def set_embeddings_step(self):
        for key in self.embeddings.keys():
            if key == 'chars':
                continue

            for f in self.embeddings[key].parameters():
                f.data.sub_(f.grad.data * self.args.emblr)

            self.embeddings[key].zero_grad()

    def test_embeddings(self):
        for key in self.embeddings.keys():
            self.embeddings[key].eval()

    def get_char_vector(self, toks_sent):
        words = self.vocabs['toks'].convertToLabels(toks_sent.numpy(), None)
        char_vectors = []

        for word in words:
            if word == Constants.UNK_WORD:
                char_vectors.append(F.torch.zeros(self.vocabs['chars'].size()))
            else:
                char_vector = []
                for char in word:
                    if self.vocabs['chars'].getIndex(char) != None:
                        char_vector.append(self.vocabs['chars'].getIndex(char) )
                    else:
                        char_vector.append(0)

                char_vector = torch.tensor(char_vector)
                char_vectors.append(F.torch.sum(self.embeddings['chars'](char_vector), 0) / len(char_vector))

        return F.torch.unsqueeze(torch.stack(char_vectors), 1)

    def get_data(self, data, num_classes):
        tree, toks_sent, pos_sent, rels_sent, label = data
        toks_sent = Var(toks_sent)
        # pos_sent = Var(pos_sent)
        rels_sent = Var(rels_sent)


        target = Var(utils.map_label_to_target(label, num_classes, self.vocabs['output']))

        toks_emb = F.torch.unsqueeze(self.embeddings['toks'](toks_sent), 1)
        # pos_emb = F.torch.unsqueeze(self.embeddings['pos'](pos_sent), 1)
        rels_emb = F.torch.unsqueeze(self.embeddings['rels'](rels_sent), 1)
        # chars_emb = self.get_char_vector(toks_sent)

        # return tree, torch.cat((toks_emb, pos_emb, rels_emb, chars_emb), 2), target
        return tree, torch.cat((toks_emb, rels_emb), 2), target

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.train_embeddings()

        self.optimizer.zero_grad()
        total_loss, k = 0.0, 0
        indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')

        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            tree, emb, target = self.get_data(dataset[indices[idx]], dataset.num_classes)

            output = self.model.forward(tree, emb, training=True)
            err = self.criterion(output, target.cuda())

            total_loss += err.item()
            err.backward()
            k += 1

            if k == self.args.batchsize:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.set_embeddings_step()
                k = 0

        self.epoch += 1
        return total_loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        self.test_embeddings()

        total_loss = 0
        predictions = torch.zeros(len(dataset))

        for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
            torch.no_grad()
            tree, emb, target = self.get_data(dataset[idx], dataset.num_classes)

            output = self.model.forward(tree, emb, training=True)
            err = self.criterion(output, target.cuda())
            total_loss += err.item()

            _, pred = torch.max(output, 1)

            predictions[idx] = pred.data.cpu()[0]
        return total_loss / len(dataset), predictions


class TrainerTOK(object):
    def __init__(self, args, model, embeddings, vocabs, criterion, optimizer):
        super(TrainerTOK, self).__init__()
        self.args = args
        self.model = model
        self.embeddings = embeddings
        self.vocabs = vocabs
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = 0

    def train_embeddings(self):
        for key in self.embeddings.keys():
            self.embeddings[key].train()
            self.embeddings[key].zero_grad()

    def set_embeddings_step(self):
        for key in self.embeddings.keys():
            if key == 'chars':
                continue

            for f in self.embeddings[key].parameters():
                f.data.sub_(f.grad.data * self.args.emblr)

            self.embeddings[key].zero_grad()

    def test_embeddings(self):
        for key in self.embeddings.keys():
            self.embeddings[key].eval()

    def get_char_vector(self, toks_sent):
        words = self.vocabs['toks'].convertToLabels(toks_sent.numpy(), None)
        char_vectors = []

        for word in words:
            if word == Constants.UNK_WORD:
                char_vectors.append(F.torch.zeros(self.vocabs['chars'].size()))
            else:
                char_vector = []
                for char in word:
                    if self.vocabs['chars'].getIndex(char) != None:
                        char_vector.append(self.vocabs['chars'].getIndex(char) )
                    else:
                        char_vector.append(0)

                char_vector = torch.tensor(char_vector)
                char_vectors.append(F.torch.sum(self.embeddings['chars'](char_vector), 0) / len(char_vector))

        return F.torch.unsqueeze(torch.stack(char_vectors), 1)

    def get_data(self, data, num_classes):
        tree, toks_sent, pos_sent, rels_sent, label = data
        # tree, toks_sent, label = data
        toks_sent = Var(toks_sent)
        # pos_sent = Var(pos_sent)
        # rels_sent = Var(rels_sent)


        target = Var(utils.map_label_to_target(label, num_classes, self.vocabs['output']))

        toks_emb = F.torch.unsqueeze(self.embeddings['toks'](toks_sent), 1)
        # pos_emb = F.torch.unsqueeze(self.embeddings['pos'](pos_sent), 1)
        # rels_emb = F.torch.unsqueeze(self.embeddings['rels'](rels_sent), 1)
        # chars_emb = self.get_char_vector(toks_sent)

        # return tree, torch.cat((toks_emb, pos_emb, rels_emb, chars_emb), 2), target
        return tree, toks_emb, target

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.train_embeddings()

        self.optimizer.zero_grad()
        total_loss, k = 0.0, 0
        indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')

        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            tree, emb, target = self.get_data(dataset[indices[idx]], dataset.num_classes)

            output = self.model.forward(tree, emb, training=True)
            err = self.criterion(output, target.cuda())

            total_loss += err.item()
            err.backward()
            k += 1

            if k == self.args.batchsize:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.set_embeddings_step()
                k = 0

        self.epoch += 1
        return total_loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        self.test_embeddings()

        total_loss = 0
        predictions = torch.zeros(len(dataset))

        for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
            torch.no_grad()
            tree, emb, target = self.get_data(dataset[idx], dataset.num_classes)

            output = self.model.forward(tree, emb, training=True)
            err = self.criterion(output, target.cuda())
            total_loss += err.item()

            _, pred = torch.max(output, 1)

            predictions[idx] = pred.data.cpu()[0]
        return total_loss / len(dataset), predictions


class TrainerPOSTAG(object):
    def __init__(self, args, model, embeddings, vocabs, criterion, optimizer):
        super(TrainerPOSTAG, self).__init__()
        self.args = args
        self.model = model
        self.embeddings = embeddings
        self.vocabs = vocabs
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = 0

    def train_embeddings(self):
        for key in self.embeddings.keys():
            self.embeddings[key].train()
            self.embeddings[key].zero_grad()

    def set_embeddings_step(self):
        for key in self.embeddings.keys():
            if key == 'chars':
                continue

            for f in self.embeddings[key].parameters():
                f.data.sub_(f.grad.data * self.args.emblr)

            self.embeddings[key].zero_grad()

    def test_embeddings(self):
        for key in self.embeddings.keys():
            self.embeddings[key].eval()

    def get_char_vector(self, toks_sent):
        words = self.vocabs['toks'].convertToLabels(toks_sent.numpy(), None)
        char_vectors = []

        for word in words:
            if word == Constants.UNK_WORD:
                char_vectors.append(F.torch.zeros(self.vocabs['chars'].size()))
            else:
                char_vector = []
                for char in word:
                    if self.vocabs['chars'].getIndex(char) != None:
                        char_vector.append(self.vocabs['chars'].getIndex(char) )
                    else:
                        char_vector.append(0)

                char_vector = torch.tensor(char_vector)
                char_vectors.append(F.torch.sum(self.embeddings['chars'](char_vector), 0) / len(char_vector))

        return F.torch.unsqueeze(torch.stack(char_vectors), 1)

    def get_data(self, data, num_classes):
        tree, toks_sent, pos_sent, tag_sent, rels_sent, label = data
        toks_sent = Var(toks_sent)
        pos_sent = Var(pos_sent)
        tag_sent = Var(tag_sent)
        rels_sent = Var(rels_sent)
        target = Var(utils.map_label_to_target(label, num_classes, self.vocabs['output']))

        toks_emb = F.torch.unsqueeze(self.embeddings['toks'](toks_sent), 1)
        pos_emb = F.torch.unsqueeze(self.embeddings['pos'](pos_sent), 1)
        tag_emb = F.torch.unsqueeze(self.embeddings['tag'](tag_sent), 1)
        rels_emb = F.torch.unsqueeze(self.embeddings['rels'](rels_sent), 1)
        chars_emb = self.get_char_vector(toks_sent)

        return tree, torch.cat((toks_emb, pos_emb, rels_emb, tag_emb, chars_emb), 2), target
        # return tree, torch.cat((toks_emb, pos_emb, rels_emb), 2), target

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.train_embeddings()

        self.optimizer.zero_grad()
        total_loss, k = 0.0, 0
        indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')

        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            tree, emb, target = self.get_data(dataset[indices[idx]], dataset.num_classes)

            output = self.model.forward(tree, emb, training=True)
            err = self.criterion(output, target.cuda())

            total_loss += err.item()
            err.backward()
            k += 1

            if k == self.args.batchsize:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.set_embeddings_step()
                k = 0

        self.epoch += 1
        return total_loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        self.test_embeddings()

        total_loss = 0
        predictions = torch.zeros(len(dataset))

        for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
            torch.no_grad()
            tree, emb, target = self.get_data(dataset[idx], dataset.num_classes)

            output = self.model.forward(tree, emb, training=True)
            err = self.criterion(output, target.cuda())
            total_loss += err.item()

            _, pred = torch.max(output, 1)

            predictions[idx] = pred.data.cpu()[0]
        return total_loss / len(dataset), predictions


class TrainerENT(object):
    def __init__(self, args, model, embeddings, vocabs, criterion, optimizer):
        super(TrainerENT, self).__init__()
        self.args = args
        self.model = model
        self.embeddings = embeddings
        self.vocabs = vocabs
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = 0

    def train_embeddings(self):
        for key in self.embeddings.keys():
            self.embeddings[key].train()
            self.embeddings[key].zero_grad()

    def set_embeddings_step(self):
        for key in self.embeddings.keys():
            if key == 'chars':
                continue

            for f in self.embeddings[key].parameters():
                f.data.sub_(f.grad.data * self.args.emblr)

            self.embeddings[key].zero_grad()

    def test_embeddings(self):
        for key in self.embeddings.keys():
            self.embeddings[key].eval()

    def get_char_vector(self, toks_sent):
        words = self.vocabs['toks'].convertToLabels(toks_sent.numpy(), None)
        char_vectors = []

        for word in words:
            if word == Constants.UNK_WORD:
                char_vectors.append(F.torch.zeros(self.vocabs['chars'].size()))
            else:
                char_vector = []
                for char in word:
                    if self.vocabs['chars'].getIndex(char) != None:
                        char_vector.append(self.vocabs['chars'].getIndex(char) )
                    else:
                        char_vector.append(0)

                char_vector = torch.tensor(char_vector)
                char_vectors.append(F.torch.sum(self.embeddings['chars'](char_vector), 0) / len(char_vector))

        return F.torch.unsqueeze(torch.stack(char_vectors), 1)

    def get_data(self, data, num_classes):
        tree, toks_sent, pos_sent, tag_sent, ent_sent, rels_sent, label = data
        toks_sent = Var(toks_sent)
        pos_sent = Var(pos_sent)
        tag_sent = Var(tag_sent)
        ent_sent = Var(ent_sent)
        rels_sent = Var(rels_sent)


        target = Var(utils.map_label_to_target(label, num_classes, self.vocabs['output']))

        toks_emb = F.torch.unsqueeze(self.embeddings['toks'](toks_sent), 1)
        pos_emb = F.torch.unsqueeze(self.embeddings['pos'](pos_sent), 1)
        tag_emb = F.torch.unsqueeze(self.embeddings['tag'](tag_sent), 1)
        ent_emb = F.torch.unsqueeze(self.embeddings['ent'](ent_sent), 1)
        rels_emb = F.torch.unsqueeze(self.embeddings['rels'](rels_sent), 1)
        chars_emb = self.get_char_vector(toks_sent)

        return tree, torch.cat((toks_emb, pos_emb, rels_emb, tag_emb, ent_emb, chars_emb), 2), target
        # return tree, torch.cat((toks_emb, pos_emb, rels_emb), 2), target

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.train_embeddings()

        self.optimizer.zero_grad()
        total_loss, k = 0.0, 0
        indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')

        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            tree, emb, target = self.get_data(dataset[indices[idx]], dataset.num_classes)

            output = self.model.forward(tree, emb, training=True)
            err = self.criterion(output, target.cuda())

            total_loss += err.item()
            err.backward()
            k += 1

            if k == self.args.batchsize:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.set_embeddings_step()
                k = 0

        self.epoch += 1
        return total_loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        self.test_embeddings()

        total_loss = 0
        predictions = torch.zeros(len(dataset))

        for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
            torch.no_grad()
            tree, emb, target = self.get_data(dataset[idx], dataset.num_classes)

            output = self.model.forward(tree, emb, training=True)
            err = self.criterion(output, target.cuda())
            total_loss += err.item()

            _, pred = torch.max(output, 1)

            predictions[idx] = pred.data.cpu()[0]
        return total_loss / len(dataset), predictions


class TrainerTest(object):
    def __init__(self, args, model, embeddings, vocabs, criterion, optimizer):
        super(TrainerTest, self).__init__()
        self.args = args
        self.model = model
        self.embeddings = embeddings
        self.vocabs = vocabs
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = 0

    def train_embeddings(self):
        for key in self.embeddings.keys():
            self.embeddings[key].train()
            self.embeddings[key].zero_grad()

    def set_embeddings_step(self):
        for key in self.embeddings.keys():
            if key == 'chars':
                continue

            for f in self.embeddings[key].parameters():
                f.data.sub_(f.grad.data * self.args.emblr)

            self.embeddings[key].zero_grad()

    def test_embeddings(self):
        for key in self.embeddings.keys():
            self.embeddings[key].eval()

    def get_char_vector(self, toks_sent):
        words = self.vocabs['toks'].convertToLabels(toks_sent.numpy(), None)
        char_vectors = []

        for word in words:
            if word == Constants.UNK_WORD:
                char_vectors.append(F.torch.zeros(self.vocabs['chars'].size()))
            else:
                char_vector = []
                for char in word:
                    if self.vocabs['chars'].getIndex(char) != None:
                        char_vector.append(self.vocabs['chars'].getIndex(char) )
                    else:
                        char_vector.append(0)

                char_vector = torch.tensor(char_vector)
                char_vectors.append(F.torch.sum(self.embeddings['chars'](char_vector), 0) / len(char_vector))

        return F.torch.unsqueeze(torch.stack(char_vectors), 1)

    def get_data(self, data, num_classes):
        tree, toks_sent, pos_sent, rels_sent, label = data
        toks_sent = Var(toks_sent)
        pos_sent = Var(pos_sent)
        rels_sent = Var(rels_sent)


        target = Var(utils.map_label_to_target(label, num_classes, self.vocabs['output']))

        toks_emb = F.torch.unsqueeze(self.embeddings['toks'](toks_sent), 1)
        pos_emb = F.torch.unsqueeze(self.embeddings['pos'](pos_sent), 1)
        rels_emb = F.torch.unsqueeze(self.embeddings['rels'](rels_sent), 1)
        chars_emb = self.get_char_vector(toks_sent)

        return tree, torch.cat((toks_emb, pos_emb, rels_emb, chars_emb), 2), target
        # return tree, torch.cat((toks_emb, pos_emb, rels_emb), 2), target

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.train_embeddings()

        self.optimizer.zero_grad()
        total_loss, k = 0.0, 0
        indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')

        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            tree, emb, target = self.get_data(dataset[indices[idx]], dataset.num_classes)

            output = self.model.forward(tree, emb, training=True)
            err = self.criterion(output, target.cuda())

            total_loss += err.item()
            err.backward()
            k += 1

            if k == self.args.batchsize:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.set_embeddings_step()
                k = 0

        self.epoch += 1
        return total_loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        self.test_embeddings()

        total_loss = 0
        predictions = torch.zeros(len(dataset))

        for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
            torch.no_grad()
            tree, emb, target = self.get_data(dataset[idx], dataset.num_classes)

            output = self.model.forward(tree, emb, training=False)
            err = self.criterion(output, target.cuda())
            total_loss += err.item()

            _, pred = torch.max(output, 1)

            predictions[idx] = pred.data.cpu()[0]
        return total_loss / len(dataset), predictions


class TrainerLEMMA(object):
    def __init__(self, args, model, embeddings, vocabs, criterion, optimizer):
        super(TrainerLEMMA, self).__init__()
        self.args = args
        self.model = model
        self.embeddings = embeddings
        self.vocabs = vocabs
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = 0

    def train_embeddings(self):
        for key in self.embeddings.keys():
            self.embeddings[key].train()
            self.embeddings[key].zero_grad()

    def set_embeddings_step(self):
        for key in self.embeddings.keys():
            if key == 'chars':
                continue

            for f in self.embeddings[key].parameters():
                f.data.sub_(f.grad.data * self.args.emblr)

            self.embeddings[key].zero_grad()

    def test_embeddings(self):
        for key in self.embeddings.keys():
            self.embeddings[key].eval()

    def get_char_vector(self, toks_sent):
        words = self.vocabs['toks'].convertToLabels(toks_sent.numpy(), None)
        char_vectors = []

        for word in words:
            if word == Constants.UNK_WORD:
                char_vectors.append(F.torch.zeros(self.vocabs['chars'].size()))
            else:
                char_vector = []
                for char in word:
                    if self.vocabs['chars'].getIndex(char) != None:
                        char_vector.append(self.vocabs['chars'].getIndex(char) )
                    else:
                        char_vector.append(0)

                char_vector = torch.tensor(char_vector)
                char_vectors.append(F.torch.sum(self.embeddings['chars'](char_vector), 0) / len(char_vector))

        return F.torch.unsqueeze(torch.stack(char_vectors), 1)

    def get_data(self, data, num_classes):
        tree, toks_sent, lem_sent, pos_sent, tag_sent, rels_sent, label = data
        toks_sent = Var(toks_sent)
        lem_sent = Var(lem_sent)
        pos_sent = Var(pos_sent)
        tag_sent = Var(tag_sent)
        rels_sent = Var(rels_sent)


        target = Var(utils.map_label_to_target(label, num_classes, self.vocabs['output']))

        toks_emb = F.torch.unsqueeze(self.embeddings['toks'](toks_sent), 1)
        lem_emb = F.torch.unsqueeze(self.embeddings['lem'](lem_sent), 1)
        pos_emb = F.torch.unsqueeze(self.embeddings['pos'](pos_sent), 1)
        tag_emb = F.torch.unsqueeze(self.embeddings['tag'](tag_sent), 1)
        rels_emb = F.torch.unsqueeze(self.embeddings['rels'](rels_sent), 1)
        chars_emb = self.get_char_vector(toks_sent)

        return tree, torch.cat((toks_emb, lem_emb, pos_emb, rels_emb, tag_emb, chars_emb), 2), target
        # return tree, torch.cat((toks_emb, pos_emb, rels_emb), 2), target

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.train_embeddings()

        self.optimizer.zero_grad()
        total_loss, k = 0.0, 0
        indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')

        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            tree, emb, target = self.get_data(dataset[indices[idx]], dataset.num_classes)

            output = self.model.forward(tree, emb, training=True)
            err = self.criterion(output, target.cuda())

            total_loss += err.item()
            err.backward()
            k += 1

            if k == self.args.batchsize:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.set_embeddings_step()
                k = 0

        self.epoch += 1
        return total_loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        self.test_embeddings()

        total_loss = 0
        predictions = torch.zeros(len(dataset))

        for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
            torch.no_grad()
            tree, emb, target = self.get_data(dataset[idx], dataset.num_classes)

            output = self.model.forward(tree, emb, training=True)
            err = self.criterion(output, target.cuda())
            total_loss += err.item()

            _, pred = torch.max(output, 1)

            predictions[idx] = pred.data.cpu()[0]
        return total_loss / len(dataset), predictions
