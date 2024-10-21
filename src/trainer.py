# third party
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from .temperature_scaling import ModelWithTemperature

from .hardness import *

def softmax(outputs):
    return (torch.exp(outputs.t()) / torch.sum(torch.exp(outputs), dim=1)).t()

def encode(targets, num_classes):
    encoded_targets = torch.zeros(targets.size(0), num_classes).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    encoded_targets.scatter_(1, targets.view(-1, 1).long(), 1).float()
    return encoded_targets

def cross_entropy(inp, target, num_classes):
    inp = softmax(inp)
    target = encode(target, num_classes)
    return torch.mean(-torch.sum(target * torch.log(inp), 1))

def convert_lst(l):
    if not isinstance(l, list):
        lst_scores = l.flatten().tolist()
    else:
        lst_scores = l
    return lst_scores

def accuracy(output, target, topk=(1,5,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# The PyTorchTrainer class is a helper class for training PyTorch models with various characterization
# methods.
class PyTorchTrainer:
    def __init__(
        self,
        model: nn.Module,
        alpha: nn.Module,
        beta: nn.Module,
        delta: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device = None,
        lr: float = 0.001,
        alpha_lr: float = 0.01,
        beta_lr: float = 0.01,
        delta_lr: float = 0.01,
        wd: float = 0.001,
        alpha_wd: float = 0.01,
        beta_wd: float = 0.01,
        delta_wd: float = 0.01,
        warmup: int = 0, 
        epochs: int = 10,
        total_samples: int = 10000,
        num_classes: int = 10,
        reweight: bool = True,
        clean_val: bool = True,
        calibrate: bool = True,
        metainfo: str = True,
        characterization_methods: list = [
            "aum",
            "data_uncert",
            "el2n",
            "grand",
            #"cleanlab",
            "forgetting",
            "vog",
            #"prototypicality",
            #"allsh",
            "loss",
            #"conf_agree",
            #"detector",
        ],
    ):
        """
        This is a constructor function that initializes various parameters for a machine learning model.

        Args:
          model (nn.Module): a PyTorch neural network model
          criterion (nn.Module): The loss function used to train the model. It is a nn.Module object.
          optimizer (optim.Optimizer): The optimizer is an object that specifies the algorithm used to
        update the parameters of the neural network during training. It is responsible for minimizing
        the loss function by adjusting the weights and biases of the model. In this code, the optimizer
        is an instance of the `optim.Optimizer` class from the Py
          device (torch.device): The device parameter specifies the device on which the model will be
        trained. It can be set to "cpu" or "cuda" depending on whether you want to train the model on
        CPU or GPU. If this parameter is not specified, the default device will be used.
          lr (float): learning rate for the optimizer
          epochs (int): The number of training epochs
          total_samples (int): The total number of samples in the dataset.
          num_classes (int): The number of classes in the classification problem. For example, if you
        are working on a dataset with images of cats and dogs, the number of classes would be 2.
          characterization_methods (list): This is a list of strings that represents the different HCMs to assess
        """

        # for memory reasons
        if "prototypicality" in characterization_methods and "ResNet" in str(model):
            characterization_methods.remove("prototypicality")

        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.lr = lr
        self.alpha_lr = alpha_lr
        self.beta_lr = beta_lr
        self.delta_lr = delta_lr
        self.wd = wd
        self.alpha_wd = alpha_wd
        self.beta_wd = beta_wd
        self.delta_wd = delta_wd
        self.epochs = epochs
        self.warmup = warmup
        self.total_samples = total_samples
        self.num_classes = num_classes
        self.reweight = reweight
        self.clean_val = clean_val
        self.calibrate = calibrate
        self.characterization_methods = characterization_methods
        self.metainfo = metainfo
        self.sigmoid = nn.Sigmoid()

    def fit(self, dataloader, dataloader_unshuffled, val_dataloader, val_dataloader_unshuffled, test_dataloader, test_dataloader_unshuffled, wandb_num=[0,0]):
        """
        This function trains a model and updates various HCMs
        Args:
          dataloader: A PyTorch DataLoader object that provides batches of data for training the model.
          dataloader_unshuffled: A dataloader object that provides unshuffled data for use in computing
        certain metrics during training and/or after training.
        """

        self.aum = (
            AUM_Class(save_dir=str(wandb_num[0]) + '-' + str(wandb_num[1])) 
            if "aum" in self.characterization_methods 
            else None
        )
        self.data_uncert = (
            DataIQ_Maps_Class(dataloader=dataloader_unshuffled)
            if "data_uncert" in self.characterization_methods
            else None
        )
        self.el2n = (
            EL2N_Class(dataloader=dataloader_unshuffled)
            if "el2n" in self.characterization_methods
            else None
        )
        self.grand = (
            GRAND_Class(dataloader=dataloader_unshuffled)
            if "grand" in self.characterization_methods
            else None
        )
        self.cleanlab = (
            Cleanlab_Class(dataloader=dataloader_unshuffled)
            if "cleanlab" in self.characterization_methods
            else None
        )

        self.forgetting = (
            Forgetting_Class(
                dataloader=dataloader_unshuffled, total_samples=self.total_samples
            )
            if "forgetting" in self.characterization_methods
            else None
        )

        self.vog = (
            VOG_Class(
                dataloader=dataloader_unshuffled, total_samples=self.total_samples
            )
            if "vog" in self.characterization_methods
            else None
        )

        self.prototypicality = (
            Prototypicality_Class(
                dataloader=dataloader_unshuffled, num_classes=self.num_classes
            )
            if "prototypicality" in self.characterization_methods
            else None
        )

        self.allsh = (
            AllSH_Class(dataloader=dataloader_unshuffled)
            if "allsh" in self.characterization_methods
            else None
        )

        self.loss = (
            Large_Loss_Class(dataloader=dataloader_unshuffled)
            if "loss" in self.characterization_methods
            else None
        )

        self.conf_agree = (
            Conf_Agree_Class(dataloader=dataloader_unshuffled)
            if "conf_agree" in self.characterization_methods
            else None
        )

        self.detector = (
            Detector_Class() if "detector" in self.characterization_methods else None
        )

        # Move model to device
        self.model.to(self.device)
        self.alpha.to(self.device)
        self.beta.to(self.device)
        self.delta.to(self.device)

        # Set model to training mode
        self.optimizer.lr = self.lr
        c = 0
        dictionary = {}
        for epoch in range(self.epochs):
            print('EPOCH ' + str(epoch))
            self.model.train()
            running_loss = 0.0
            val_running_loss = 0.0
            test_running_loss = 0.0
            running_observed_ce = 0.0
            val_running_observed_ce = 0.0
            running_true_ce = 0.0
            val_running_true_ce = 0.0
            test_running_true_ce = 0.0
            running_acc = 0.0
            val_running_acc = 0.0
            test_running_acc = 0.0
            running_top1_acc = 0.0
            val_running_top1_acc = 0.0
            test_running_top1_acc = 0.0
            running_top5_acc = 0.0
            val_running_top5_acc = 0.0
            test_running_top5_acc = 0.0
            dictionary[epoch] = {'index': [], 'predicted_output': [], 'max_output': [],
            'predicted_label': [], 'true_label': [], 'observed_label': [],
            'aum': [], 'dataiq': [], 'datamaps': [], 'dataiq_conf': [], 'datamaps_conf': [],
            'el2n': [], 'grand': [], 'forgetting': [], 'vog': [], 'loss': []}
            if self.reweight:
                dictionary[epoch]['alpha'] = []
                dictionary[epoch]['beta'] = []
                dictionary[epoch]['delta'] = []
                dictionary[epoch]['weight'] = []
            for i, data in enumerate(dataloader_unshuffled):
                inputs, true_label, observed_label, indices = data
                dictionary[epoch]['index'].extend(indices.flatten().tolist())
                dictionary[epoch]['true_label'].extend(true_label.flatten().tolist())
                dictionary[epoch]['observed_label'].extend(observed_label.flatten().tolist())
                if self.reweight:
                    dictionary[epoch]['alpha'].extend([self.alpha]*len(indices))
                    dictionary[epoch]['beta'].extend([self.beta]*len(indices))
                    dictionary[epoch]['delta'].extend([self.delta]*len(indices))

                inputs = inputs.to(self.device)
                true_label = true_label.to(self.device)
                observed_label = observed_label.to(self.device)

                self.optimizer.zero_grad()

                self.alpha.requires_grad = False
                self.beta.requires_grad = False
                self.delta.requires_grad = False

                outputs = self.model(inputs)

                if self.aum is not None:
                    self.aum.updates(
                        y_pred=outputs, y_batch=observed_label, sample_ids=indices
                    )

                outputs = outputs.float()
                dictionary[epoch]['predicted_label'].extend(torch.argmax(outputs, 1).flatten().tolist())
                observed_label = observed_label.long()

                correct_outputs, max_outputs, weights, train_loss = self.criterion(outputs, observed_label, epoch=epoch)
                dictionary[epoch]['predicted_output'].extend(correct_outputs.flatten().tolist())
                dictionary[epoch]['max_output'].extend(max_outputs.flatten().tolist())
                if self.reweight:
                    if epoch > self.warmup:
                        dictionary[epoch]['weight'].extend(weights.flatten().tolist())
                    else:
                        dictionary[epoch]['weight'].extend([1]*len(indices))

                acc = (torch.argmax(outputs, 1) == observed_label).type(torch.float)
                running_acc += acc.mean().item()

                topk_acc = accuracy(outputs, observed_label)
                top1_acc = topk_acc[0]
                top5_acc = topk_acc[1]
                running_top1_acc += top1_acc.mean().item()
                running_top5_acc += top5_acc.mean().item()
                
                train_observed_ce = cross_entropy(outputs, observed_label, self.num_classes)
                train_true_ce = cross_entropy(outputs, true_label, self.num_classes)
                running_observed_ce += train_observed_ce.mean().item()
                running_true_ce += train_true_ce.mean().item()

                train_loss.mean().backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                running_loss += train_loss.mean().item()

                for j, val_data in enumerate(val_dataloader_unshuffled):
                    val_inputs, val_true_label, val_observed_label, val_indices = val_data

                    if self.reweight and epoch > self.warmup:
                        self.alpha.requires_grad = True
                        self.beta.requires_grad = True
                        self.delta.requires_grad = True

                    val_inputs = val_inputs.to(self.device)
                    val_true_label = val_true_label.to(self.device)
                    val_observed_label = val_observed_label.to(self.device)

                    val_outputs = self.model(val_inputs)

                    val_outputs = val_outputs.float()
                    val_observed_label = val_observed_label.long()

                    if self.clean_val:
                        correct_outputs, max_outputs, weights, val_loss = self.criterion(val_outputs, val_true_label, epoch=epoch)
                        val_acc = (torch.argmax(val_outputs, 1) == val_true_label).type(torch.float)
                        val_topk_acc = accuracy(val_outputs, val_true_label)
                        val_observed_ce = cross_entropy(val_outputs, val_observed_label, self.num_classes)
                        val_true_ce = cross_entropy(val_outputs, val_true_label, self.num_classes)
                    else:
                        correct_outputs, max_outputs, weights, val_loss = self.criterion(val_outputs, val_observed_label, epoch=epoch)
                        val_acc = (torch.argmax(val_outputs, 1) == val_observed_label).type(torch.float)
                        val_topk_acc = accuracy(val_outputs, val_observed_label)
                        val_observed_ce = cross_entropy(val_outputs, val_observed_label, self.num_classes)
                        val_true_ce = cross_entropy(val_outputs, val_true_label, self.num_classes)
                    
                    val_running_acc += val_acc.mean().item()

                    val_top1_acc = val_topk_acc[0]
                    val_top5_acc = val_topk_acc[1]
                    val_running_top1_acc += val_top1_acc.mean().item()
                    val_running_top5_acc += val_top5_acc.mean().item()

                    val_running_observed_ce += val_observed_ce.mean().item()
                    val_running_true_ce += val_true_ce.mean().item()

                    val_loss.mean().backward()
                    val_running_loss += val_loss.mean().item()

                    if self.reweight and epoch > self.warmup:
                        with torch.no_grad():
                            self.alpha -= self.alpha_lr * (self.alpha.grad + self.alpha_wd*self.alpha)
                            self.alpha.data.clamp_(min=1.0)
                            self.alpha.grad.zero_()

                            self.beta -= self.beta_lr * (self.beta.grad + self.beta_wd*self.beta)
                            self.beta.data.clamp_(min=1.0)
                            self.beta.grad.zero_()

                            self.delta -= self.delta_lr * (self.delta.grad + self.delta_wd*self.delta)
                            self.delta.data.clamp_(min=1.0)
                            self.delta.grad.zero_()

                            wandb.log({"alpha": self.alpha.detach().item(), "step": c})
                            wandb.log({"beta": self.beta.detach().item(), "step": c})
                            wandb.log({"delta": self.delta.detach().item(), "step": c})
                            c += 1

                    self.optimizer.zero_grad()
                    
                    break

            if self.calibrate:
                scaled_model = ModelWithTemperature(self.model)
                _, before_temperature_nll, before_temperature_ece, after_temperature_nll, after_temperature_ece = scaled_model.set_temperature(val_dataloader)

            self.model.eval()
            for k, test_data in enumerate(test_dataloader_unshuffled):
                test_inputs, test_true_label, test_observed_label, test_indices = test_data

                test_inputs = test_inputs.to(self.device)
                test_true_label = test_true_label.to(self.device)
                test_observed_label = test_observed_label.to(self.device)
                if self.calibrate:
                    test_outputs = scaled_model.model(test_inputs)
                else:
                    test_outputs = self.model(test_inputs)

                test_outputs = test_outputs.float()
                test_observed_label = test_observed_label.long()
                test_true_label = test_true_label.long()
                correct_outputs, max_outputs, weights, test_loss = self.criterion(test_outputs, test_true_label, epoch=epoch)
                test_acc = (torch.argmax(test_outputs, 1) == test_true_label).type(torch.float)
                test_running_acc += test_acc.mean().item()

                test_topk_acc = accuracy(test_outputs, test_true_label)
                test_top1_acc = test_topk_acc[0]
                test_top5_acc = test_topk_acc[1]
                test_running_top1_acc += test_top1_acc.mean().item()
                test_running_top5_acc += test_top5_acc.mean().item()

                test_true_ce = cross_entropy(test_outputs, test_true_label, self.num_classes)
                test_running_true_ce += test_true_ce.mean().item()
                test_running_loss += test_loss.mean().item()

            self.model.train()
            epoch_loss = running_loss / len(dataloader)
            val_epoch_loss = val_running_loss / len(dataloader)
            test_epoch_loss = test_running_loss / len(test_dataloader)
            epoch_observed_ce = running_observed_ce / len(dataloader)
            val_epoch_observed_ce = val_running_observed_ce / len(dataloader)
            epoch_true_ce = running_true_ce / len(dataloader)
            val_epoch_true_ce = val_running_true_ce / len(dataloader)
            test_epoch_true_ce = test_running_true_ce / len(test_dataloader)
            epoch_acc = running_acc / len(dataloader)
            val_epoch_acc = val_running_acc / len(dataloader)
            test_epoch_acc = test_running_acc / len(test_dataloader)
            epoch_top1_acc = running_top1_acc / len(dataloader)
            val_epoch_top1_acc = val_running_top1_acc / len(dataloader)
            test_epoch_top1_acc = test_running_top1_acc / len(test_dataloader)
            epoch_top5_acc = running_top5_acc / len(dataloader)
            val_epoch_top5_acc = val_running_top5_acc / len(dataloader)
            test_epoch_top5_acc = test_running_top5_acc / len(test_dataloader)
            if self.calibrate:
                wandb.log({"before_temperature_nll": before_temperature_nll, "epoch": epoch})
                wandb.log({"before_temperature_ece": before_temperature_ece, "epoch": epoch})
                wandb.log({"after_temperature_nll": after_temperature_nll, "epoch": epoch})
                wandb.log({"after_temperature_ece": after_temperature_ece, "epoch": epoch})
            wandb.log({"train_loss": epoch_loss, "epoch": epoch})
            wandb.log({"val_loss": val_epoch_loss, "epoch": epoch})
            wandb.log({"test_loss": test_epoch_loss, "epoch": epoch})
            wandb.log({"train_observed_ce": epoch_observed_ce, "epoch": epoch})
            wandb.log({"val_observed_ce": val_epoch_observed_ce, "epoch": epoch})
            wandb.log({"train_true_ce": epoch_true_ce, "epoch": epoch})
            wandb.log({"val_true_ce": val_epoch_true_ce, "epoch": epoch})
            wandb.log({"test_true_ce": test_epoch_true_ce, "epoch": epoch})
            wandb.log({"train_acc": epoch_acc, "epoch": epoch})
            wandb.log({"val_acc": val_epoch_acc, "epoch": epoch})
            wandb.log({"test_acc": test_epoch_acc, "epoch": epoch})
            wandb.log({"train_top1_acc": epoch_top1_acc, "epoch": epoch})
            wandb.log({"val_top1_acc": val_epoch_top1_acc, "epoch": epoch})
            wandb.log({"test_top1_acc": test_epoch_top1_acc, "epoch": epoch})
            wandb.log({"train_top5_acc": epoch_top5_acc, "epoch": epoch})
            wandb.log({"val_top5_acc": val_epoch_top5_acc, "epoch": epoch})
            wandb.log({"test_top5_acc": test_epoch_top5_acc, "epoch": epoch})
            print(f"Epoch {epoch+1}/{self.epochs}: Train Loss={epoch_loss:.4f} | Val Loss={val_epoch_loss:.4f} | Test Loss={test_epoch_loss:.4f}")

            # streamline repeated computation across methods
            print('compute scores')
            if any(
                method in self.characterization_methods
                for method in ["el2n", "cleanlab", "forgetting", "loss"]
            ):
                logits, targets, probs, indices = self.get_intermediate_outputs(
                    net=self.model, device=self.device, dataloader=dataloader_unshuffled
                )

            if self.data_uncert is not None:
                self.data_uncert.updates(net=self.model, device=self.device)
                self.data_uncert.compute_scores(datamaps=False)
                dictionary[epoch]['dataiq'].extend(convert_lst(self.data_uncert._scores[0]))
                dictionary[epoch]['dataiq_conf'].extend(convert_lst(self.data_uncert._scores[1]))
                self.data_uncert.compute_scores(datamaps=True)
                dictionary[epoch]['datamaps'].extend(convert_lst(self.data_uncert._scores[0]))
                dictionary[epoch]['datamaps_conf'].extend(convert_lst(self.data_uncert._scores[1]))

            if self.el2n is not None:
                self.el2n.updates(logits=logits, targets=targets)
                self.el2n.compute_scores()
                dictionary[epoch]['el2n'].extend(convert_lst(self.el2n._scores))

            if self.forgetting is not None:
                self.forgetting.updates(
                    logits=logits, targets=targets, probs=probs, indices=indices
                )
                self.forgetting.compute_scores()
                dictionary[epoch]['forgetting'].extend(convert_lst(self.forgetting._scores))

            if self.grand is not None:
                self.grand.updates(net=self.model, device=self.device)
                self.grand.compute_scores()
                dictionary[epoch]['grand'].extend(convert_lst(self.grand._scores))

            if self.vog is not None: # and epoch % 2 == 0 and epoch < 6:
                self.vog.updates(net=self.model, device=self.device)
                self.vog.compute_scores()
                dictionary[epoch]['vog'].extend(convert_lst(self.vog._scores))

            if self.loss is not None:
                self.loss.updates(logits=logits, targets=targets)
                self.loss.compute_scores()
                dictionary[epoch]['loss'].extend(convert_lst(self.loss._scores))

        # These HCMs are applied after training
        if self.prototypicality is not None:
            self.prototypicality.updates(net=self.model, device=self.device)

        if self.allsh is not None:
            self.allsh.updates(net=self.model, device=self.device)

        if self.conf_agree is not None:
            self.conf_agree.updates(net=self.model, device=self.device)

        if self.cleanlab is not None:
            self.cleanlab.updates(logits=logits, targets=targets, probs=probs)

        if self.detector is not None:
            self.detector.updates(
                data_uncert_class=self.data_uncert.data_eval, device=self.device
            )

        df = pd.DataFrame(dictionary)
        store = pd.HDFStore(self.metainfo.replace(':', '').replace('.', '') + '.h5')
        store['df'] = df  # save it
        #store['df']  # load it

    def get_intermediate_outputs(self, net, dataloader, device):
        """
        This function takes a neural network, a dataloader, and a device, and returns the logits, targets,
        probabilities, and indices of the intermediate outputs of the network on the given data.

        Args:
          net: a PyTorch neural network model
          dataloader: A PyTorch DataLoader object that provides batches of data to the model for inference
        or evaluation.
          device: The device on which the computation is being performed, such as "cpu" or "cuda".

        Returns:
          four tensors: logits, targets, probs, and indices.
        """
        logits_array = []
        targets_array = []
        indices_array = []
        with torch.no_grad():
            net.eval()
            for x, _, y, indices in dataloader:
                x = x.to(device)
                y = y.to(device)
                outputs = net(x)
                logits_array.append(outputs)
                targets_array.append(y.view(-1))
                indices_array.append(indices.view(-1))

            logits = torch.cat(logits_array, dim=0)
            targets = torch.cat(targets_array, dim=0)
            indices = torch.cat(indices_array, dim=0)
            probs = torch.nn.functional.softmax(logits, dim=1)

        logits = logits.float()  # Ensure the outputs are float
        targets = targets.long()  # Ensure the labels are long
        return logits, targets, probs, indices

    def get_hardness_methods(self):
        """
        This function returns a dictionary of HCMs and their corresponding instantiations

        Returns:
          a dictionary containing the hardness values for each characterization method that has a non-None
        value. The keys of the dictionary are the method names and the values are the corresponding instantiations
        """
        hardness_dict = {}

        for method in self.characterization_methods:
            if self.el2n is not None and method == "el2n":
                hardness_dict[method] = self.el2n

            if self.el2n is not None and method == "forgetting":
                hardness_dict[method] = self.forgetting

            if self.cleanlab is not None and method == "cleanlab":
                hardness_dict[method] = self.cleanlab

            if self.grand is not None and method == "grand":
                hardness_dict[method] = self.grand

            if self.aum is not None and method == "aum":
                hardness_dict[method] = self.aum

            if self.data_uncert is not None and method == "data_uncert":
                hardness_dict["dataiq"] = self.data_uncert
                hardness_dict["datamaps"] = self.data_uncert

            if self.vog is not None and method == "vog":
                hardness_dict[method] = self.vog

            if self.prototypicality is not None and method == "prototypicality":
                hardness_dict[method] = self.prototypicality

            if self.allsh is not None and method == "allsh":
                hardness_dict[method] = self.allsh

            if self.loss is not None and method == "loss":
                hardness_dict[method] = self.loss

            if self.conf_agree is not None and method == "conf_agree":
                hardness_dict[method] = self.conf_agree

            if self.detector is not None and method == "detector":
                hardness_dict[method] = self.detector

        return hardness_dict
