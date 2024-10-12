# third party
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .hardness import *


# The PyTorchTrainer class is a helper class for training PyTorch models with various characterization
# methods.
class PyTorchTrainer:
    def __init__(
        self,
        model: nn.Module,
        alpha: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device = None,
        lr: float = 0.001,
        epochs: int = 10,
        total_samples: int = 10000,
        num_classes: int = 10,
        reweight: bool = True,
        characterization_methods: list = [
            "aum",
            "data_uncert",
            "el2n",
            "grand",
            "cleanlab",
            "forgetting",
            "vog",
            "prototypicality",
            "allsh",
            "loss",
            "conf_agree",
            "detector",
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
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.total_samples = total_samples
        self.num_classes = num_classes
        self.reweight = reweight
        self.characterization_methods = characterization_methods

    def fit(self, dataloader, dataloader_unshuffled, val_dataloader, val_dataloader_unshuffled, test_dataloader, test_dataloader_unshuffled, wandb_num=[0,0]):
        """
        This function trains a model and updates various HCMs
        Args:
          dataloader: A PyTorch DataLoader object that provides batches of data for training the model.
          dataloader_unshuffled: A dataloader object that provides unshuffled data for use in computing
        certain metrics during training and/or after training.
        """

        self.aum = AUM_Class(save_dir=str(wandb_num[0]) + '-' + str(wandb_num[1])) if "aum" in self.characterization_methods else None
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

        # Set model to training mode
        self.optimizer.lr = self.lr
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            val_running_loss = 0.0
            test_running_loss = 0.0
            for i, data in enumerate(dataloader):
                inputs, true_label, observed_label, indices = data

                inputs = inputs.to(self.device)
                true_label = true_label.to(self.device)
                observed_label = observed_label.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)

                #if self.aum is not None:
                #    self.aum.updates(
                #        y_pred=outputs, y_batch=observed_label, sample_ids=indices
                #    )

                outputs = outputs.float()  # Ensure the outputs are float
                observed_label = observed_label.long()  # Ensure the labels are long
                train_loss = self.criterion(outputs, observed_label)
                print('TRAIN')
                print(train_loss)

                train_loss.mean().backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.reweight:
                    self.alpha.grad.zero_()

                if self.reweight and (torch.isnan(self.alpha) or torch.isnan(self.alpha.grad)):
                    break

                for j, val_data in enumerate(val_dataloader):
                    val_inputs, val_true_label, val_observed_label, val_indices = val_data

                    val_inputs = val_inputs.to(self.device)
                    val_true_label = val_true_label.to(self.device)
                    val_observed_label = val_observed_label.to(self.device)

                    val_outputs = self.model(val_inputs)

                    val_outputs = val_outputs.float()
                    val_observed_label = val_observed_label.long()
                    val_loss = self.criterion(val_outputs, val_observed_label)
                    print('VAL')
                    print(val_loss)
                    val_loss.mean().backward()

                    if self.reweight:
                        with torch.no_grad():
                            self.alpha -= self.alpha.grad
                            self.alpha.grad.zero_()
                            u = 1

                    val_running_loss += val_loss.mean().item()

                    break

                running_loss += train_loss.mean().item()

            self.model.eval()
            for k, test_data in enumerate(test_dataloader):
                test_inputs, test_true_label, test_observed_label, test_indices = test_data

                test_inputs = test_inputs.to(self.device)
                test_true_label = test_true_label.to(self.device)
                test_observed_label = test_observed_label.to(self.device)
                test_outputs = self.model(test_inputs)

                test_outputs = test_outputs.float()  # Ensure the outputs are float
                test_observed_label = test_observed_label.long()  # Ensure the labels are long
                test_loss = self.criterion(test_outputs, test_true_label)
                test_running_loss += test_loss.mean().item()

            self.model.train()
            epoch_loss = running_loss / len(dataloader)
            val_epoch_loss = val_running_loss / len(val_dataloader)
            test_epoch_loss = test_running_loss / len(test_dataloader)
            wandb.log({"train_loss": epoch_loss, "epoch": epoch})
            wandb.log({"val_loss": val_epoch_loss, "epoch": epoch})
            wandb.log({"test_loss": test_epoch_loss, "epoch": epoch})
            print(f"Epoch {epoch+1}/{self.epochs}: Train Loss={epoch_loss:.4f} | Val Loss={val_epoch_loss:.4f} | Test Loss={test_epoch_loss:.4f}")

            # streamline repeated computation across methods
            if any(
                method in self.characterization_methods
                for method in ["el2n", "cleanlab", "forgetting", "loss"]
            ):
                logits, targets, probs, indices = self.get_intermediate_outputs(
                    net=self.model, device=self.device, dataloader=dataloader_unshuffled
                )

            if self.data_uncert is not None:
                print("data_uncert compute")
                self.data_uncert.updates(net=self.model, device=self.device)

            if self.el2n is not None:
                print("el2n")
                self.el2n.updates(logits=logits, targets=targets)

            if self.forgetting is not None:
                print("forgetting")
                self.forgetting.updates(
                    logits=logits, targets=targets, probs=probs, indices=indices
                )

            if self.grand is not None:
                print("grand")
                self.grand.updates(net=self.model, device=self.device)

            if self.vog is not None and epoch % 2 == 0 and epoch < 6:
                print("vog")
                self.vog.updates(net=self.model, device=self.device)

            if self.loss is not None:
                print("loss")
                self.loss.updates(logits=logits, targets=targets)

        # These HCMs are applied after training
        if self.prototypicality is not None:
            print("prototypicality")
            self.prototypicality.updates(net=self.model, device=self.device)

        if self.allsh is not None:
            print("allsh")
            self.allsh.updates(net=self.model, device=self.device)

        if self.conf_agree is not None:
            print("conf_agree")
            self.conf_agree.updates(net=self.model, device=self.device)

        if self.cleanlab is not None:
            print("cleanlab")
            self.cleanlab.updates(logits=logits, targets=targets, probs=probs)

        if self.detector is not None:
            print("detector")
            self.detector.updates(
                data_uncert_class=self.data_uncert.data_eval, device=self.device
            )

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
