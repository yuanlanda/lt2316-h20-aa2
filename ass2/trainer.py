
import os
import torch


class Trainer:


    def __init__(self, dump_folder="/tmp/ass2_models/"):
        self.dump_folder = dump_folder
        os.makedirs(dump_folder, exist_ok=True)


    def save_model(self, model, optimizer, loss, epoch, scores, hyperparamaters, model_name):
        # More info about saving and loading here:
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
        save_dict = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'hyperparamaters': hyperparamaters,
                        'loss': loss,
                        'scores': scores
                        }

        torch.save(save_dict, os.path.join(self.dump_folder, model_name + ".pt"))


    def load_model(self):
        # Finish this function so that it loads a trained model
        pass


    def train(self, train_X, train_y, val_X, val_y, model_class, hyperparamaters):
        # Finish this function so that it trains and saves a model.
        pass


    def test(self, test_X, test_y, model_class, best_model_path):
        # Finish this function so that it loads a model, test is and print results.
        pass
