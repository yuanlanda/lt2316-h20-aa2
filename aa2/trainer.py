
import os
import torch


class Trainer:


    def __init__(self, dump_folder="/tmp/aa2_models/"):
        self.dump_folder = dump_folder
        os.makedirs(dump_folder, exist_ok=True)


    def save_model(self, epoch, model, optimizer, loss, scores, hyperparamaters, model_name):
        # epoch = epoch
        # model =  a train pytroch model
        # optimizer = a pytorch Optimizer
        # loss = loss (detach it from GPU)
        # scores = dict where keys are names of metrics and values the value for the metric
        # hyperparamaters = dict of hyperparamaters
        # model_name = name of the model you have trained, make this name unique for each hyperparamater.  I suggest you name them:
        # model_1, model_2 etc 
        #  
        #
        # More info about saving and loading here:
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training

        save_dict = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'hyperparamaters': hyperparamaters,
                        'loss': loss,
                        'scores': scores,
                        'model_name': model_name
                        }

        torch.save(save_dict, os.path.join(self.dump_folder, model_name + ".pt"))


    def load_model(self):
        # Finish this function so that it loads a model and return the appropriate variables
        pass


    def train(self, train_X, train_y, val_X, val_y, model_class, hyperparamaters):
        # Finish this function so that it set up model then trains and saves it.
        pass


    def test(self, test_X, test_y, model_class, best_model_path):
        # Finish this function so that it loads a model, test is and print results.
        pass
