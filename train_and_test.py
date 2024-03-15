import OpenFaceDataset as dataloader
import ensemble_model as model
import time
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import torch
from sklearn.metrics import mean_squared_error
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import hydra
from omegaconf import DictConfig

accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
devices = 1 if torch.cuda.is_available() else None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SwapHyperparametersCallback(Callback):
    def __init__(self, swap_epoch=60):
        self.swap_epoch = swap_epoch
        self.beta = 1.0
        self.gamma = 0.0
        self.swap = False  

    def on_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch != 0 and epoch % self.swap_epoch == 0:
            self.swap = not self.swap
            
        if self.swap:
            self.beta, self.gamma = self.gamma, self.beta
            pl_module.update_hyperparameters(beta=self.beta, gamma=self.gamma)
            print(f"Epoch {epoch}: Swap beta and gamma -> beta={self.beta}, gamma={self.gamma}")

@hydra.main(config_path="configs", config_name="configures", version_base="1.1")
def train(cfg: DictConfig):
    data_module = dataloader.create_data_module(cfg)

    print("Number of features are: ", data_module.train_dataset[0]["original_sequence"].shape[1])

    features = ""
    for i in cfg.data.attributes:
        features += "-"
        features += i

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint.dirpath,
        filename="best-" + str(cfg.data.frame_size) + features + unique_key,
        save_top_k=1,
        verbose=True,
        monitor="validation_loss",
        mode="min"
    )

    pred_model = model.EnsembleModel(
        data_module.train_dataset[0]["original_sequence"].shape[1],
        cfg.model)
    print(pred_model)
    
#     swap_hyperparams_callback = SwapHyperparametersCallback()

    trainer = pl.Trainer(
        callbacks=checkpoint_callback, # [checkpoint_callback, swap_hyperparams_callback]
        max_epochs=cfg.model.train.n_epochs,
        accelerator=accelerator,
        devices=devices,
        deterministic=True,
        sync_batchnorm=True if torch.cuda.is_available() else False
        )
    trainer.fit(pred_model, data_module)


@hydra.main(config_path="configs", config_name="configures", version_base="1.1")
def test(cfg: DictConfig):
    data_module = dataloader.create_data_module(cfg)
    features = ""

    for i in cfg.data.attributes:
        features += "-"
        features += i
#     best_model_path = cfg.checkpoint.dirpath+"/best-" + str(cfg.data.frame_size) + features + unique_key + ".ckpt"
    best_model_path = cfg.checkpoint.dirpath+ "/best-100-gaze_seg-aus_seg_15_00:08:11_2024.ckpt"
    trained_model = model.EnsembleModel.load_from_checkpoint(
        best_model_path,
        n_features=data_module.train_dataset[0]["original_sequence"].shape[1],
        hparam=cfg.model).to(device)

    trained_model.freeze()
    predictions = []
    labels = []
    total = 0
    for item in tqdm(data_module.val_dataloader()):
        total += 1
        sequence = item["original_sequence"].to(device)
        label = item["original_label"].to(device)
        name = item["video_name"]
        out_transformer, out_mlp = trained_model(sequence.cuda(), label.cuda())
        # save picture of time dim attention:
        # title = str(name)[2:-6] + " predition: " + str(round(out_mlp[0].cpu().numpy().mean(),2)) + " OpenFace_features: " + str(round(OpenFace_features.item(),2))
        # save_image(out_weights.cpu().numpy().flatten(), title, root_path='/content/drive/MyDrive/project/results/',unique_key = "",value_name = "The Temporal Attentions ")
        predictions.append(out_mlp[0].cpu().numpy().mean())
        labels.append(label.item())

    MSE_test = "---- MSE of " + str(cfg.data.frame_size) + " frame with features: " + features + "- is " + str(
        round(mean_squared_error(labels, predictions), 4)) + " ----"

    print(MSE_test)


def save_image(data, file_name, root_path='/raid/yifengliu/COMP8755/results/', leg='leg', c='#4e9dec', unique_key="",
               value_name="the Engagement Scores "):
    save_path = str(root_path + file_name + unique_key + ".png")
    plt.plot(data, color=c, alpha=0.8, label=leg, marker="o", linestyle=":")
    plt.title(value_name + "for " + file_name)
    if os.path.exists(save_path):
        os.remove(save_path)
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    current_time = time.ctime()
    unique_key = current_time[7:].replace(" ", "_")
    unique_key += ""
#     train()
    test()
