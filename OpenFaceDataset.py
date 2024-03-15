import os
import math
import random
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from tsfresh.feature_extraction import feature_calculators
from torchsampler import ImbalancedDatasetSampler
import hydra
from omegaconf import DictConfig

class OpenFaceDatasetProcessData(Dataset):
    def __init__(self, cfg: DictConfig, case):
        pl.seed_everything(cfg.model.seed, workers=True)
        root = cfg.data.root
        l_dir = cfg.data.l_dir

        self.case = case
        self.file_list = []
        self.label_list = []
        self.name_list = []
        self.columns = []
        self.level = cfg.data.level
        self.dir_root = cfg.data.dir_root
        labels = pd.read_csv(l_dir, header=None)
        self.frame_size = cfg.data.frame_size
        self.step_size = cfg.data.step_size
        self.gaze_range = cfg.data.gaze_range
        self.head_range = cfg.data.head_range
        self.rot_range = cfg.data.rot_range
        self.aus_range = cfg.data.aus_range
        self.attributes = cfg.data.attributes
        self.functions = cfg.data.functions
        self.n_samples = cfg.data.n_samples

        for face_feature in os.listdir(root + self.case):
            # if "txt" not in face_feature:
            #     face_feature = face_feature + ".txt"

            label = labels[labels[0] == face_feature.split(".")[0]]
            if len(label) == 0:
                continue
            self.label_list.append(label[1].values[0])
            self.name_list.append(face_feature)
            self.file_list.append(os.path.join(root + self.case, face_feature))

        self.col_names = pd.read_csv(self.file_list[0], delimiter=',').columns
        self.all_features, self.feature_names = self.get_feature()

    def get_feature(self):
        features = []

        def norm_rad(x):
            return math.sin(math.radians(x))

        print("Start loading data:")
        for idx in tqdm(range(len(self.label_list))):
            # segment video to 10 segments, return features
            file_dir, label = self.file_list[idx], self.label_list[idx]
            # print(file_dir)
            v_data = pd.read_csv(file_dir, delimiter=',')
            v_data = np.array(v_data)
            v_data = np.delete(v_data, 0, 0)  # delete table caption
            v_data = v_data.astype(np.float)
            # remove nan
            v_data = v_data[~np.isnan(v_data).any(axis=1)]

            m_total = np.mean(v_data, axis=0)

            interval = int(v_data.shape[0] / self.frame_size)
            feature = []
            for i in range(self.frame_size):
                seg = v_data[i * interval:int((i + self.step_size) * interval), :]
                gaze_seg = seg[:, self.gaze_range[0]:self.gaze_range[1]]
                head_seg = seg[:, self.head_range[0]:self.head_range[1]]
                rot_seg = seg[:, self.rot_range[0]:self.rot_range[1]]
                aus_seg = seg[:, self.aus_range[0]:self.aus_range[1]]

                gaze_cols = self.col_names[self.gaze_range[0]:self.gaze_range[1]]
                head_cols = self.col_names[self.head_range[0]:self.head_range[1]]
                rot_cols = self.col_names[self.rot_range[0]:self.rot_range[1]]
                aus_cols = self.col_names[self.aus_range[0]:self.aus_range[1]]

                selected_feature = []
                selected_feature_names = []
                for att in self.attributes:

                    if att == "gaze_seg":

                        gaze = np.mean(np.abs(seg[:, self.gaze_range[0]:self.gaze_range[1]] - m_total[
                                                                                              self.gaze_range[0]:
                                                                                              self.gaze_range[1]]),
                                       axis=0)
                        selected_feature.append(gaze)
                        for i in gaze_cols:
                            selected_feature_names.append(str(i) + "_" + "separate")
                        gaze_max = np.max(np.abs(seg[:, self.gaze_range[0]:self.gaze_range[1]] - m_total[
                                                                                                 self.gaze_range[0]:
                                                                                                 self.gaze_range[1]]),
                                          axis=0)
                        selected_feature.append(gaze_max)
                        for i in gaze_cols:
                            selected_feature_names.append("max_" + str(i) + "_" + "separate")

                    elif att == "head_seg":
                        head = np.mean(np.abs(seg[:, self.head_range[0]:self.head_range[1]] - m_total[
                                                                                              self.head_range[0]:
                                                                                              self.head_range[1]]),
                                       axis=0)
                        selected_feature.append(head)
                        for i in head_cols:
                            selected_feature_names.append(str(i) + "_" + "separate")
                        head_max = np.max(np.abs(seg[:, self.head_range[0]:self.head_range[1]] - m_total[
                                                                                                 self.head_range[0]:
                                                                                                 self.head_range[1]]),
                                          axis=0)
                        selected_feature.append(head_max)
                        for i in head_cols:
                            selected_feature_names.append("max_" + str(i) + "_" + "separate")
                    elif att == "aus_seg":

                        # ====
                        # au = np.mean(np.abs(seg[:,self.aus_range[0]:self.aus_range[1]] - m_total[self.aus_range[0]:self.aus_range[1]]), axis=0)
                        # selected_feature.append(au)
                        # for i in aus_cols:

                        #     selected_feature_names.append(str(i)+ "_"+"separate")

                        # =====

                        au_max = np.max(np.abs(
                            seg[:, self.aus_range[0]:self.aus_range[1]] - m_total[self.aus_range[0]:self.aus_range[1]]),
                            axis=0)
                        selected_feature.append(au_max)
                        for i in aus_cols:
                            selected_feature_names.append("max_" + str(i) + "_" + "separate")

                    for func in self.functions:
                        method_to_call = getattr(feature_calculators, func)
                        selected_feature.append(np.apply_along_axis(method_to_call, 0, locals()[att]))

                        if att == "gaze_seg":
                            for i in gaze_cols:
                                selected_feature_names.append(str(i) + "_" + func[:3])

                        elif att == "head_seg":
                            for i in head_cols:
                                selected_feature_names.append(str(i) + "_" + func[:3])
                        elif att == "aus_seg":
                            for i in aus_cols:
                                selected_feature_names.append(str(i) + "_" + func[:3])

                feature.append(torch.FloatTensor(np.concatenate(selected_feature)))
            features.append(feature)
        selected = "_".join(self.attributes)
        with open(self.dir_root + self.case + '_features_' + str(self.frame_size) + "_" + selected + '.txt',
                  'wb') as fp:
            pickle.dump(features, fp)
        with open(self.dir_root + self.case + '_label_' + str(self.frame_size) + "_" + selected + '.txt',
                  'wb') as fp:
            pickle.dump(self.label_list, fp)
        with open(self.dir_root + self.case + '_names_' + str(self.frame_size) + "_" + selected + '.txt',
                  'wb') as fp:
            pickle.dump(self.name_list, fp)
        with open(self.dir_root + self.case + '_selected_feature_names_' + str(
                self.frame_size) + "_" + selected + '.txt', 'wb') as fp:
            pickle.dump(selected_feature_names, fp)

        return features, selected_feature_names

    def __getitem__(self, idx):
        x, y = self.all_features[idx], self.label_list[idx]
        return x, y

    def __len__(self):
        return len(self.label_list)

    def get_labels(self):

        return self.label_list


class OpenFaceDataset(Dataset):
    def __init__(self, cfg: DictConfig, case):
        pl.seed_everything(cfg.model.seed, workers=True)

        self.case = case
        self.level = cfg.data.level
        self.dir_root = cfg.data.dir_root
        self.frame_size = cfg.data.frame_size
        self.attributes = cfg.data.attributes
        self.n_samples = cfg.data.n_samples
        self.all_features, self.label_list, self.name_list, self.selected_feature_names = self.get_feature()
        self.label_dict = self.group_labels()

    def group_labels(self):

        sequences_dict = {}
        zero = []
        one = []
        two = []
        three = []

        for i in range(len(self.label_list)):
            if abs(self.label_list[i] - 0) < 0.01:
                zero.append(i)
            elif abs(self.label_list[i] - 0.33) < 0.01:
                one.append(i)

            elif abs(self.label_list[i] - 0.66) < 0.01:
                two.append(i)

            elif abs(self.label_list[i] - 1) < 0.01:
                three.append(i)

        sequences_dict["zero"] = zero
        sequences_dict["one"] = one
        sequences_dict["two"] = two
        sequences_dict["three"] = three

        return sequences_dict

    def get_feature(self):

        selected = "_".join(self.attributes)

        with open(self.dir_root + self.case + '_features_' + str(self.frame_size) + "_" + selected + '.txt',
                  'rb') as fp:
            all_features = pickle.load(fp)
        with open(self.dir_root + self.case + '_label_' + str(self.frame_size) + "_" + selected + '.txt',
                  'rb') as fp:
            label_list = pickle.load(fp)
        with open(self.dir_root + self.case + '_names_' + str(self.frame_size) + "_" + selected + '.txt',
                  'rb') as fp:
            name_list = pickle.load(fp)
        with open(self.dir_root + self.case + '_selected_feature_names_' + str(
                self.frame_size) + "_" + selected + '.txt', 'rb') as fp:
            selected_feature_names = pickle.load(fp)

        return all_features, label_list, name_list, selected_feature_names

    def __getitem__(self, idx):

        original_x, original_y, video_name = self.all_features[idx], torch.tensor(self.label_list[idx]).float(), \
            self.name_list[idx]

        if idx in self.label_dict["zero"]:

            positive_idxs = []
            for i in range(self.n_samples):
                positive_idxs.append(random.choice(self.label_dict["zero"]))

            negative_idx_1s = []
            for i in range(self.n_samples):
                negative_idx_1s.append(random.choice(self.label_dict["one"]))
            negative_idx_2s = []
            for i in range(self.n_samples):
                negative_idx_2s.append(random.choice(self.label_dict["two"]))
            negative_idx_3s = []
            for i in range(self.n_samples):
                negative_idx_3s.append(random.choice(self.label_dict["three"]))



        elif idx in self.label_dict["one"]:

            positive_idxs = []
            for i in range(self.n_samples):
                positive_idxs.append(random.choice(self.label_dict["one"]))
            negative_idx_1s = []
            for i in range(self.n_samples):
                negative_idx_1s.append(random.choice(self.label_dict["two"]))
            negative_idx_2s = []
            for i in range(self.n_samples):
                negative_idx_2s.append(random.choice(self.label_dict["three"]))
            negative_idx_3s = None


        elif idx in self.label_dict["two"]:

            positive_idxs = []
            for i in range(self.n_samples):
                positive_idxs.append(random.choice(self.label_dict["two"]))
            negative_idx_1s = []
            for i in range(self.n_samples):
                negative_idx_1s.append(random.choice(self.label_dict["one"]))
            negative_idx_2s = []
            for i in range(self.n_samples):
                negative_idx_2s.append(random.choice(self.label_dict["zero"]))
            negative_idx_3s = None


        else:

            positive_idxs = []
            for i in range(self.n_samples):
                positive_idxs.append(random.choice(self.label_dict["three"]))
            negative_idx_1s = []
            for i in range(self.n_samples):
                negative_idx_1s.append(random.choice(self.label_dict["two"]))
            negative_idx_2s = []
            for i in range(self.n_samples):
                negative_idx_2s.append(random.choice(self.label_dict["one"]))
            negative_idx_3s = []
            for i in range(self.n_samples):
                negative_idx_3s.append(random.choice(self.label_dict["zero"]))

        positive_xs, positive_ys = [self.all_features[idx] for idx in positive_idxs], [
            torch.tensor(self.label_list[idx]).float() for idx in positive_idxs]
        negative_x_1s, negative_y_1s = [self.all_features[idx] for idx in negative_idx_1s], [
            torch.tensor(self.label_list[idx]).float() for idx in negative_idx_1s]
        negative_x_2s, negative_y_2s = [self.all_features[idx] for idx in negative_idx_2s], [
            torch.tensor(self.label_list[idx]).float() for idx in negative_idx_2s]

        def shape_data(x):
            data = torch.zeros((self.frame_size, len(x[0])))
            for i in range(self.frame_size):
                data[i, :] = x[i]
            return data

        if negative_idx_3s is not None:

            negative_x_3s, negative_y_3s = [self.all_features[idx] for idx in negative_idx_3s], [
                torch.tensor(self.label_list[idx]).float() for idx in negative_idx_3s]
            negative_sequence_3s = [shape_data(x) for x in negative_x_3s]

        else:
            negative_x_3s, negative_y_3s = None, None

        positive_sequences = [shape_data(x) for x in positive_xs]

        negative_sequence_1s = [shape_data(x) for x in negative_x_1s]

        negative_sequence_2s = [shape_data(x) for x in negative_x_2s]

        if negative_y_3s is None:
            return dict(
                original_sequence=shape_data(original_x),
                original_label=original_y,
                video_name=video_name,
                positive_sequences=positive_sequences,
                positive_labels=positive_ys,
                negative_sequence_1s=negative_sequence_1s,
                negative_label_1s=negative_y_1s,
                negative_sequence_2s=negative_sequence_2s,
                negative_label_2s=negative_y_2s,
                negative_sequence_3s=negative_sequence_2s,
                negative_label_3s=negative_y_2s,
                feature_names=self.selected_feature_names
            )

        return dict(
            original_sequence=shape_data(original_x),
            original_label=original_y,
            video_name=video_name,
            positive_sequences=positive_sequences,
            positive_labels=positive_ys,
            negative_sequence_1s=negative_sequence_1s,
            negative_label_1s=negative_y_1s,
            negative_sequence_2s=negative_sequence_2s,
            negative_label_2s=negative_y_2s,
            negative_sequence_3s=negative_sequence_3s,
            negative_label_3s=negative_y_3s,
            feature_names=self.selected_feature_names
        )

    def __len__(self):
        return len(self.label_list)

    def get_labels(self):

        return self.label_list


class OpenFaceDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, cfg: DictConfig):
        super().__init__()
        self.batch_size = batch_size
        self.cfg = cfg
        self.setup()

    def setup(self, stage=None):
        self.train_dataset = OpenFaceDataset(self.cfg, case="Train")
        self.test_dataset = OpenFaceDataset(self.cfg, case="validation")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=ImbalancedDatasetSampler(self.train_dataset),
            shuffle=False,
            num_workers=2

        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=48,
            shuffle=False,
            num_workers=2
        )


def create_data_module(cfg: DictConfig):
    batch_size = cfg.data.batch_size
    data_module = OpenFaceDataModule(batch_size, cfg)
    return data_module


@hydra.main(config_path="configs", config_name="configures", version_base="1.1")
def process_data(cfg: DictConfig):
    OpenFaceDatasetProcessData(cfg, case="Train")
    OpenFaceDatasetProcessData(cfg, case="validation")


if __name__ == "__main__":
    process_data()
