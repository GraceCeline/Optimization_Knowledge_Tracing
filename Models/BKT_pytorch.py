import os
from math import sqrt
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, accuracy_score
# Mainly inspired by https://github.com/JonathanSilver/pyKT/blob/main/bkt.py

class BKT(nn.Module):

    def __init__(self, n_tasks, forgetting=False):
        super(BKT, self).__init__()
        self.forgetting = forgetting
        self.n_tasks = n_tasks
        self.L0 = nn.Parameter(torch.tensor(np.random.randn()), requires_grad=True)
        self.T = nn.Parameter(torch.tensor(np.random.randn()), requires_grad=True)
        if forgetting:
            self.F = nn.Parameter(torch.tensor(np.random.randn()), requires_grad=True)
        # multigs: one guess/slip value per task instead of one shared pair per skill
        self.G = nn.Parameter(torch.tensor(np.random.randn(n_tasks)), requires_grad=True)
        self.S = nn.Parameter(torch.tensor(np.random.randn(n_tasks)), requires_grad=True)

    def forward(self, x, task):
        """
        :param x: (num_action, batch_size)
        :param task: (num_action, batch_size), long tensor of task ids aligned with x
        :return: (num_pred, batch_size)
        """
        trans = torch.sigmoid(self.T)
        if self.forgetting:
            forget = torch.sigmoid(self.F)
        else:
            forget = torch.tensor(0.)
        guess = torch.sigmoid(self.G[task])
        slip = torch.sigmoid(self.S[task])
        one = torch.ones(x.size(1))
        learn = one * (0.1 * torch.sigmoid(self.L0))
        y = torch.zeros(x.size())
        for t in range(x.size(0)):
            correct = learn * (one - slip[t]) + (one - learn) * guess[t]
            y[t] = correct
            conditional_probability = x[t] * (learn * (one - slip[t]) / correct) \
                                      + (one - x[t]) * (learn * slip[t] / (one - correct))
            learn = conditional_probability * (one - forget) + (one - conditional_probability) * trans
        return y


class TorchBKT:
    """
    Per-skill BKT (multigs: per-task guess/slip) trainer/predictor.

    Expects a dataframe with columns: student_id, task, skill_id, success, step
    """

    def __init__(self, epochs=100, fits=10, forget=False, betas=(.9, .999), seed=1):
        self.epochs = epochs
        self.fits = fits
        self.forget = forget
        self.betas = betas
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.skills = []
        self.task_id_maps = {}   # skill -> {task_value: task_index}
        self.models = {}         # skill -> fitted BKT nn.Module
        self.params = {}         # skill -> dict of prior/learn/forget/guess/slip

    def build_task_id_map(self, df, skill):
        task_values = sorted(df.loc[df['skill_id'] == skill, 'task'].unique())
        return {t: i for i, t in enumerate(task_values)}

    def prepare_data(self, df, skill, task_id_map, student_ids=None):
        ret_data = []
        ret_task = []
        ret_max_length = 0
        skill_df = df[df['skill_id'] == skill]
        for student_id, student_df in skill_df.groupby('student_id'):
            if student_ids is not None and student_id not in student_ids:
                continue
            student_df = student_df.sort_values('step')
            tasks = student_df['task'].tolist()
            if any(t not in task_id_map for t in tasks):
                # skip students with tasks unseen at fit time (e.g. new scenario data)
                continue
            record = student_df['success'].tolist()
            record_task = [task_id_map[t] for t in tasks]
            if len(record):
                ret_data.append(record)
                ret_task.append(record_task)
                ret_max_length = max(ret_max_length, len(record))
        return ret_data, ret_task, ret_max_length

    def convert(self, data, data_max_length):
        batch_size = len(data)
        ret_x = np.zeros((data_max_length, batch_size))
        ret_mask = np.zeros((data_max_length, batch_size))
        for idx in range(batch_size):
            for i in range(len(data[idx])):
                ret_x[i][idx] = data[idx][i]
                ret_mask[i][idx] = 1
        return torch.tensor(ret_x), torch.tensor(ret_mask)

    def convert_task(self, task_data, data_max_length):
        batch_size = len(task_data)
        ret_task = np.zeros((data_max_length, batch_size), dtype=np.int64)
        for idx in range(batch_size):
            for i in range(len(task_data[idx])):
                ret_task[i][idx] = task_data[idx][i]
        return torch.tensor(ret_task, dtype=torch.long)

    def fit_one_skill(self, x, mask, task, n_tasks):
        """randomly initialize self.fits BKT models, optimize MSE with Adam,
        keep the one with the highest training ROC AUC."""
        best_bkt = None
        best_score = 0

        counter = 0
        while counter <= self.fits:
            counter += 1

            bkt = BKT(n_tasks, forgetting=self.forget)
            loss_fn = nn.MSELoss()
            optimizer = optim.Adam(bkt.parameters(), lr=1e-2, betas=self.betas)

            for epoch in range(self.epochs):
                bkt.train()
                with torch.enable_grad():
                    optimizer.zero_grad()
                    y = bkt(x, task)
                    loss = loss_fn(y * mask, x)
                    loss.backward()
                    nn.utils.clip_grad_norm_(bkt.parameters(), max_norm=2.)
                    optimizer.step()

            bkt.eval()
            with torch.no_grad():
                y = bkt(x, task) * mask

            y_true = x.masked_select(mask != 0).numpy()
            y_pred = y.masked_select(mask != 0).numpy()

            try:
                score = roc_auc_score(y_true, y_pred)
                if score > best_score:
                    best_bkt = bkt
                    best_score = score
            except ValueError as e:
                print('during fitting model %d: %s -- refitting...' % (counter, e))
                counter -= 1

        return best_bkt

    def fit(self, df, verbose=True):
        """Fit one BKT model per skill on the full dataframe."""
        self.skills = sorted(df['skill_id'].unique())
        self.task_id_maps = {}
        self.models = {}
        self.params = {}

        for skill in self.skills:
            task_id_map = self.build_task_id_map(df, skill)
            n_tasks = len(task_id_map)

            data, task_data, max_length = self.prepare_data(df, skill, task_id_map)
            if not max_length:
                continue

            x, mask = self.convert(data, max_length)
            task = self.convert_task(task_data, max_length)

            if verbose:
                print("fitting skill '%s' (%d tasks, %d students)" % (skill, n_tasks, x.size(1)))

            best_bkt = self.fit_one_skill(x, mask, task, n_tasks)

            self.task_id_maps[skill] = task_id_map
            self.models[skill] = best_bkt
            self.params[skill] = {
                'prior': (0.1 * torch.sigmoid(best_bkt.L0)).item(),
                'learn': torch.clamp(torch.sigmoid(best_bkt.T), max=1).item(),
                'forget': torch.sigmoid(best_bkt.F).item() if self.forget else 0.,
                'guess': torch.clamp(torch.sigmoid(best_bkt.G), max=1).tolist(),  # one value per task
                'slip': torch.clamp(torch.sigmoid(best_bkt.S), max=1).tolist(),
            }

            if verbose:
                pprint(self.params[skill])

        return self
    
    def predict(self, df):
        
        if not self.models:
            raise RuntimeError('call fit() (or load()) before predict()')

        preds = np.full(len(df), np.nan)

        for skill, model in self.models.items():
            task_id_map = self.task_id_maps[skill]
            skill_mask = (df['skill_id'] == skill).values
            skill_df = df[skill_mask]

            for student_id, student_df in skill_df.groupby('student_id'):
                order = student_df.sort_values('step')
                tasks = order['task'].tolist()
                if any(t not in task_id_map for t in tasks):
                    continue  # unseen task for this skill, cannot score with this model
                record = order['success'].tolist()
                record_task = [task_id_map[t] for t in tasks]

                x, mask = self.convert([record], len(record))
                task = self.convert_task([record_task], len(record))

                model.eval()
                with torch.no_grad():
                    y = model(x, task).squeeze(1).numpy()

                preds[order.index.values] = y

        out = df.copy()
        out['pred'] = preds
        return out

    def score(self, df):
        
        pred_df = self.predict(df).dropna(subset=['pred'])
        y_true = pred_df['success'].to_numpy()
        y_pred = pred_df['pred'].to_numpy()
        auc = roc_auc_score(y_true, y_pred)
        y_pred_binary = (y_pred >= 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred_binary)
        return auc, acc
    

    def save(self, model_dir, scenario_id):
        os.makedirs(model_dir, exist_ok=True)

        state_dicts = {skill: model.state_dict() for skill, model in self.models.items()}

        checkpoint = {
            'scenario_id'  : scenario_id,
            'skills'       : self.skills,
            'task_id_maps' : self.task_id_maps,
            'params'       : self.params,
            'forget'       : self.forget,
            'state_dicts'  : state_dicts,
        }
        path = os.path.join(model_dir, f'TorchBKT_scenario_{scenario_id}.pth')
        torch.save(checkpoint, path)

        print(f"[Scenario {scenario_id}] TorchBKT saved → {path}")
        return path


    def load(self, model_dir, scenario_id):
        path = os.path.join(model_dir, f'TorchBKT_scenario_{scenario_id}.pth')
        checkpoint = torch.load(path, weights_only=False)

        self.skills = checkpoint['skills']
        self.task_id_maps = checkpoint['task_id_maps']
        self.params = checkpoint['params']
        self.forget = checkpoint['forget']

        self.models = {}
        for skill in self.skills:
            n_tasks = len(self.task_id_maps[skill])
            model = BKT(n_tasks, forgetting=self.forget)
            model.load_state_dict(checkpoint['state_dicts'][skill])
            model.eval()
            self.models[skill] = model

        print(f"[Scenario {scenario_id}] TorchBKT loaded ← {path}")
        return self