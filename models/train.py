# ── Block 1 — Imports ─────────────────────────────────────

import numpy as np

import pandas as pd

import matplotlib

matplotlib.use('Agg')  # no display on HPC — must come before pyplot

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch import Tensor

from torch.nn.modules.transformer import TransformerEncoderLayer

from torch.optim.lr_scheduler import ExponentialLR

from torch.utils.data import TensorDataset, DataLoader

import math

import time

import os



print("All imports successful")

print("PyTorch version:", torch.__version__)

print("GPU available:  ", torch.cuda.is_available())

if torch.cuda.is_available():

    print("GPU name:       ", torch.cuda.get_device_name(0))

    print("GPU memory:     ", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), "GB")



# ── Block 2 — Load Data ───────────────────────────────────

df_global = pd.read_csv("data/global.csv")

df_local  = pd.read_csv("data/local.csv")



print("=== GLOBAL DATA ===")

print("Shape:", df_global.shape)

print("Columns:", list(df_global.columns))

print(df_global.head(3).to_string())

print()



print("=== LOCAL DATA ===")

print("Shape:", df_local.shape)

print("Columns:", list(df_local.columns))

print(df_local.head(3).to_string())

print()



print("Same rows:", len(df_global) == len(df_local))

print("Global GHI — min:", df_global['shortwave_radiation_instant (W/m²)'].min(),

      "max:", df_global['shortwave_radiation_instant (W/m²)'].max())

print("Local  GHI — min:", df_local['Solar Radiation (W/m^2)'].min(),

      "max:", df_local['Solar Radiation (W/m^2)'].max())



# ── Block 3 — PositionalEncoding ─────────────────────────

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):

        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(

            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)

        )

        pe = torch.zeros(max_len, 1, d_model)

        pe[:, 0, 0::2] = torch.sin(position * div_term)

        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)



    def forward(self, x: Tensor) -> Tensor:

        x = x + self.pe[:x.size(1)].transpose(0, 1)

        return self.dropout(x)



print("PositionalEncoding defined")



# ── Block 4 — ForecastingModel ────────────────────────────

class ForecastingModel(torch.nn.Module):

    def __init__(self, seq_len, embed_size=8, nhead=2,

                 dim_feedforward=1024, dropout=0.1,

                 conv1d_emb=True, conv1d_kernel_size=3,

                 device="cuda"):

        super(ForecastingModel, self).__init__()



        self.device             = device

        self.conv1d_emb         = conv1d_emb

        self.conv1d_kernel_size = conv1d_kernel_size

        self.seq_len            = seq_len

        self.embed_size         = embed_size



        if conv1d_emb:

            if conv1d_kernel_size % 2 == 0:

                raise Exception("conv1d_kernel_size must be odd.")

            self.conv1d_padding  = conv1d_kernel_size - 1

            self.input_embedding = nn.Conv1d(1, embed_size,

                                              kernel_size=conv1d_kernel_size)

        else:

            self.input_embedding = nn.Linear(1, embed_size)



        self.position_encoder = PositionalEncoding(

            d_model=embed_size, dropout=dropout, max_len=seq_len

        )

        self.transformer_encoder = TransformerEncoderLayer(

            d_model=embed_size, nhead=nhead,

            dim_feedforward=dim_feedforward,

            dropout=dropout, batch_first=True

        )

        self.linear1  = nn.Linear(seq_len * embed_size, int(dim_feedforward))

        self.linear2  = nn.Linear(int(dim_feedforward),     int(dim_feedforward / 2))

        self.linear3  = nn.Linear(int(dim_feedforward / 2), int(dim_feedforward / 4))

        self.linear4  = nn.Linear(int(dim_feedforward / 4), int(dim_feedforward / 16))

        self.linear5  = nn.Linear(int(dim_feedforward / 16),int(dim_feedforward / 64))

        self.outlayer = nn.Linear(int(dim_feedforward / 64), 1)



        self.relu    = nn.ReLU()

        self.dropout = nn.Dropout(dropout)



    def forward(self, x):

        src_mask = self._generate_square_subsequent_mask()

        src_mask.to(self.device)



        if self.conv1d_emb:

            x = F.pad(x, (0, 0, self.conv1d_padding, 0), "constant", -1)

            x = self.input_embedding(x.transpose(1, 2))

            x = x.transpose(1, 2)

        else:

            x = self.input_embedding(x)



        x = self.position_encoder(x)

        x = self.transformer_encoder(x, src_mask=src_mask).reshape(

            (-1, self.seq_len * self.embed_size)

        )

        x = self.relu(self.dropout(self.linear1(x)))

        x = self.relu(self.dropout(self.linear2(x)))

        x = self.relu(self.dropout(self.linear3(x)))

        x = self.relu(self.dropout(self.linear4(x)))

        x = self.relu(self.linear5(x))

        return self.outlayer(x)



    def _generate_square_subsequent_mask(self):

        return torch.triu(

            torch.full((self.seq_len, self.seq_len), float('-inf'),

                       dtype=torch.float32, device=self.device),

            diagonal=1,

        )



print("ForecastingModel defined")



# Quick forward pass test

device     = "cuda" if torch.cuda.is_available() else "cpu"

test_model = ForecastingModel(seq_len=48, device=device).to(device)

test_x     = torch.randn(4, 48, 1).to(device)

test_y     = test_model(test_x)

print("Forward pass: input", test_x.shape, "→ output", test_y.shape)

print("Output correct (should be [4,1]):", test_y.shape == torch.Size([4, 1]))

del test_model

torch.cuda.empty_cache()



# ── Block 5 — Prepare Data ────────────────────────────────

Hours_24 = 48



data_x = list(df_global['shortwave_radiation_instant (W/m²)'])

data_y = list(df_local['Solar Radiation (W/m^2)'])



x_raw       = np.array(data_x[:-Hours_24])

y_raw       = np.array(data_y[:-Hours_24])

forcast_raw = np.array(data_y[-Hours_24:])



x_max = x_raw.max()

y_max = y_raw.max()



x       = x_raw / x_max

y       = y_raw / y_max

forcast = forcast_raw / y_max



print("=== DATA SPLIT AND NORMALIZATION ===")

print("Training input  (x):  ", len(x), "points =", int(len(x)/48), "days")

print("Training target (y):  ", len(y), "points =", int(len(y)/48), "days")

print("Test forecast:        ", len(forcast), "points = last day")

print("x_max:", round(x_max, 2), "W/m²")

print("y_max:", round(y_max, 2), "W/m²")



# ── Block 6 — Grid Search ─────────────────────────────────

seq_len_list = [24, 48, 98]

batch_list   = [16, 64]

epochs_list  = [100, 200]
lr_list      = [0.001, 0.0001]





total  = len(seq_len_list)*len(batch_list)*len(epochs_list)*len(lr_list)



print("Device:         " + device)

print("Total models:   " + str(total))



results_full      = []

run_number        = 1

best_rmse_so_far  = float('inf')

best_model_params = {}



grid_start = time.time()



for sl in seq_len_list:

    for bs in batch_list:

        for ep in epochs_list:

            for lr in lr_list:



                X = np.array([x[ii:ii+sl]

                              for ii in range(0, x.shape[0]-sl)]).reshape((-1, sl, 1))

                Y = np.array([y[ii+sl]

                              for ii in range(0, y.shape[0]-sl)]).reshape((-1, 1))



                start = time.time()

                model = ForecastingModel(

                    seq_len=sl, embed_size=8, nhead=2,

                    dim_feedforward=1024, dropout=0.1,

                    conv1d_emb=True, conv1d_kernel_size=3,

                    device=device

                ).to(device)



                model.train()

                criterion  = torch.nn.HuberLoss()

                optimizer  = torch.optim.AdamW(model.parameters(), lr=lr)

                scheduler  = ExponentialLR(optimizer, gamma=0.98)

                dataset    = TensorDataset(

                    torch.Tensor(X).to(device),

                    torch.Tensor(Y).to(device)

                )

                dataloader = DataLoader(dataset, batch_size=bs)



                loss_list = []

                for epoch in range(ep):

                    for xx, yy in dataloader:

                        optimizer.zero_grad()

                        out  = model(xx)

                        loss = criterion(out, yy)

                        loss.backward()

                        optimizer.step()

                    scheduler.step()

                    loss_list.append(loss.item())



                elapsed = time.time() - start



                model.eval()

                x_copy      = np.copy(x)

                predictions = []

                with torch.no_grad():

                    for ff in range(len(forcast)):

                        xxx = x_copy[-sl:]

                        yyy = model(

                            torch.Tensor(xxx).reshape((1, sl, 1)).to(device)

                        )

                        x_copy = np.concatenate(

                            (x_copy, np.array([yyy.cpu().item()]))

                        )

                        predictions.append(yyy.cpu().item())



                predictions = np.array(predictions)

                rmse_norm   = float(np.sqrt(np.mean((predictions - forcast)**2)))

                rmse_wm2    = rmse_norm * y_max



                if rmse_norm < best_rmse_so_far:

                    best_rmse_so_far  = rmse_norm

                    best_model_params = {

                        'seq_len'        : sl,

                        'batch'          : bs,

                        'epochs'         : ep,

                        'lr'             : lr,

                        'predictions'    : predictions.copy(),

                        'predictions_wm2': (predictions * y_max).copy(),

                        'loss_curve'     : loss_list.copy()

                    }



                results_full.append({

                    'run'       : run_number,

                    'seq_len'   : sl,

                    'batch'     : bs,

                    'epochs'    : ep,

                    'lr'        : lr,

                    'RMSE_norm' : round(rmse_norm, 6),

                    'RMSE_wm2'  : round(rmse_wm2,  2),

                    'time_sec'  : round(elapsed,    1),

                    'loss_final': round(loss_list[-1], 6)

                })



                marker = " <- BEST" if rmse_norm == best_rmse_so_far else ""

                elapsed_total = round((time.time() - grid_start)/60, 1)

                print(

                    "Run " + str(run_number).rjust(3) + "/" + str(total) +

                    " | seq=" + str(sl).rjust(3) +

                    " batch=" + str(bs).rjust(3) +

                    " ep=" + str(ep).rjust(3) +

                    " lr=" + str(lr) +

                    " | RMSE=" + str(round(rmse_wm2, 1)).rjust(7) + " W/m2" +

                    " | loss=" + str(round(loss_list[-1], 6)) +

                    " | " + str(round(elapsed, 1)) + "s" +

                    " | total=" + str(elapsed_total) + "min" +

                    marker

                )



                if run_number % 50 == 0:

                    ckpt = pd.DataFrame(results_full).sort_values('RMSE_wm2')

                    ckpt.to_csv('results/grid_search_checkpoint.csv', index=False)

                    print("  [Checkpoint saved — run " + str(run_number) +

                          " | best so far: " + str(round(best_rmse_so_far * y_max, 2)) +

                          " W/m2]")



                run_number += 1

                torch.cuda.empty_cache()



total_time = round((time.time() - grid_start) / 60, 1)

print()

print("=" * 70)

print("GRID SEARCH COMPLETE — total time: " + str(total_time) + " minutes")

print("=" * 70)



results_full_df = pd.DataFrame(results_full).sort_values('RMSE_wm2').reset_index(drop=True)

results_full_df.to_csv('results/grid_search_results_full.csv', index=False)

print("Results saved to results/grid_search_results_full.csv")



print("Top 10 models:")

print(results_full_df[['run','seq_len','batch','epochs','lr',

                        'RMSE_wm2','loss_final','time_sec']].head(10).to_string(index=False))



b = results_full_df.iloc[0]

print("Best model:")

print("  seq_len  : " + str(int(b['seq_len'])))

print("  batch    : " + str(int(b['batch'])))

print("  epochs   : " + str(int(b['epochs'])))

print("  lr       : " + str(b['lr']))

print("  RMSE     : " + str(b['RMSE_wm2']) + " W/m2")



# ── Block 7 — Plots ───────────────────────────────────────

predictions_wm2 = best_model_params['predictions_wm2']

forcast_wm2     = forcast * y_max

rmse_final      = float(np.sqrt(np.mean((predictions_wm2 - forcast_wm2)**2)))

time_axis       = range(len(forcast_wm2))



# Plot 1: Observed vs Predicted

plt.figure(figsize=(14, 5))

plt.plot(time_axis, forcast_wm2, color='blue', label='Observed', linewidth=1.5)

plt.plot(time_axis, predictions_wm2, color='red',

         label='Predicted  RMSE=' + str(round(rmse_final, 2)) + ' W/m2',

         linewidth=2, linestyle='--')

plt.title('Observed vs Predicted Solar Irradiance')

plt.xlabel('Time steps (30-min)')

plt.ylabel('Solar Irradiance (W/m2)')

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig('results/prediction_plot.png', dpi=150, bbox_inches='tight')

plt.close()

print("Saved: results/prediction_plot.png")



# Plot 2: Loss curve

plt.figure(figsize=(14, 4))

plt.plot(best_model_params['loss_curve'], color='orange', linewidth=1)

plt.title('Training Loss Curve — Best Model')

plt.xlabel('Epoch')

plt.ylabel('Huber Loss (normalized)')

plt.grid(True)

plt.tight_layout()

plt.savefig('results/loss_curve.png', dpi=150, bbox_inches='tight')

plt.close()

print("Saved: results/loss_curve.png")



# Plot 3: Error bars

error = predictions_wm2 - forcast_wm2

plt.figure(figsize=(14, 4))

plt.bar(time_axis, error,

        color=['red' if e > 0 else 'blue' for e in error], alpha=0.7)

plt.axhline(y=0, color='black', linewidth=0.8)

plt.title('Prediction Error per Time Step')

plt.xlabel('Time steps (30-min)')

plt.ylabel('Error (W/m2)')

plt.grid(True, axis='y')

plt.tight_layout()

plt.savefig('results/error_plot.png', dpi=150, bbox_inches='tight')

plt.close()

print("Saved: results/error_plot.png")



# Plot 4: Full overview

fig, axs = plt.subplots(2, 1, figsize=(16, 8))

axs[0].plot(df_global['shortwave_radiation_instant (W/m²)'],

            color='blue', linewidth=0.5, alpha=0.7, label='Global NSRDB (input)')

axs[0].plot(df_local['Solar Radiation (W/m^2)'],

            color='green', linewidth=0.5, alpha=0.7, label='Local station (target)')

axs[0].axvline(x=1440, color='red', linestyle='--', linewidth=1.5,

               label='Train/Test split')

axs[0].set_title('Training Data')

axs[0].set_ylabel('Solar Irradiance (W/m²)')

axs[0].legend(fontsize=9)

axs[0].grid(True, alpha=0.3)



axs[1].plot(time_axis, forcast_wm2, color='blue', label='Observed', linewidth=1.5)

axs[1].plot(time_axis, predictions_wm2, color='red',

            label='Predicted  RMSE=' + str(round(rmse_final, 2)) + ' W/m²',

            linewidth=2, linestyle='--')

axs[1].set_title('Test Day Prediction')

axs[1].set_xlabel('Time steps (30-min)')

axs[1].set_ylabel('Solar Irradiance (W/m²)')

axs[1].legend(fontsize=9)

axs[1].grid(True, alpha=0.3)



plt.tight_layout()

plt.savefig('results/full_overview.png', dpi=150, bbox_inches='tight')

plt.close()

print("Saved: results/full_overview.png")



error = predictions_wm2 - forcast_wm2

print()

print("Error analysis:")

print("  Max overpredict:  " + str(round(error.max(), 1)) + " W/m2")

print("  Max underpredict: " + str(round(error.min(), 1)) + " W/m2")

print("  Mean error:       " + str(round(error.mean(), 1)) + " W/m2")

print("ALL DONE.")


