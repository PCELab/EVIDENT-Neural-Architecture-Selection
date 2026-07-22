import os
import math
import copy
from types import MethodType

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


# =========================
# Bayesian Conv1d (complete)
# =========================
class BBBConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True, priors=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = bias

        self.W_mu  = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        self.W_rho = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        if bias:
            self.bias_mu  = nn.Parameter(torch.Tensor(out_channels))
            self.bias_rho = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        ps = priors or {}
        prior_mu    = float(ps.get('prior_mu', 0.0))
        prior_sigma = float(ps.get('prior_sigma', 0.17))
        self.register_buffer('W_prior_mu',    torch.zeros_like(self.W_mu)  + prior_mu)
        self.register_buffer('W_prior_sigma', torch.full_like(self.W_mu, prior_sigma))
        if bias:
            self.register_buffer('b_prior_mu',    torch.zeros_like(self.bias_mu)  + prior_mu)
            self.register_buffer('b_prior_sigma', torch.full_like(self.bias_mu, prior_sigma))

        self.post_mu_init  = ps.get('posterior_mu_initial',  (0.0, 0.1))   
        self.post_rho_init = ps.get('posterior_rho_initial', (-3.0, 0.1)) 

        self.reset_parameters()

    def reset_parameters(self):
        mu_m, mu_s   = self.post_mu_init
        rho_m, rho_s = self.post_rho_init
        with torch.no_grad():
            self.W_mu.normal_(mu_m, mu_s)
            self.W_rho.normal_(rho_m, rho_s)
            if self.use_bias:
                self.bias_mu.normal_(mu_m, mu_s)
                self.bias_rho.normal_(rho_m, rho_s)

    def _sample_wb(self, sample=True):
        if sample:
            W_sigma = torch.log1p(torch.exp(self.W_rho))
            W = self.W_mu + W_sigma * torch.randn_like(self.W_mu)
            b = None
            if self.use_bias:
                b_sigma = torch.log1p(torch.exp(self.bias_rho))
                b = self.bias_mu + b_sigma * torch.randn_like(self.bias_mu)
        else:
            W = self.W_mu
            b = self.bias_mu if self.use_bias else None
        return W, b

    def forward(self, x, sample=True):
        W, b = self._sample_wb(sample=sample)
        return F.conv1d(
            x, W, b,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )

    @staticmethod
    def _kl_div(mu_q, sigma_q, mu_p, sigma_p):
        return 0.5 * (2*torch.log(sigma_p/sigma_q) - 1
                      + (sigma_q/sigma_p).pow(2)
                      + ((mu_p - mu_q)/sigma_p).pow(2)).sum()

    def kl_loss(self):
        W_sigma = torch.log1p(torch.exp(self.W_rho))
        kl = self._kl_div(self.W_mu, W_sigma, self.W_prior_mu, self.W_prior_sigma)
        if self.use_bias:
            b_sigma = torch.log1p(torch.exp(self.bias_rho))
            kl += self._kl_div(self.bias_mu, b_sigma, self.b_prior_mu, self.b_prior_sigma)
        return kl


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, priors=None):
        super(TemporalBlock, self).__init__()
        self.conv1 = BBBConv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation, priors=priors)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()

        self.conv2 = BBBConv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation, priors=priors)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        if isinstance(self.conv1, BBBConv1d):
            self.conv1.reset_parameters()
        if isinstance(self.conv2, BBBConv1d):
            self.conv2.reset_parameters()
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

    def kl_loss(self):
        kl = self.conv1.kl_loss() + self.conv2.kl_loss()
        return kl


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, pool_kernel=3, priors=None):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.pools = nn.ModuleList()
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 3 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, priors=priors)]
            if i < num_levels-1:
                self.pools.append(nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_kernel))
        self.layers = nn.ModuleList(layers)
        self.num_levels = num_levels

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_levels-1:
                x = self.pools[i](x)
        return x


class TCNEncoder(nn.Module):
    def __init__(self, num_channels, input_channels=3, kernel_size=3, priors=None):
        super(TCNEncoder, self).__init__()
        self.network = TemporalConvNet(input_channels, num_channels, kernel_size, priors=priors)

    def forward(self, x):
        x = x.permute(0, 2, 1)  
        return self.network(x)


class CausalBayesianDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, priors=None):
        super().__init__()
        pad = (kernel_size - 1) 
        self.conv1  = BBBConv1d(in_channels, out_channels, kernel_size,
                                padding=pad, priors=priors)
        self.chomp1 = Chomp1d(pad)
        self.relu1  = nn.ReLU()

        self.conv2  = BBBConv1d(out_channels, out_channels, kernel_size,
                                padding=pad, priors=priors)
        self.chomp2 = Chomp1d(pad)
        self.relu2  = nn.ReLU()

        self.downsample = BBBConv1d(in_channels, out_channels, 1, priors=priors) \
                          if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.chomp1(self.conv1(x)))
        out = self.relu2(self.chomp2(self.conv2(out)))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

    def kl_loss(self):
        kl = self.conv1.kl_loss() + self.conv2.kl_loss()
        if self.downsample is not None:
            kl += self.downsample.kl_loss()
        return kl


class TCNDecoder(nn.Module):
    def __init__(self, num_channels, input_channels, output_size, output_seq_len,
                 kernel_size=3, upsample_kernel=3, priors=None):
        super().__init__()
        self.layers     = nn.ModuleList()
        self.upsamples  = nn.ModuleList()
        self.output_seq_len = output_seq_len
        in_ch = input_channels

        for i, ch in enumerate(num_channels):
            if i > 0:
                self.upsamples.append(nn.Upsample(scale_factor=upsample_kernel,
                                                  mode='linear', align_corners=False))
            self.layers.append(CausalBayesianDecoderBlock(in_ch, ch, kernel_size, priors=priors))
            in_ch = ch

        k_final = 5
        self.final      = BBBConv1d(in_ch, output_size, kernel_size=k_final,
                                    padding=k_final-1, priors=priors)
        self.final_chomp = Chomp1d(k_final-1)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i > 0:
                x = self.upsamples[i-1](x)
            x = layer(x)
        x = self.final_chomp(self.final(x))  
        x = x.transpose(1, 2)                
        return x[:, -self.output_seq_len:, :]  

    def kl_loss(self):
        kl = 0
        for layer in self.layers:
            kl += layer.kl_loss()
        kl += self.final.kl_loss()
        return kl


class TCNModel(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, output_size,
                 input_seq_length, output_seq_length, input_channels=3, kernel_size=3, priors=None):
        super(TCNModel, self).__init__()
        self.encoder = TCNEncoder(encoder_channels, input_channels=input_channels, kernel_size=kernel_size, priors=priors)
        self.decoder = TCNDecoder(decoder_channels, encoder_channels[-1], output_size, output_seq_length,
                                  kernel_size=kernel_size, priors=priors)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def kl_loss(self):
        kl = 0.0
        for block in self.encoder.network.layers:
            kl += block.kl_loss()
        kl += self.decoder.kl_loss()
        return kl


class ELBO(nn.Module):
    def __init__(self, train_size, sigma_noise=10.3):
        super(ELBO, self).__init__()
        self.train_size = train_size
        self.sigma_noise = torch.tensor(sigma_noise, dtype=torch.float32)

    def forward(self, prediction, target, kl):
        mse_term = (F.mse_loss(prediction, target, reduction='mean') * self.train_size) / (2 * self.sigma_noise**2)
        log_term = 0.5 * self.train_size * torch.log(2.0 * torch.tensor(np.pi) * self.sigma_noise)
        loss = log_term + mse_term + kl
        return loss


# =============
# Data loading
# =============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv('/vscratch/grp-danialfa/Azharul/TCN/TCN3/diabetes_data_insulin.csv', delimiter=",")
df['Time'] = pd.to_datetime(df['Time'], utc=True)
df['Time'] = df['Time'].dt.tz_localize(None)
df['date'] = df['Time'].astype(str).str.split(" ").str[0]

data = df[['Glucose', 'MealSize', 'Insulin']].values
data_tensor = torch.tensor(data, dtype=torch.float32)


def create_sequences(data, input_seq_length, output_seq_length):
    xs, ys = [], []
    for i in range(len(data) - input_seq_length - output_seq_length + 1):
        x = data[i:(i + input_seq_length)]
        y = data[(i + input_seq_length):(i + input_seq_length + output_seq_length), 0:1]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)


input_seq_length = 1400
output_seq_length = 60
split_point = int(0.6 * len(data_tensor))

train_data = data_tensor[:split_point]
test_data  = data_tensor[split_point - input_seq_length - output_seq_length + 1:]

x_seq_train, y_seq_train = create_sequences(train_data, input_seq_length, output_seq_length)
x_seq_test,  y_seq_test  = create_sequences(test_data,  input_seq_length, output_seq_length)

X_train, Y_train = x_seq_train, y_seq_train
X_test,  Y_test  = x_seq_test,  y_seq_test

batch_size = 32
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test,  Y_test),  batch_size=16, shuffle=False)

n_train_windows = len(X_train)
n_test_windows  = len(X_test)
starts_train = np.arange(0, n_train_windows, dtype=int)
test_offset  = split_point - input_seq_length - output_seq_length + 1
starts_test  = test_offset + np.arange(0, n_test_windows, dtype=int)


# =================
# Model + training
# =================
priors = {
    'prior_mu': 0,
    'prior_sigma': 0.17,
    'posterior_mu_initial': (0, 0.1),
    'posterior_rho_initial': (-3, 0.1),
}

model = TCNModel([8], [8], 1, input_seq_length, output_seq_length, input_channels=3, kernel_size=5, priors=priors)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=50, verbose=True)


def save_sigmas(model, epoch, filepath):
    with open(filepath, "a") as f:
        f.write(f"Epoch {epoch}:\n")
        for name, module in model.named_modules():
            if isinstance(module, BBBConv1d):
                W_sigma = torch.log1p(torch.exp(module.W_rho)).detach().cpu().numpy()
                f.write(f"{name} W_sigma:\n")
                flat_sigma = W_sigma.flatten()
                np.savetxt(f, flat_sigma[None], fmt="%.6f")
                f.write("\n")


def train_model(model, epoch, loader=None, sigma_noise=10.3, lr=None, scheduler=None, **kwargs):
    if loader is None:
        loader = train_loader

    if lr is not None:
        opt = optim.Adam(model.parameters(), lr=lr)
    else:
        opt = optimizer
    sch = scheduler if scheduler is not None else globals().get('scheduler', None)

    criterion = ELBO(train_size=len(X_train), sigma_noise=sigma_noise).to(device)

    for ep in range(epoch):
        model.train()
        total_loss = 0.0
        total_kl = 0.0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            opt.zero_grad()
            output = model(x_batch)
            kl_loss = model.kl_loss()
            loss = criterion(output, y_batch, kl_loss)
            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            total_kl   += kl_loss.item()

        avg_loss = total_loss / max(1, len(loader))
        if sch is not None:
            if isinstance(sch, ReduceLROnPlateau):
                sch.step(avg_loss)
            else:
                sch.step()

        sigma_save_path = "all_conv_sigmas.txt"
        save_sigmas(model, ep + 1, sigma_save_path)

        if ep % 50 == 0 or ep == epoch - 1:
            print(f'Epoch {ep + 1}, Loss: {avg_loss:.4f}, KL Loss: {total_kl / max(1, len(loader)):.4f}')


def test_model_mean_r2(n_samples=10):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            batch_preds = []
            for _ in range(n_samples):
                pred = model(inputs)
                batch_preds.append(pred.cpu().numpy())
            batch_preds_mean = np.mean(np.stack(batch_preds, axis=0), axis=0)
            all_preds.append(batch_preds_mean)
            all_targets.append(targets.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    r2 = r2_score(all_targets.flatten(), all_preds.flatten())
    print("R² score on test set (averaged over {} predictions): {:.4f}".format(n_samples, r2))
    return r2


def load_test_series(X_test):
    return X_test.to(device)


def get_uncertainty_per_series(model, input_series, T=50):
    input_series = input_series.unsqueeze(0).to(device)
    predictions = []
    with torch.no_grad():
        for _ in range(T):
            output = model(input_series)
            predictions.append(output.cpu().numpy())
    predictions = np.array(predictions).squeeze(1)
    mean_prediction = np.mean(predictions, axis=0)
    std_prediction = np.std(predictions, axis=0)
    aleatoric_uncertainty = np.mean(np.abs(predictions - mean_prediction), axis=0)
    epistemic_uncertainty = std_prediction
    return mean_prediction, epistemic_uncertainty, aleatoric_uncertainty, predictions


def visualize_uncertainty_series(model, input_series, T=50):
    mean_pred, epistemic_uncertainty, aleatoric_uncertainty, individual_predictions = get_uncertainty_per_series(model, input_series, T=T)
    mean_pred_denorm = mean_pred.flatten()
    epistemic_uncertainty = epistemic_uncertainty.flatten()
    aleatoric_uncertainty = aleatoric_uncertainty.flatten()

    print(f"Average Predicted Series (Mean Prediction): {mean_pred_denorm}")
    print("Epistemic Uncertainty (Std. of Predictions):", epistemic_uncertainty)
    print("Aleatoric Uncertainty (Mean Absolute Deviation):", aleatoric_uncertainty)

    plt.figure(figsize=(12, 6))
    plt.plot(mean_pred_denorm, label='Mean Prediction')
    plt.fill_between(range(len(mean_pred_denorm)),
                     mean_pred_denorm - epistemic_uncertainty,
                     mean_pred_denorm + epistemic_uncertainty,
                     color='blue', alpha=0.2, label='Epistemic Uncertainty')
    plt.fill_between(range(len(mean_pred_denorm)),
                     mean_pred_denorm - aleatoric_uncertainty,
                     mean_pred_denorm + aleatoric_uncertainty,
                     color='orange', alpha=0.2, label='Aleatoric Uncertainty')
    series = pd.Series(mean_pred_denorm)
    smoothed = series.rolling(window=20, center=True, min_periods=1).mean()

def run_tcn_uncertainty(model, X_test, T=50):
    model.to(device)
    input_series = load_test_series(X_test[-1])
    visualize_uncertainty_series(model, input_series, T=T)


def sample_and_average_weights(model, n_samples=100):
    param_names = []
    for name, p in model.named_parameters():
        if name.endswith('W_mu') or name.endswith('bias_mu'):
            param_names.append(name)
    mean_params = {name: torch.zeros_like(dict(model.named_parameters())[name].data)
                   for name in param_names}

    for _ in range(n_samples):
        sample_dict = {}
        for name in param_names:
            mu = dict(model.named_parameters())[name].data
            rho_name = name.replace('mu', 'rho')
            rho = dict(model.named_parameters())[rho_name].data
            sigma = torch.log1p(torch.exp(rho))
            sample_dict[name] = mu + sigma * torch.randn_like(mu)
        for name in param_names:
            mean_params[name] += sample_dict[name]

    for name in mean_params:
        mean_params[name] /= n_samples

    return mean_params


def set_model_to_mean_weights(model, mean_params):
    for name, param in model.named_parameters():
        if name in mean_params:
            param.data.copy_(mean_params[name])
    return model


def make_convs_deterministic(model):
    for layer in model.modules():
        if isinstance(layer, BBBConv1d):
            orig_forward = layer.forward
            def forward_no_sample(self, x, orig=orig_forward):
                return orig(x, sample=False)
            layer.forward = MethodType(forward_no_sample, layer)
    return model


mean_params = sample_and_average_weights(model, n_samples=100)
model_mean = copy.deepcopy(model)
model_mean = set_model_to_mean_weights(model_mean, mean_params)
model_mean.to(device)
model_mean.eval()
model_mean = make_convs_deterministic(model_mean)

full_series = data_tensor[:, 0].detach().cpu().numpy()
full_series_denorm = full_series
time_axis = np.arange(len(full_series_denorm))

last_seq_start = len(full_series_denorm) - (input_seq_length + output_seq_length)
input_seq_plot = full_series_denorm[last_seq_start : last_seq_start + input_seq_length]
actual_output_plot = full_series_denorm[last_seq_start + input_seq_length : last_seq_start + input_seq_length + output_seq_length]
input_time = np.arange(last_seq_start, last_seq_start + input_seq_length)
output_time = np.arange(last_seq_start + input_seq_length, last_seq_start + input_seq_length + output_seq_length)

mean_pred, epistemic_uncertainty, aleatoric_uncertainty, all_preds = get_uncertainty_per_series(model, X_test[-1], T=100)
mean_pred = mean_pred.flatten()
epistemic_uncertainty = epistemic_uncertainty.flatten()
aleatoric_uncertainty = aleatoric_uncertainty.flatten()
mean_pred_denorm = mean_pred
epistemic_uncertainty_denorm = epistemic_uncertainty
aleatoric_uncertainty_denorm = aleatoric_uncertainty
ci_lower = np.percentile(all_preds, 2.5, axis=0).flatten()
ci_upper = np.percentile(all_preds, 97.5, axis=0).flatten()

input_seq_plot = input_seq_plot[1000:]
input_time      = input_time[1000:]
actual_output_plot = actual_output_plot
mean_pred_denorm   = mean_pred_denorm
ci_lower           = ci_lower
ci_upper           = ci_upper
output_time        = output_time

with torch.no_grad():
    inp = X_test[-1].unsqueeze(0).to(device)
    mean_pred_weights = model_mean(inp)
    mean_pred_weights = mean_pred_weights.squeeze().cpu().numpy()

model_mu = copy.deepcopy(model)
model_mu = make_convs_deterministic(model_mu)
model_mu.eval().to(device)

inp = X_test[-1].unsqueeze(0).to(device)
with torch.no_grad():
    pred_theta_mu = model_mu(inp)
pred_theta_mu = pred_theta_mu.squeeze().cpu().numpy()
true_future = Y_test[-1].squeeze().cpu().numpy()

param_names = [name for name, _ in model.named_parameters() if name.endswith('W_mu') or name.endswith('bias_mu')]
inp = X_test[-1].unsqueeze(0).to(device)

predictions = []
for i in range(100):
    sample_dict = {}
    for name in param_names:
        mu = dict(model.named_parameters())[name].data
        rho = dict(model.named_parameters())[name.replace('mu', 'rho')].data
        sigma = torch.log1p(torch.exp(rho))
        sample_dict[name] = mu + sigma * torch.randn_like(mu)
    mc_model = copy.deepcopy(model)
    mc_model = set_model_to_mean_weights(mc_model, sample_dict)
    mc_model = make_convs_deterministic(mc_model).to(device).eval()
    with torch.no_grad():
        out = mc_model(inp)
        predictions.append(out.squeeze().cpu().numpy())

predictions = np.stack(predictions, axis=0)
mean_of_10 = predictions.mean(axis=0)
var_of_10 = predictions.var(axis=0)
std_of_10 = np.sqrt(var_of_10)
lower = mean_of_10 - 2*std_of_10
upper = mean_of_10 + 2*std_of_10

def save_posterior_npz(model, path="posterior_params.npz"):
    arrays = {}
    for name, p in model.named_parameters():
        arrays[name] = p.detach().cpu().numpy()
    np.savez(path, **arrays)
    print(f"[posterior] saved all tensors -> {path}")


# ===================================
# HyperParameter Optimization (YOURS)
# ===================================
def extract_mean_and_std(model):
    mus = []
    sigmas = []
    for name, module in model.named_modules():
        if isinstance(module, BBBConv1d):
            W_mu_flat = module.W_mu.data.view(-1)
            W_sigma_flat = torch.log1p(torch.exp(module.W_rho.data)).view(-1)
            mus.append(W_mu_flat)
            sigmas.append(W_sigma_flat)
            if module.use_bias:
                b_mu_flat = module.bias_mu.data.view(-1)
                b_sigma_flat = torch.log1p(torch.exp(module.bias_rho.data)).view(-1)
                mus.append(b_mu_flat)
                sigmas.append(b_sigma_flat)
    theta_mu = torch.cat(mus, dim=0)
    theta_sigma = torch.cat(sigmas, dim=0)
    return theta_mu, theta_sigma


theta_mu, theta_sigma = extract_mean_and_std(model)


def negative_log_evidence(
    model,
    theta_mu,
    theta_sigma,
    X,
    Y,
    sigma_noise_sq,
    alpha_sq,
    device
):
    with torch.no_grad():
        original_state = {}
        idx = 0
        for name, module in model.named_modules():
            if isinstance(module, BBBConv1d):
                param_count = module.W_mu.data.numel()
                original_state[name+"_W_mu"] = module.W_mu.data.clone()
                module.W_mu.data.copy_(theta_mu[idx: idx+param_count].view_as(module.W_mu))
                idx += param_count
                if module.use_bias:
                    param_count_b = module.bias_mu.data.numel()
                    original_state[name+"_b_mu"] = module.bias_mu.data.clone()
                    module.bias_mu.data.copy_(theta_mu[idx: idx+param_count_b].view_as(module.bias_mu))
                    idx += param_count_b
        X = X.to(device)
        Y = Y.to(device)
        pred = model(X)
        idx = 0
        for name, module in model.named_modules():
            if isinstance(module, BBBConv1d):
                param_count = module.W_mu.data.numel()
                module.W_mu.data.copy_(original_state[name+"_W_mu"])
                idx += param_count
                if module.use_bias:
                    param_count_b = module.bias_mu.data.numel()
                    module.bias_mu.data.copy_(original_state[name+"_b_mu"])
                    idx += param_count_b

    pred_flat = pred.reshape(-1)
    Y_flat = Y.reshape(-1)
    N = Y_flat.shape[0]
    sse = torch.sum((pred_flat - Y_flat)**2)
    P = theta_mu.shape[0]
    sum_sq_means = torch.sum(theta_mu**2 + theta_sigma**2)
    sum_log_sigma = torch.sum(torch.log(theta_sigma))
    nle = 0.5 * N * torch.log(2 * torch.tensor(np.pi)) \
          + 0.5 * N * torch.log(sigma_noise_sq) \
          + 0.5 * sse / sigma_noise_sq \
          + 0.5 * P * torch.log(2 * torch.tensor(np.pi)) \
          + 0.5 * P * torch.log(alpha_sq) \
          + 0.5 * sum_sq_means / alpha_sq \
          - (sum_log_sigma)
    return nle


class NoisePriorHyperparams(nn.Module):
    def __init__(self, init_log_noise=0.0, init_log_alpha=0.0):
        super().__init__()
        self.log_noise = nn.Parameter(torch.tensor(init_log_noise))
        self.log_alpha = nn.Parameter(torch.tensor(init_log_alpha))
    def forward(self):
        sigma_noise_sq = torch.exp(self.log_noise)
        alpha_sq = torch.exp(self.log_alpha)
        return sigma_noise_sq, alpha_sq


def optimize_noise_and_prior(
    model,
    theta_mu,
    theta_sigma,
    X,
    Y,
    device,
    init_log_noise=4.66,
    init_log_alpha=-3.54,
    lr=1e-2,
    num_steps=1
):
    hyper = NoisePriorHyperparams(init_log_noise, init_log_alpha).to(device)
    opt = torch.optim.Adam(hyper.parameters(), lr=lr)
    model.eval()
    X = X.to(device)
    Y = Y.to(device)
    for step in range(num_steps):
        opt.zero_grad()
        sigma_noise_sq, alpha_sq = hyper()
        nle = negative_log_evidence(
            model,
            theta_mu,
            theta_sigma,
            X,
            Y,
            sigma_noise_sq,
            alpha_sq,
            device
        )
        nle.backward()
        opt.step()
        if step % 20 == 0 or step == num_steps - 1:
            print(f"Step {step:03d} | -LogEvidence: {nle.item():.4f} "
                  f"| sigma_noise^2={sigma_noise_sq.item():.4f} "
                  f"| alpha^2={alpha_sq.item():.4f}")
    final_sigma_noise_sq, final_alpha_sq = hyper()
    return final_sigma_noise_sq.detach().cpu().item(), final_alpha_sq.detach().cpu().item()


theta_mu, theta_sigma = extract_mean_and_std(model)
X_all = X_train
Y_all = Y_train


def arch_tag(arch):
    enc = "-".join(map(str, arch['encoder_channels']))
    dec = "-".join(map(str, arch['decoder_channels']))
    return f"enc-{enc}__dec-{dec}__k{arch['kernel_size']}"


@torch.no_grad()
def export_evidence_constants(
    model,
    theta_mu,
    theta_sigma,
    X,
    Y,
    device,
    out_path="evidence_constants.npz",
):
    model.eval()
    X = X.to(device); Y = Y.to(device)

    saved = {}
    idx = 0
    for name, module in model.named_modules():
        if isinstance(module, BBBConv1d):
            nW = module.W_mu.numel()
            saved[name+"_W_mu"] = module.W_mu.data.clone()
            module.W_mu.data.copy_(theta_mu[idx:idx+nW].view_as(module.W_mu))
            idx += nW
            if module.use_bias:
                nb = module.bias_mu.numel()
                saved[name+"_b_mu"] = module.bias_mu.data.clone()
                module.bias_mu.data.copy_(theta_mu[idx:idx+nb].view_as(module.bias_mu))
                idx += nb

    pred = model(X)

    for name, module in model.named_modules():
        if isinstance(module, BBBConv1d):
            module.W_mu.data.copy_(saved[name+"_W_mu"])
            if module.use_bias:
                module.bias_mu.data.copy_(saved[name+"_b_mu"])

    pred_flat = pred.reshape(-1).cpu().numpy()
    Y_flat    = Y.reshape(-1).cpu().numpy()
    sse = float(np.sum((pred_flat - Y_flat)**2))
    N   = int(Y_flat.size)
    P   = int(theta_mu.numel())
    sum_sq_means = float(torch.sum(theta_mu**2 + theta_sigma**2).cpu().item())
    sum_log_sigma = float(torch.sum(torch.log(theta_sigma)).cpu().item())
    c_like  = 0.5 * N * np.log(2.0*np.pi)
    c_prior = 0.5 * P * np.log(2.0*np.pi)

    np.savez(out_path,
             N=N, P=P, sse=sse,
             sum_sq_means=sum_sq_means,
             sum_log_sigma=sum_log_sigma,
             c_like=c_like, c_prior=c_prior)


def log_predictive_ll_last_series(model, X_test, Y_test, n_samples=50, sigma_constant=10.3, device=None):
    if device is None:
        device = next(model.parameters()).device
    named_param_dict = dict(model.named_parameters())
    param_names = [name for name in named_param_dict if name.endswith('W_mu') or name.endswith('bias_mu')]
    model.eval()
    X_last = X_test[-1].unsqueeze(0).to(device)  
    Y_last = Y_test[-1].unsqueeze(0).to(device)  
    sigma_obs = torch.tensor(float(sigma_constant), device=device)

    preds_all = []
    logsum = None

    with torch.inference_mode():
        for _ in range(n_samples):
            sample_dict = {}
            for name in param_names:
                mu = named_param_dict[name].data
                rho = named_param_dict[name.replace('mu', 'rho')].data
                std = torch.log1p(torch.exp(rho))
                sample_dict[name] = mu + std * torch.randn_like(mu)

            mc_model = copy.deepcopy(model)
            mc_model = set_model_to_mean_weights(mc_model, sample_dict)
            mc_model = make_convs_deterministic(mc_model).to(device).eval()

            pred = mc_model(X_last)
            preds_all.append(pred.cpu().numpy())

            lp = torch.distributions.Normal(pred, sigma_obs).log_prob(Y_last)
            logsum = lp if logsum is None else torch.logaddexp(logsum, lp)

        LL = logsum - math.log(n_samples)

    return LL.sum().item(), LL.mean().item()


def softplus_np(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def inv_softplus_torch(x):
    eps = 1e-12
    return torch.log(torch.clamp(torch.exp(x) - 1.0, min=eps))

def apply_priors_from_npz(model, npz_path, verbose=True):
    if not os.path.exists(npz_path):
        if verbose: print(f"[prior] NPZ not found: {npz_path} (skipping)")
        return False
    z = np.load(npz_path, allow_pickle=True)
    have_any = False
    for name, module in model.named_modules():
        if not isinstance(module, BBBConv1d): continue
        k_w_mu  = f"{name}.W_mu";  k_w_rho = f"{name}.W_rho"
        k_b_mu  = f"{name}.bias_mu"; k_b_rho = f"{name}.bias_rho"
        if k_w_mu in z and k_w_rho in z:
            W_mu_np  = z[k_w_mu]
            W_rho_np = z[k_w_rho]
            W_sigma_np = softplus_np(W_rho_np)
            if tuple(W_mu_np.shape) == tuple(module.W_prior_mu.shape):
                module.W_prior_mu.copy_(torch.from_numpy(W_mu_np).to(module.W_prior_mu.device))
                module.W_prior_sigma.copy_(torch.from_numpy(W_sigma_np).to(module.W_prior_sigma.device))
                have_any = True
            else:
                if verbose: print(f"[prior] shape mismatch for {k_w_mu}: npz {W_mu_np.shape} vs module {tuple(module.W_prior_mu.shape)}")
        if module.use_bias and (k_b_mu in z) and (k_b_rho in z):
            b_mu_np  = z[k_b_mu]; b_rho_np = z[k_b_rho]; b_sigma_np = softplus_np(b_rho_np)
            if tuple(b_mu_np.shape) == tuple(module.b_prior_mu.shape):
                module.b_prior_mu.copy_(torch.from_numpy(b_mu_np).to(module.b_prior_mu.device))
                module.b_prior_sigma.copy_(torch.from_numpy(b_sigma_np).to(module.b_prior_sigma.device))
            else:
                if verbose: print(f"[prior] shape mismatch for {k_b_mu}: npz {b_mu_np.shape} vs module {tuple(module.b_prior_mu.shape)}")
    if verbose:
        print(f"[prior] Applied priors from: {npz_path}" if have_any else f"[prior] No matching keys in: {npz_path}")
    return have_any

def copy_posterior_to_prior(dst_model, src_model, verbose=True):
    copied = 0
    for (n_dst, m_dst) in dst_model.named_modules():
        if not isinstance(m_dst, BBBConv1d): continue
        m_src = dict(src_model.named_modules()).get(n_dst, None)
        if m_src is None or not isinstance(m_src, BBBConv1d):
            if verbose: print(f"[prior] WARN: no matching src module for {n_dst}")
            continue
        with torch.no_grad():
            m_dst.W_prior_mu.copy_(m_src.W_mu.data)
            m_dst.W_prior_sigma.copy_(torch.log1p(torch.exp(m_src.W_rho.data)))
            if m_dst.use_bias and (m_src.bias_mu is not None):
                m_dst.b_prior_mu.copy_(m_src.bias_mu.data)
                m_dst.b_prior_sigma.copy_(torch.log1p(torch.exp(m_src.bias_rho.data)))
        copied += 1
    if verbose: print(f"[prior] Copied posterior→prior for {copied} conv modules.")
    return copied > 0

def init_posterior_from_prior(model, jitter_rho=0.0):
    with torch.no_grad():
        for m in model.modules():
            if not isinstance(m, BBBConv1d): continue
            m.W_mu.copy_(m.W_prior_mu)
            m.W_rho.copy_(inv_softplus_torch(m.W_prior_sigma.clamp_min(1e-6)))
            if jitter_rho != 0.0:
                m.W_rho.add_(jitter_rho * torch.randn_like(m.W_rho))
            if m.use_bias:
                m.bias_mu.copy_(m.b_prior_mu)
                m.bias_rho.copy_(inv_softplus_torch(m.b_prior_sigma.clamp_min(1e-6)))
                if jitter_rho != 0.0:
                    m.bias_rho.add_(jitter_rho * torch.randn_like(m.bias_rho))


def compute_log_evidence(
    model, X_train, Y_train, theta_mu, theta_sigma,
    sigma_noise_sq_opt, alpha_sq_opt, device,
    hyperprior_type='log-normal', hyperprior_m=0, hyperprior_s=1
):
    tr_H_inv = float(torch.sum(theta_sigma ** 2).cpu().numpy())
    P = int(theta_mu.shape[0])
    N_D = int(X_train.shape[0])

    ln_alpha_sq = np.log(alpha_sq_opt)
    ln_sigma_noise_sq = np.log(sigma_noise_sq_opt)

    numerator_p2 = 2 * (ln_alpha_sq) ** 2
    denominator_p2 = P - (1 / ln_alpha_sq) * tr_H_inv
    if denominator_p2 <= 0:
        raise ValueError("Denominator for σ²_ln(σ_p²) is non-positive.")
    sigma_ln_p2 = np.sqrt(numerator_p2 / denominator_p2)

    numerator_noise = 2 * (ln_sigma_noise_sq) ** 2
    denominator_noise = abs(N_D - P + (1 / ln_sigma_noise_sq) * tr_H_inv)
    if denominator_noise <= 0:
        raise ValueError("Denominator for σ²_ln(σ_noise²) is non-positive.")
    sigma_ln_noise = np.sqrt(numerator_noise / denominator_noise)

    named_param_dict = dict(model.named_parameters())
    param_names = [name for name in named_param_dict if name.endswith('W_mu') or name.endswith('bias_mu')]

    preds_all = []
    for i in range(50):
        sample_dict = {}
        for name in param_names:
            mu = named_param_dict[name].data
            rho_name = name.replace('mu', 'rho')
            rho = named_param_dict[rho_name].data
            sigma = torch.log1p(torch.exp(rho))
            sample_dict[name] = mu + sigma * torch.randn_like(mu)
        mc_model = copy.deepcopy(model)
        mc_model = set_model_to_mean_weights(mc_model, sample_dict)
        mc_model = make_convs_deterministic(mc_model).to(device).eval()
        with torch.no_grad():
            out = mc_model(X_train.to(device))
            preds_all.append(out.cpu().numpy())
    preds_all = np.stack(preds_all, axis=0)
    mean_pred = preds_all.mean(axis=0)
    mean_pred_flat = mean_pred.flatten()
    y_true_flat = Y_train.cpu().numpy().flatten()
    sse = abs(float(np.sum((mean_pred_flat - y_true_flat)**2)))

    sum_log_sigma = np.sum(np.log(theta_sigma.detach().cpu().numpy()))
    log_likelihood =  -0.5 * N_D * np.log(2 * np.pi * sigma_noise_sq_opt) - sse / (2 * np.log(sigma_noise_sq_opt))

    sum_sq_means = float(torch.sum(theta_mu ** 2).cpu().numpy())
    log_prior =  -0.5 * P * np.log(2 * np.pi * alpha_sq_opt) - sum_sq_means / (2 * alpha_sq_opt)

    if hyperprior_type == 'log-uniform':
        log_hyperprior = -np.log(sigma_noise_sq_opt) - np.log(alpha_sq_opt)
    elif hyperprior_type == 'log-normal':
        s = hyperprior_s
        m = hyperprior_m
        log_hyperprior = (
            -np.log(sigma_noise_sq_opt) - np.log(s)
            - 0.5 * np.log(2 * np.pi)
            - ((np.log(sigma_noise_sq_opt) - m) ** 2) / (2 * s ** 2)
            -np.log(alpha_sq_opt) - np.log(s)
            - 0.5 * np.log(2 * np.pi)
            - ((np.log(alpha_sq_opt) - m) ** 2) / (2 * s ** 2)
        )
    else:
        raise ValueError(f"Unsupported hyperprior_type: {hyperprior_type}")

    log_term = np.log(2 * np.pi * sigma_ln_p2) + np.log(2 * np.pi * sigma_ln_noise)

    log_evidence = log_likelihood - log_prior + log_hyperprior + log_term


    return log_evidence


architectures = [
    {'encoder_channels': [2,4,8,16,32], 'decoder_channels': [32,16,8,4,2], 'kernel_size': 7}
]

best_log_evidence = -np.inf
best_arch = None

seed_npz_default = "/vscratch/grp-danialfa/Azharul/TCN/TCN3/posterior_params__enc-2-4-8-16-32__dec-32-16-8-4-2__k7.npz"

for arch in architectures:
    print(f"\nEvaluating architecture: {arch}")

    scalar_priors = priors  

    model = TCNModel(
        encoder_channels=arch['encoder_channels'],
        decoder_channels=arch['decoder_channels'],
        output_size=1,
        input_seq_length=input_seq_length,
        output_seq_length=output_seq_length,
        input_channels=3,
        kernel_size=arch['kernel_size'],
        priors=scalar_priors
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    applied = apply_priors_from_npz(model, seed_npz_default, verbose=True)
    if applied:
        init_posterior_from_prior(model, jitter_rho=0.0)

    train_model(model, epoch=800, loader=train_loader, sigma_noise=10.3, lr=1e-3)

    theta_mu, theta_sigma = extract_mean_and_std(model)

    sigma_noise_sq_opt, alpha_sq_opt = optimize_noise_and_prior(
        model,
        theta_mu,
        theta_sigma,
        X_all,
        Y_all,
        device=device,
        init_log_noise=4.66,
        init_log_alpha=-3.54,
        lr=1e-2,
        num_steps=1000
    )

    priors_retrained = dict(priors)
    priors_retrained['prior_sigma'] = np.sqrt(alpha_sq_opt)

    model_retrain = TCNModel(
        encoder_channels=arch['encoder_channels'],
        decoder_channels=arch['decoder_channels'],
        output_size=1,
        input_seq_length=input_seq_length,
        output_seq_length=output_seq_length,
        input_channels=3,
        kernel_size=arch['kernel_size'],
        priors=priors_retrained
    ).to(device)
    optimizer = optim.Adam(model_retrain.parameters(), lr=0.001)

    def train_model_with_custom_sigma(model_in, sigma_noise):
        model_in.train()
        criterion = ELBO(train_size=len(X_train), sigma_noise=sigma_noise).to(device)
        for ep in range(800):
            total_loss = 0
            total_kl = 0
            for i, (x_batch, y_batch) in enumerate(train_loader):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                output = model_in(x_batch)
                kl_loss = model_in.kl_loss()
                loss = criterion(output, y_batch, kl_loss)
                loss.backward()
                nn_utils.clip_grad_norm_(model_in.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                total_kl += kl_loss.item()
            avg_loss = total_loss / len(train_loader)

            sch = globals().get('scheduler', None)
            if sch is not None and isinstance(sch, ReduceLROnPlateau):
                sch.step(avg_loss)
            print(f'Epoch {ep + 1}, Loss: {avg_loss:.4f}, KL Loss: {total_kl / len(train_loader):.4f}')

    copy_posterior_to_prior(model_retrain, model, verbose=True)
    init_posterior_from_prior(model_retrain, jitter_rho=0.0)

    train_model(model_retrain, epoch=1, loader=train_loader, sigma_noise=(sigma_noise_sq_opt ** 0.5), lr=1e-3)
    train_model_with_custom_sigma(model_retrain, sigma_noise_sq_opt**0.5)

    theta_mu2, theta_sigma2 = extract_mean_and_std(model_retrain)

    os.makedirs("evidence_npz", exist_ok=True)
    npz_path = os.path.join("evidence_npz", f"evidence_constants__{arch_tag(arch)}.npz")
    export_evidence_constants(
        model,
        theta_mu, theta_sigma,
        X_train, Y_train,
        device,
        out_path=npz_path
    )

    save_posterior_npz(model_retrain, path=f"posterior_params__{arch_tag(arch)}.npz")

    @torch.no_grad()
    def mc_predict_window_with_weight_sampling(
        model_source,
        X_test_,
        Y_test_,
        idx,
        n_samples=50,
        device=None
    ):
        if device is None:
            device = next(model_source.parameters()).device

        named_param_dict = dict(model_source.named_parameters())
        param_names = [n for n in named_param_dict if n.endswith('W_mu') or n.endswith('bias_mu')]

        inp_ = X_test_[idx].unsqueeze(0).to(device)
        preds_ = []

        for _ in range(n_samples):
            sample_dict = {}
            for name in param_names:
                mu = named_param_dict[name].data
                rho = named_param_dict[name.replace('mu', 'rho')].data
                sigma = torch.log1p(torch.exp(rho))
                sample_dict[name] = mu + sigma * torch.randn_like(mu)

            mc_model = copy.deepcopy(model_source)
            mc_model = set_model_to_mean_weights(mc_model, sample_dict)
            mc_model = make_convs_deterministic(mc_model).to(device).eval()

            out = mc_model(inp_)
            preds_.append(out.squeeze().cpu().numpy())

        preds_ = np.stack(preds_, axis=0)
        mean_ = preds_.mean(axis=0)
        var_  = preds_.var(axis=0)
        std_  = np.sqrt(var_)
        lower = mean_ - 2.0 * std_
        upper = mean_ + 2.0 * std_

        y_true = Y_test_[idx].squeeze().cpu().numpy()
        return preds_, mean_, std_, var_, lower, upper, y_true

    indices = list(range(0, len(X_test), output_seq_length))


    all_rows  = []
    per_window = {}
    n_samples_csv = 50

    for idx in indices:
        preds, mean_, std_, var_, lower, upper, y_true = mc_predict_window_with_weight_sampling(
            model_retrain, X_test, Y_test, idx, n_samples=n_samples_csv, device=device
        )

        true_start = int(starts_test[idx])
        t0         = true_start + input_seq_length
        t_idx      = np.arange(t0, t0 + output_seq_length)

        assert (len(t_idx) == len(mean_) == len(lower) == len(upper) == len(y_true)), \
            f"Length mismatch for window {idx}"

        per_window[idx] = {
            't': t_idx,
            'actual': y_true,
            'mean': mean_,
            'lower': lower,
            'upper': upper
        }

        row = {
            'window_index': np.repeat(idx, len(t_idx)),
            'time_index':   t_idx,
            'actual':       y_true,
            'mean':         mean_,
            'variance':     var_,
            'std':          std_,
            'lower_2sigma': lower,
            'upper_2sigma': upper
        }
        for k in range(preds.shape[0]):
            row[f'sample_{k+1}'] = preds[k]
        all_rows.append(pd.DataFrame(row))

    df_both = pd.concat(all_rows, axis=0, ignore_index=True)
    fname_ids = "_".join(map(str, indices))
    csv_name = f"mc_predictions_windows_{arch['encoder_channels']}_{arch['kernel_size']}.csv"
    df_both.to_csv(csv_name, index=False)
    print(f"Saved CSV for windows {indices} -> {csv_name}")

    plt.figure(figsize=(14, 7))
    common_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    line_styles = ['-', '--', ':', '-.']

    actual_labeled = False
    band_labeled   = False
    for j, idx in enumerate(indices):
        t      = per_window[idx]['t']
        mean_  = per_window[idx]['mean']
        lower  = per_window[idx]['lower']
        upper  = per_window[idx]['upper']
        actual = per_window[idx]['actual']

        plt.plot(
            t, actual, color='black', linestyle='--', linewidth=1.6, alpha=0.7,
            label='Actual' if not actual_labeled else None
        )
        actual_labeled = True

        ls = line_styles[j % len(line_styles)]
        plt.plot(
            t, mean_, color=common_color, linestyle=ls, linewidth=2.0,
            label=f'Window {idx} mean'
        )

        plt.fill_between(
            t, lower, upper, color=common_color, alpha=0.18,
            label=('±2σ' if not band_labeled else None)
        )
        band_labeled = True

    plt.xlabel('Global time index')
    plt.ylabel('Blood Glucose (mg/dL)')
    plt.title(f'Test predictions for windows {indices}')
    plt.legend(loc='upper left', ncol=2)
    plt.tight_layout()
    plt.savefig(f"combined_windows_{arch['encoder_channels']}_{arch['kernel_size']}.png", dpi=200)
    plt.close()

    y_true_comb = np.concatenate([per_window[idx]['actual'].ravel() for idx in indices])
    y_pred_comb = np.concatenate([per_window[idx]['mean'].ravel()   for idx in indices])

    rmse_comb = np.sqrt(mean_squared_error(y_true_comb, y_pred_comb))
    r2_comb   = r2_score(y_true_comb, y_pred_comb)
    rl2e_comb = np.sqrt(np.sum((y_pred_comb - y_true_comb)**2) / np.sum(y_true_comb**2))

    print(f"[windows {indices}] Combined Mean-of-θ Prediction RMSE: {rmse_comb:.4f}")
    print(f"[windows {indices}] Combined Mean-of-θ Prediction R²:   {r2_comb:.4f}")
    print(f"[windows {indices}] Combined Mean-of-θ Prediction rL2E: {rl2e_comb:.4f}")

    try:
        log_evidence = compute_log_evidence(
            model_retrain, X_train, Y_train, theta_mu2, theta_sigma2,
            sigma_noise_sq_opt, alpha_sq_opt, device, hyperprior_type='log-normal', hyperprior_m=0, hyperprior_s=1
        )
        print(f"Log Evidence: {log_evidence:.2f}")
        if log_evidence > best_log_evidence:
            best_log_evidence = log_evidence
            best_arch = arch
    except ValueError as e:
        print(f"Skipping due to error: {e}")

    sigma_const = (sigma_noise_sq_opt ** 0.5)
    lppl_total, lppl_mean = log_predictive_ll_last_series(
        model_retrain, X_test, Y_test,
        n_samples=50, sigma_constant=sigma_const, device=device
    )
    print(f"LPPL (last series): total={lppl_total:.2f}, mean/elem={lppl_mean:.6f}")

print(f"Best by Log Evidence: {best_arch}  | value = {best_log_evidence:.2f}")
