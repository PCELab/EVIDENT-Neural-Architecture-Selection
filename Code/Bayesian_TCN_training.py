import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import r2_score
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.utils as nn_utils

class BBBConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, priors=None):
        super(BBBConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = bias

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.17,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        self.W_rho = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))

        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
            self.bias_rho = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)
        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, x, sample=True):
        W_sigma = torch.log1p(torch.exp(self.W_rho))
        if self.use_bias:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        else:
            bias_sigma = None

        act_mu = F.conv1d(x, self.W_mu, self.bias_mu, self.stride, self.padding, self.dilation)
        act_var = 1e-8 + F.conv1d(x ** 2, W_sigma ** 2, bias_sigma**2, self.stride, self.padding, self.dilation)
        act_std = torch.sqrt(act_var)

        if self.training or sample:
            eps = torch.randn_like(act_mu, device=act_mu.device)
            return act_mu + act_std * eps
        else:
            return act_mu

    def kl_loss(self):
        kl = self._kl_div(self.W_mu, torch.log1p(torch.exp(self.W_rho)), self.prior_mu, self.prior_sigma)
        if self.use_bias:
            kl += self._kl_div(self.bias_mu, torch.log1p(torch.exp(self.bias_rho)), self.prior_mu, self.prior_sigma)
        return kl

    @staticmethod
    def _kl_div(mu_q, sigma_q, mu_p, sigma_p):
        return 0.5 * (2 * torch.log(sigma_p / sigma_q) - 1 +
                      (sigma_q / sigma_p).pow(2) +
                      ((mu_p - mu_q) / sigma_p).pow(2)).sum()

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

class BayesianDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, priors=None):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = BBBConv1d(in_channels, out_channels, kernel_size, padding=padding, priors=priors)
        self.relu1 = nn.ReLU()
        self.conv2 = BBBConv1d(out_channels, out_channels, kernel_size, padding=padding, priors=priors)
        self.relu2 = nn.ReLU()
        self.downsample = BBBConv1d(in_channels, out_channels, 1, priors=priors) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
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
        self.layers = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.output_seq_len = output_seq_len
        self.output_size = output_size
        in_channels = input_channels
        for i, channels in enumerate(num_channels):
            if i > 0:
                self.upsamples.append(nn.Upsample(scale_factor=upsample_kernel, mode='linear', align_corners=True))
            self.layers.append(BayesianDecoderBlock(in_channels, channels, kernel_size=kernel_size, priors=priors))
            in_channels = channels
        self.final_layer = BBBConv1d(in_channels, output_size, kernel_size=5, priors=priors)
        self.num_levels = len(num_channels)
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i > 0:
                x = self.upsamples[i-1](x)
            x = layer(x)
        x = self.final_layer(x)
        x = x.transpose(1, 2) 
        return x[:, :self.output_seq_len, :]

    def kl_loss(self):
        kl = 0
        for layer in self.layers:
           kl += layer.kl_loss()  
        kl += self.final_layer.kl_loss()
        return kl

class TCNModel(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, output_size, input_seq_length, output_seq_length, input_channels=3, kernel_size=3, priors=None):
        super(TCNModel, self).__init__()
        self.encoder = TCNEncoder(encoder_channels, input_channels=input_channels, kernel_size=kernel_size, priors=priors)
        self.decoder = TCNDecoder(decoder_channels, encoder_channels[-1], output_size, output_seq_length, kernel_size=kernel_size, priors=priors)

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import numpy as np

import pandas as pd

df = pd.read_csv('/u/mislam21/azharul/TCN3/diabetes_data_insulin.csv', delimiter=",")
df['Time'] = pd.to_datetime(df['Time'], utc=True)
df['Time'] = df['Time'].dt.tz_localize(None)
df['date'] = df['Time'].astype(str).str.split(" ").str[0]

data = df[['Glucose', 'MealSize', 'Insulin']].values
data_tensor = torch.tensor(data, dtype=torch.float32)

MEAL_THRESH = 0.0 
meal_mask = (df['MealSize'].fillna(0).to_numpy() > MEAL_THRESH)

starts = np.where(meal_mask & np.concatenate(([True], ~meal_mask[:-1])))[0]
ends   = np.where(meal_mask & np.concatenate((~meal_mask[1:], [True])))[0]
assert len(starts) == len(ends), "Starts/ends mismatch for meal episodes."

if len(starts) >= 2:
    gaps = starts[1:] - ends[:-1] - 1
    valid_gaps = gaps[gaps > 0]
    if len(valid_gaps) == 0:
        output_seq_length = 1
    else:
        output_seq_length = int(valid_gaps.min())
else:
    output_seq_length = 60 

input_seq_length = 1440
print(f"Chosen lengths -> input_seq_length={input_seq_length}, output_seq_length={output_seq_length}")


def create_meal_safe_sequences_by_pid(
    df,
    pid_col,
    input_len,
    output_len,
    stride=1,
    meal_thresh=0.0,
    channels=('Glucose','MealSize','Insulin')
):

    xs, ys, starts_global, pids_list = [], [], [], []
    offset = 0 

    for pid, sub in df.groupby(pid_col, sort=False):
        sub_len = len(sub)
        if sub_len < input_len + output_len:
            offset += sub_len
            continue  

        meal_mask = (sub['MealSize'].fillna(0).to_numpy() > meal_thresh)

        last_start = sub_len - input_len - output_len
        arr = sub.loc[:, list(channels)].to_numpy(dtype=np.float32)
        glu = sub['Glucose'].to_numpy(dtype=np.float32)

        for i in range(0, last_start + 1, stride):
            y_lo = i + input_len
            y_hi = y_lo + output_len

            if meal_mask[y_lo:y_hi].any():
                continue

            xs.append(arr[i:i+input_len])             
            ys.append(glu[y_lo:y_hi, None])        
            starts_global.append(offset + i)      
            pids_list.append(pid)

        offset += sub_len 

    if not xs:
        raise ValueError("No meal-safe windows found; adjust output_len/stride or check MealSize.")

    X = torch.from_numpy(np.asarray(xs, dtype=np.float32))
    Y = torch.from_numpy(np.asarray(ys, dtype=np.float32))
    starts = np.asarray(starts_global, dtype=np.int64)
    pids   = np.asarray(pids_list)

    return X, Y, starts, pids


X, Y, starts, pids = create_meal_safe_sequences_by_pid(
    df,
    pid_col='PID',
    input_len=input_seq_length,
    output_len=output_seq_length,
    stride=1,
    meal_thresh=MEAL_THRESH,
    channels=('Glucose','MealSize','Insulin')
)

split = int(0.8 * len(X))
X_train, Y_train, starts_train = X[:split], Y[:split], starts[:split]
X_test,  Y_test,  starts_test  = X[split:], Y[split:], starts[split:]
pids_train, pids_test = pids[:split], pids[split:]

batch_size = 32
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=False)
train_data = torch.tensor(df.loc[sorted({i for s in (starts_train.tolist() if hasattr(starts_train, "tolist") else list(starts_train)) for i in range(int(s), int(s)+input_seq_length+output_seq_length)}), ['Glucose', 'MealSize', 'Insulin']].values, dtype=torch.float32)

priors = {
    'prior_mu': 0,
    'prior_sigma': 0.17,
    'posterior_mu_initial': (0, 0.1),
    'posterior_rho_initial': (-3, 0.1),
}
model = TCNModel([8], [8], 1, input_seq_length, output_seq_length, input_channels=3, kernel_size=5, priors=priors)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer,
                              mode='min',
                              factor=0.8,
                              patience=50,
                              verbose=True)

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
def train_model(model, epoch):
    model.train()
    criterion = ELBO(train_size=len(X_train), sigma_noise=10.3).to(device)
    for epoch in range(epoch): 
        total_loss = 0
        total_kl = 0
        for i, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            kl_loss = model.kl_loss()
            loss = criterion(output, y_batch, kl_loss)
            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_kl += kl_loss.item()
            
        avg_loss = total_loss / len(train_loader)
     
        scheduler.step(avg_loss)
        sigma_save_path = "all_conv_sigmas.txt"
        save_sigmas(model, epoch + 1, sigma_save_path)
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, KL Loss: {total_kl / len(train_loader):.4f}')

def test_model_mean_r2(n_samples=10):
    model.eval()
    all_preds = []
    all_targets = []
    
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=16, shuffle=False)
    
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

import copy
from types import MethodType

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
actual_output_plot = full_series_denorm[
last_seq_start + input_seq_length :
last_seq_start + input_seq_length + output_seq_length
    ]

input_time = np.arange(last_seq_start, last_seq_start + input_seq_length)
output_time = np.arange(
last_seq_start + input_seq_length,
last_seq_start + input_seq_length + output_seq_length
    )

mean_pred, epistemic_uncertainty, aleatoric_uncertainty, all_preds = get_uncertainty_per_series(
        model, X_test[-1], T=100
    )

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

import copy
import matplotlib.pyplot as plt

model_mu = copy.deepcopy(model)
model_mu = make_convs_deterministic(model_mu)
model_mu.eval().to(device)

inp = X_test[-1].unsqueeze(0).to(device)  

with torch.no_grad():
    pred_theta_mu = model_mu(inp)    
pred_theta_mu = pred_theta_mu.squeeze().cpu().numpy()  

true_future = Y_test[-1].squeeze().cpu().numpy()       

param_names = [name for name, _ in model.named_parameters() 
               if name.endswith('W_mu') or name.endswith('bias_mu')]

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


plt.figure(figsize=(14,7))

for idx, pred in enumerate(predictions):
    plt.plot(output_time,
             pred,
             alpha=0.7,
             label=f'Sample {idx+1}')

plt.plot(output_time,
         mean_of_10,
         'r-',
         linewidth=2,
         label='Mean of 10 θ-samples')

plt.plot(output_time,
         actual_output_plot,
         'k--',
         linewidth=2,
         label='Actual')


plt.figure(figsize=(14,7))

plt.plot(output_time,
         actual_output_plot,
         'k--',
         linewidth=2,
         label='Actual')

plt.plot(output_time,
         mean_of_10,
         'r-',
         linewidth=2,
         label='Mean of 10 θ-samples')

plt.fill_between(output_time,
                 lower,
                 upper,
                 color='red',
                 alpha=0.3,
                 label='Uncertainty (±1 σ)')


def save_posterior_npz(model, path="posterior_params.npz"):
    arrays = {}
    for name, p in model.named_parameters():
        arrays[name] = p.detach().cpu().numpy()
    np.savez(path, **arrays)
    print(f"[posterior] saved all tensors -> {path}")



#################################
### HyperParameter Optimization ###
##################################

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


################################################################################

import numpy as np
import torch
import os

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

import math

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
 
import copy
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

@torch.no_grad()
def predict_train_uncertainty_and_plot(
    model,
    X_train,
    starts_train,            
    full_glucose_np,         
    input_seq_length,
    output_seq_length,
    T=50,                     
    batch_size=16,
    device=None,
    save_path="train_uncertainty.png",
    csv_path="train_uncertainty.csv",
    ci_sigma=2.0          
):
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    starts_train = np.asarray(starts_train, dtype=np.int64)
    out_len = int(output_seq_length)
    train_end = int(starts_train.max()) + int(input_seq_length) + out_len
    train_end = min(train_end, len(full_glucose_np))  

    train_glucose_np = full_glucose_np[:train_end].astype(float)
    train_len = int(train_end)

    agg_sum    = np.zeros(train_len, dtype=np.float64)
    agg_sum_sq = np.zeros(train_len, dtype=np.float64)
    agg_count  = np.zeros(train_len, dtype=np.int64)

    def global_range_for_sample(g):
        t0 = int(starts_train[g]) + int(input_seq_length)
        t1 = t0 + out_len
        return t0, t1

    param_dict  = dict(model.named_parameters())
    param_names = [n for n in param_dict if n.endswith('W_mu') or n.endswith('bias_mu')]

    ds = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    base = 0
    for t in range(T):
        sample_dict = {}
        for name in param_names:
            mu  = param_dict[name].data
            rho = param_dict[name.replace('mu','rho')].data
            std = torch.log1p(torch.exp(rho))
            sample_dict[name] = mu + std * torch.randn_like(mu)

        mc_model = copy.deepcopy(model)
        for name, p in mc_model.named_parameters():
            if name in sample_dict:
                p.data.copy_(sample_dict[name])
        mc_model = make_convs_deterministic(mc_model).to(device).eval()

        base = 0
        for (xb,) in ds:
            B = xb.size(0)
            xb = xb.to(device)
            pred = mc_model(xb).squeeze(-1).cpu().numpy()  

            for j in range(B):
                g = base + j
                s, e = global_range_for_sample(g)
                if s >= train_len:
                    continue
                e_clamped = min(e, train_len)
                L = e_clamped - s
                if L <= 0:
                    continue

                y = pred[j, :L]
                agg_sum[s:e_clamped]    += y
                agg_sum_sq[s:e_clamped] += y * y
                agg_count[s:e_clamped]  += 1

            base += B

    with np.errstate(divide='ignore', invalid='ignore'):
        mean_series = np.full(train_len, np.nan, dtype=np.float64)
        second_mom  = np.full(train_len, np.nan, dtype=np.float64)
        mask = agg_count > 0
        mean_series[mask] = agg_sum[mask] / agg_count[mask]
        second_mom[mask]  = agg_sum_sq[mask] / agg_count[mask]
        var_series = np.zeros(train_len, dtype=np.float64)
        var_series[mask] = np.maximum(second_mom[mask] - mean_series[mask]**2, 0.0)
        std_series = np.sqrt(var_series)

    lower = mean_series - ci_sigma * std_series
    upper = mean_series + ci_sigma * std_series

    try:
        df_plot = pd.DataFrame({
            "t":      np.arange(train_len, dtype=int),
            "actual": train_glucose_np,
            "mean":   mean_series,
            "lower":  lower,
            "upper":  upper,
            "count":  agg_count
        })
        df_plot.to_csv(csv_path, index=False)
        print(f"[train-uncertainty] Saved: {csv_path}")
    except Exception as e:
        print(f"[train-uncertainty] CSV save skipped ({e})")

    # plot
    plt.figure(figsize=(16,6))
    t = np.arange(train_len)
    plt.plot(t, train_glucose_np, lw=1.2, label="Train actual", alpha=0.7, color='k')
    plt.plot(t[mask], mean_series[mask], lw=1.6, label=f"MC mean (T={T})", color='r')
    plt.fill_between(t[mask], lower[mask], upper[mask], alpha=0.25, label=f"±{ci_sigma}σ band")
    plt.xlabel("Time index (train)")
    plt.ylabel("Blood Glucose (mg/dL)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=150)
        print(f"[train-uncertainty] Figure saved: {save_path}")
    except Exception as e:
        print(f"[train-uncertainty] Figure save skipped ({e})")
    plt.close()

    return mean_series, lower, upper, agg_count

 
                          
#################################
### Architecture Optimization ###
#################################

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
    {'encoder_channels': [2,4,8,16,32], 'decoder_channels': [32,16,8,4,2], 'kernel_size': 7},
    {'encoder_channels': [2,4,8,16,32], 'decoder_channels': [32,16,8,4,2], 'kernel_size': 9},
    {'encoder_channels': [2,4,8,16,32,64], 'decoder_channels': [64,32,16,8,4,2], 'kernel_size': 3},
    {'encoder_channels': [4,8,16,32,64], 'decoder_channels': [64,32,16,8,4], 'kernel_size': 7},
    {'encoder_channels': [4,8,16,32,64], 'decoder_channels': [64,32,16,8,4], 'kernel_size': 9},
    {'encoder_channels': [4,8,16,32,64,128], 'decoder_channels': [128,64,32,16,8,4], 'kernel_size': 3}
]

best_log_evidence = -np.inf
best_arch = None

for arch in architectures:
    print(f"\nEvaluating architecture: {arch}")
    
    model = TCNModel(
        encoder_channels=arch['encoder_channels'],
        decoder_channels=arch['decoder_channels'],
        output_size=1,
        input_seq_length=input_seq_length,
        output_seq_length=output_seq_length,
        input_channels=3,
        kernel_size=arch['kernel_size'],
        priors=priors
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model,epoch=800) 

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
    num_steps=1000)

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

    def train_model_with_custom_sigma(model, sigma_noise):
     model.train()
     criterion = ELBO(train_size=len(X_train), sigma_noise=sigma_noise).to(device)
     for epoch in range(600):
        total_loss = 0
        total_kl = 0
        for i, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            kl_loss = model.kl_loss()
            loss = criterion(output, y_batch, kl_loss)
            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_kl += kl_loss.item()
            
        avg_loss = total_loss / len(train_loader)
     
        scheduler.step(avg_loss)
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, KL Loss: {total_kl / len(train_loader):.4f}')

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

    train_glucose_np = train_data[:, 0].cpu().numpy() 
    
    _ = predict_train_uncertainty_and_plot(
            model_retrain,
            X_train,
            starts_train=starts_train,                       
            full_glucose_np=df['Glucose'].to_numpy(),         
            input_seq_length=input_seq_length,
            output_seq_length=output_seq_length,
            T=50,
            batch_size=16,
            device=device,
            save_path=f"train_uncertainty__{arch_tag(arch)}.png",
            csv_path=f"train_uncertainty__{arch_tag(arch)}.csv",
            ci_sigma=2.0,
    )
    
    
    import copy
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch

    @torch.no_grad()
    def mc_predict_window_with_weight_sampling(
        model_source,
        X_test,
        Y_test,
        idx,
        n_samples=50,
        device=None
    ):
        if device is None:
            device = next(model_source.parameters()).device

        named_param_dict = dict(model_source.named_parameters())
        param_names = [n for n in named_param_dict if n.endswith('W_mu') or n.endswith('bias_mu')]

        inp = X_test[idx].unsqueeze(0).to(device)  
        preds = []

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

            out = mc_model(inp)  
            preds.append(out.squeeze().cpu().numpy())  

        preds = np.stack(preds, axis=0)       
        mean_ = preds.mean(axis=0)
        var_  = preds.var(axis=0)
        std_  = np.sqrt(var_)
        lower = mean_ - 2.0 * std_
        upper = mean_ + 2.0 * std_

        y_true = Y_test[idx].squeeze().cpu().numpy()  

        return preds, mean_, std_, var_, lower, upper, y_true

    stride_raw = 1
    step = max(1, output_seq_length // stride_raw)  
    indices = list(range(len(X_test) - 1, -1, -step))

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

        assert (
            len(t_idx) == len(mean_) == len(lower) == len(upper) == len(y_true)
        ), f"Length mismatch for window {idx}: t={len(t_idx)}, mean={len(mean_)}, lower={len(lower)}, upper={len(upper)}, y={len(y_true)}"

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
    csv_name = f"mc_predictions_windows_{fname_ids}_{arch['encoder_channels']}_{arch['kernel_size']}.csv"
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
    plt.savefig(f"combined_windows_{indices[0]}_{indices[-1]}_{arch['encoder_channels']}_{arch['kernel_size']}.png", dpi=200)
    plt.close()



    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

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