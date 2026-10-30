import torch
import torch.nn as nn
import torch.optim as optim

# timeGAN implementation inspired by https://github.com/benearnthof/TimeGAN
class Embedder(nn.Module):
    """maps original data space to latent space: h_t = e_X (h_S, h_{t-1}, x_t)"""
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Embedder, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        return torch.sigmoid(self.linear(out))


class Recovery(nn.Module):
    """maps latent space back to data space: x_t = r_X (e_t)"""
    def __init__(self, hidden_dim, output_dim, num_layers):
        super(Recovery, self).__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, h):
        out, _ = self.rnn(h)
        return self.linear(out)


class Generator(nn.Module):
    """generates synthetic representations in latent space: h_t = g_S(h_S, h_{t-1}, z_t"""
    def __init__(self, noise_dim, hidden_dim, num_layers):
        super(Generator, self).__init__()
        self.rnn = nn.GRU(noise_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, z):
        out, _ = self.rnn(z)
        return torch.sigmoid(self.linear(out))


class Supervisor(nn.Module):
    """captures temporal dynamics in latent space"""
    def __init__(self, hidden_dim, num_layers):
        super(Supervisor, self).__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers - 1, batch_first=True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, h):
        out, _ = self.rnn(h)
        return torch.sigmoid(self.linear(out))


class Discriminator(nn.Module):
    """distinguishes between real latent codes and synthetic codes"""
    def __init__(self, hidden_dim, num_layers):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, h):
        out, _ = self.rnn(h)
        return self.linear(out)


class TimeGAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, noise_dim, num_layers):
        super(TimeGAN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        
        # initialise sub-networks
        self.embedder = Embedder(input_dim, hidden_dim, num_layers)
        self.recovery = Recovery(hidden_dim, input_dim, num_layers)
        self.generator = Generator(noise_dim, hidden_dim, num_layers)
        self.supervisor = Supervisor(hidden_dim, num_layers)
        self.discriminator = Discriminator(hidden_dim, num_layers)

    def train_timegan(self, real_data, epochs=100, batch_size=64, lr=0.001):
        """
        executes the training phases of TimeGAN:
        1) embedding network training (reconstruction)
        2) supervisor training (temporal dynamics)
        3) joint training (generator, discriminator, supervisor)
        """
        # optimise using ADAM
        embedder_optimiser  = optim.Adam(self.embedder.parameters(), lr=lr)
        recovery_optimiser  = optim.Adam(self.recovery.parameters(), lr=lr)
        supervisor_optimiser   = optim.Adam(self.supervisor.parameters(), lr=lr)
        generator_optimiser   = optim.Adam(self.generator.parameters(), lr=lr)
        discriminator_optimiser   = optim.Adam(self.discriminator.parameters(), lr=lr)
        
        binary_cross_entropy_loss = nn.BCEWithLogitsLoss()
        MSE_loss = nn.MSELoss()
        
        dataset_size = real_data.size(0)
        
        print("starting embedding Network Training (Autoencoder)")
        for epoch in range(epochs):
            print(epoch)
            for i in range(0, dataset_size, batch_size):
                X = real_data[i:i + batch_size]
                
                H = self.embedder(X)
                X_tilde = self.recovery(H)
                
                # reconstruction Loss
                E_loss0 = MSE_loss(X_tilde, X) * 10.0
                
                embedder_optimiser.zero_grad()
                recovery_optimiser.zero_grad()
                E_loss0 .backward()
                embedder_optimiser.step()
                recovery_optimiser.step()
                
        print("starting supervisor training")
        for epoch in range(epochs):
            print(epoch)
            for i in range(0, dataset_size, batch_size):
                X = real_data[i:i + batch_size]
                
                H = self.embedder(X)
                H_hat_supervised = self.supervisor(H)
                
                # supervisor Loss (next step prediction in latent space)
                G_loss_S = MSE_loss(H_hat_supervised[:, :-1, :], H[:, 1:, :])
                
                supervisor_optimiser.zero_grad()
                G_loss_S.backward()
                supervisor_optimiser.step()
                
        print("starting joint training (adversarial + supervised)")
        for epoch in range(epochs):
            print(epoch)
            for i in range(0, dataset_size, batch_size):
                X = real_data[i:i + batch_size]
                current_batch_size = X.size(0)
                seq_len = X.size(1)
                
                # train generator
                Z = torch.randn(current_batch_size, seq_len, self.noise_dim, device=X.device)

                H = self.embedder(X)
                
                e_hat = self.generator(Z)

                H_hat = self.supervisor(e_hat)
                
                Y_fake = self.discriminator(H_hat)

                X_hat = self.recovery(H_hat)
                
                # G_loss_U, G_loss_S, G_loss_V
                G_loss_U  = binary_cross_entropy_loss(Y_fake, torch.ones_like(Y_fake))
                G_loss_S  = MSE_loss(H_hat[:, :-1, :], H[:, 1:, :])
                
                G_loss_V1 = torch.mean(torch.abs(torch.std(X_hat, dim=0) - torch.std(X, dim=0)))
                G_loss_V2 = torch.mean(torch.abs(torch.mean(X_hat, dim=0) - torch.mean(X, dim=0))) 
                
                G_loss_V = G_loss_V1 + G_loss_V2

                # combine all 3 generator losses into 1 loss
                loss_g = G_loss_U + G_loss_S + G_loss_V 
                
                generator_optimiser.zero_grad()
                loss_g.backward()
                generator_optimiser.step()
                
                # train discriminator
                Z = torch.randn(current_batch_size, seq_len, self.noise_dim, device=X.device)
                e_hat = self.generator(Z)
                H_hat = self.supervisor(e_hat).detach()
                
                Y_real = self.discriminator(H.detach())
                Y_fake = self.discriminator(H_hat)
                
                D_loss_real = binary_cross_entropy_loss(Y_real, torch.ones_like(Y_real))
                D_loss_fake = binary_cross_entropy_loss(Y_fake, torch.zeros_like(Y_fake))

                # combine discriminator losses
                loss_d = D_loss_real + D_loss_fake
                
                discriminator_optimiser.zero_grad()
                loss_d.backward()
                discriminator_optimiser.step()

    @torch.no_grad()
    def generate(self, num_samples, seq_len, device):
        """generates synthetic time-series data with the learned distribution"""
        self.eval()
        Z = torch.randn(num_samples, seq_len, self.noise_dim, device=device)
        E_hat = self.generator(Z)
        H_hat = self.supervisor(E_hat)
        X_hat = self.recovery(H_hat)
        return X_hat

