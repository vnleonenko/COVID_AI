import torch
import torch.nn as nn
from tqdm import tqdm


class SeirPINN(nn.Module):
    def __init__(self, t, I, options):
        super(SeirPINN, self).__init__()

        self.options = options

        self.N = self.options["N"]

        self.t = torch.tensor(t, requires_grad=True)
        self.t_float = self.t.float()
        self.t_batch = torch.reshape(self.t_float, (len(self.t), 1))

        self.I_hat = torch.reshape(torch.tensor(I).float(), (-1,))

        self.criterior = nn.MSELoss(reduction='mean')

        self.contact_rate_tilda = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.incubation_rate_tilda = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))
        self.infective_rate_tilda = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))

        self.net = self.Neural_net(self.options)

        self.net.register_parameter("contact_rate", self.contact_rate_tilda)
        self.net.register_parameter("incubation_rate", self.incubation_rate_tilda)
        self.net.register_parameter("infective_rate", self.infective_rate_tilda)

        self.params = list(self.net.parameters())

    @property
    def contact_rate(self):
        return (torch.tanh(self.contact_rate_tilda) *
                self.options["PARAMETERS_RANGE"]["contact_rate"]["radius"] +
                self.options["PARAMETERS_RANGE"]["contact_rate"]["center"])

    @property
    def incubation_rate(self):
        return (torch.tanh(self.incubation_rate_tilda) *
                self.options["PARAMETERS_RANGE"]["incubation_rate"]["radius"] +
                self.options["PARAMETERS_RANGE"]["incubation_rate"]["center"])

    @property
    def infective_rate(self):
        return (torch.tanh(self.infective_rate_tilda) *
                self.options["PARAMETERS_RANGE"]["infective_rate"]["radius"] +
                self.options["PARAMETERS_RANGE"]["infective_rate"]["center"])

    class Neural_net(nn.Module):
        def __init__(self, options):
            super(SeirPINN.Neural_net, self).__init__()

            self.options = options

            dim_hidden = self.options["NETWORK"]["nodes"]

            self.start = nn.Linear(1, dim_hidden)
            self.hidden_layers = [nn.Linear(dim_hidden, dim_hidden) for _ in
                                  range(self.options["NETWORK"]["hidden_layers_amount"])]
            self.out = nn.Linear(dim_hidden, 4)

        def forward(self, t_batch):
            seir = self.options["NETWORK"]["activation_function"](self.start(t_batch))

            for layer in self.hidden_layers:
                seir = self.options["NETWORK"]["activation_function"](layer(seir))

            seir = self.out(seir)
            return seir

    def loss_PDE(self, seir_hat):

        S_hat, E_hat, I_hat, R_hat = seir_hat[:, 0], seir_hat[:, 1], seir_hat[:, 2], seir_hat[:, 3]

        # S_t
        S_hat_t = torch.autograd.grad(
            S_hat, self.t_batch,
            grad_outputs=torch.ones_like(S_hat),
            retain_graph=True,
            create_graph=True
        )[0]

        # E_t
        E_hat_t = torch.autograd.grad(
            E_hat, self.t_batch,
            grad_outputs=torch.ones_like(E_hat),
            retain_graph=True,
            create_graph=True
        )[0]

        # I_t
        I_hat_t = torch.autograd.grad(
            I_hat, self.t_batch,
            grad_outputs=torch.ones_like(I_hat),
            retain_graph=True,
            create_graph=True
        )[0]

        # R_t
        R_hat_t = torch.autograd.grad(
            R_hat, self.t_batch,
            grad_outputs=torch.ones_like(R_hat),
            retain_graph=True,
            create_graph=True
        )[0]

        alpha = 1 / self.incubation_rate
        gamma = 1 / self.infective_rate
        beta = self.contact_rate / self.infective_rate

        loss_f1 = self.criterior(torch.reshape(S_hat_t.float(), (-1,)), (-beta * S_hat * I_hat).float())
        loss_f2 = self.criterior(torch.reshape(E_hat_t.float(), (-1,)),
                                 ((beta * S_hat * I_hat) - (alpha * E_hat)).float())
        loss_f3 = self.criterior(torch.reshape(I_hat_t.float(), (-1,)), (alpha * E_hat - gamma * I_hat).float())
        loss_f4 = self.criterior(torch.reshape(R_hat_t.float(), (-1,)), (gamma * I_hat).float())

        return loss_f1 + loss_f2 + loss_f3 + loss_f4

    def loss_U(self, seir_hat):
        I_pred = torch.reshape(seir_hat[:, 2], (-1,))
        loss = self.criterior(self.I_hat, I_pred)
        return loss

    def train(self, n_epochs):
        pbar = tqdm(range(n_epochs))

        for _ in pbar:
            self.optimizer.zero_grad()

            seir_hat = self.net(self.t_batch)

            lossU = self.loss_U(seir_hat)
            lossF = self.loss_PDE(seir_hat)

            loss = (lossU * self.options["LOSS"]["BALANCE_DELTA_U"] + lossF * self.options["LOSS"]["BALANCE_DELTA_F"])

            loss.backward()
            self.optimizer.step()

            if loss.item() < self.options["LOSS"]["THRESHOLD"]:
                break

            pbar.set_postfix({'loss': loss.item(),
                              'lossU': lossU.item(),
                              'lossF': lossF.item(),
                              "Contact_rate": self.contact_rate.item(),
                              "Incubation_rate": self.incubation_rate.item(),
                              "Infective_rate": self.infective_rate.item(),
                              "LR": self.optimizer.param_groups[0]['lr']
                              })