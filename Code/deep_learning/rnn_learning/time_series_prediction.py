import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pylab as plt

N = 100
L = 1000
T = 20

x = np.empty((N, L), np.float32)
x[:] = np.array(range(L)) + np.random.randint(-4*T, 4*T, N).reshape(N, 1)
y = np.sin(x/T)

# plt.figure(figsize=(10, 8))
# plt.plot(y[0])
# plt.show()

class LSTMPredictor(nn.Module):
    def __init__(self):
        super(LSTMPredictor, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.float32)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.float32)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.float32)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.float32)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)
        
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        
        outputs = torch.cat(outputs, dim=1)
        return outputs

if __name__ == '__main__':
    # y = 100, 1000
    train_input = torch.from_numpy(y[3:, :-1]) # 97, 999
    train_target = torch.from_numpy(y[3:, 1:]) # 97, 999
    test_input = torch.from_numpy(y[:3, :-1])  #  3, 999
    test_target = torch.from_numpy(y[:3, 1:])  #  3, 999

    model = LSTMPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(model.parameters(), lr=0.8)

    n_steps = 10
    
    for i in range(n_steps):
        print(f"Step {i}")

        def closure():
            optimizer.zero_grad()
            out = model(train_input)
            loss = criterion(out, train_target)

            print(f"Loss: {loss.item()}")

            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            future = 1000
            pred = model(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)

            print(f"Test loss", loss.item())
            
            y_pred = pred.detach().numpy()
            plt.figure(figsize=(12, 6))
            plt.title(f"Step {i + 1}")
            plt.xlabel("x")
            plt.xlabel("y")

            n_data = train_input.shape[1]
            def draw(y_i, color):
                plt.plot(np.arange(n_data), y_i[:n_data], color, linewidth=2.0)
                plt.plot(np.arange(n_data, n_data + future), y_i[n_data:], color + ":", linewidth=2.0)

            draw(y_pred[0], 'r')
            draw(y_pred[1], 'b')
            draw(y_pred[2], 'g')

            plt.savefig(f"predict{i}.pdf")
            plt.close()
