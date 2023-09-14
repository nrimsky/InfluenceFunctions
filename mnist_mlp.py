import torch as t
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from random import sample
from influence_functions_mlp import influence, InfluenceCalculable
import matplotlib.pyplot as plt

# Define the hyperparameters
batch_size = 128
learning_rate = 0.001
num_epochs = 30
hidden_dim = 64
input_dim = 28 * 28  
output_dim = 10 
device = t.device("cuda" if t.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1)),  # Flatten
    ]
)


class MLPBlock(InfluenceCalculable, t.nn.Module):
    def __init__(self, input_dim, output_dim, use_relu=True):
        super().__init__()
        self.linear = t.nn.Linear(input_dim, output_dim)
        self.relu = t.nn.ReLU()
        self.input = None
        self.use_relu = use_relu
        self.d_s_l = None
        self.d_w_l = None
        # Save gradient of loss wrt output of linear layer (Ds_l, where s_l = self.linear(a_l_minus_1))
        def hook_fn(module, grad_input, grad_output):
            self.d_s_l = grad_output[0]
        self.linear.register_full_backward_hook(hook_fn)

    def forward(self, x):
        self.input = x
        x = self.linear(x)
        if self.use_relu:
            x = self.relu(x)
        return x

    def get_a_l_minus_1(self):
        # Return the input to the linear layer as a homogenous vector
        return t.cat([self.input, t.ones((self.input.shape[0], 1)).to(device)], dim=-1)

    def get_d_s_l(self):
        # Return the gradient of the loss wrt the output of the linear layer
        return self.d_s_l
    
    def get_dims(self):
        # Return the dimensions of the weights - (output_dim, input_dim)
        return self.linear.weight.shape
    
    def get_d_w_l(self):
        # Return the gradient of the loss wrt the weights
        w_grad = self.linear.weight.grad
        b_grad = self.linear.bias.grad.unsqueeze(-1)
        full_grad = t.cat([w_grad, b_grad], dim=-1)
        return full_grad
        


class MLP(t.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.fc1 = MLPBlock(input_dim, hidden_dim)
        self.fc2 = MLPBlock(hidden_dim, hidden_dim)
        self.fc3 = MLPBlock(hidden_dim, output_dim, use_relu=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def train_model():
    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = MLP(input_dim, output_dim, hidden_dim)
    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = t.nn.CrossEntropyLoss()

    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()  
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    with t.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = t.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(
        f"Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%"
    )

    # Save the model checkpoint
    t.save(model.state_dict(), "model.ckpt")

    return model, train_dataset, test_dataset



def run_influence(model_path):
    model = MLP(input_dim, output_dim, hidden_dim)
    model.load_state_dict(t.load(model_path))
    model = model.to(device)
    model.train()

    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    train_subset = t.utils.data.Subset(
        train_dataset, sample(range(len(train_dataset)), 10000)
    )

    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform)
    test_subset = t.utils.data.Subset(
        test_dataset, sample(range(len(test_dataset)), 10)
    )

    mlp_blocks = [model.fc1, model.fc2, model.fc3]

    all_top_training_samples, all_top_influences = influence(
        model, mlp_blocks, test_subset, train_subset, device
    )

    for i, (top_samples, top_influences) in enumerate(
        zip(all_top_training_samples, all_top_influences)
    ):
        print(f"Query target: {test_dataset[i][1]}")

        # Prepare a figure for visualization
        plt.clf()
        plt.figure(figsize=(2 * (len(top_samples) + 1), 2))

        # Display query image
        plt.subplot(1, len(top_samples) + 1, 1)
        query_img = test_dataset[i][0].view(28, 28)
        plt.imshow(query_img, cmap="gray")
        plt.title(f"Query: {test_dataset[i][1]}")
        plt.axis("off")

        for j, (sample_idx, infl) in enumerate(zip(top_samples, top_influences)):
            print(f"Sample target {train_dataset[sample_idx][1]}: {infl:.4f}")

            # Display influential training image
            plt.subplot(1, len(top_samples) + 1, j + 2)
            infl_img = train_dataset[sample_idx][0].view(28, 28)
            plt.imshow(infl_img, cmap="gray")
            plt.title(f"Influence: {infl:.4f}")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"results_{i}.png")

if __name__ == "__main__":
    # train_model()
    run_influence("model.ckpt")
