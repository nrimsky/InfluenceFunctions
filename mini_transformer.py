import torch as t
import einops
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data import Dataset
import string
from influence_functions_transformer import influence, InfluenceCalculable

d_model = 120
n_heads = 4
d_mlp = 256
n_layers = 2
vocab_size = 128
dataset_length = 20
sequence_length = 5


def autoregressive_loss(output, target):
    output = einops.rearrange(output, "b s v -> (b s) v")
    target = einops.rearrange(target, "b s -> (b s)")
    loss = t.nn.functional.cross_entropy(output, target)
    return loss


class CharPredictDataset(Dataset):
    def __init__(self, length, seq_length):
        self.data = self._generate_data(length)
        self.seq_length = seq_length

    def _generate_data(self, length):
        alphabets = string.ascii_lowercase
        numbers = [str(i % 10) for i in range(length)]
        return "".join([alphabets[i] + numbers[i] for i in range(length)])

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of range")
        source_seq = self.data[idx : idx + self.seq_length]
        return t.tensor(
            [ord(c) for c in source_seq[:-1]], dtype=t.long
        ), t.tensor([ord(c) for c in source_seq[1:]], dtype=t.long)


class MultiHeadMaskedAttention(t.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = t.nn.Linear(d_model, d_model)
        self.k_proj = t.nn.Linear(d_model, d_model)
        self.v_proj = t.nn.Linear(d_model, d_model)
        self.out_proj = t.nn.Linear(d_model, d_model)

    def forward(self, X, mask=None):
        Q = einops.rearrange(self.q_proj(X), "b s (h d) -> b h s d", h=self.n_heads)
        K = einops.rearrange(self.k_proj(X), "b s (h d) -> b h s d", h=self.n_heads)
        V = einops.rearrange(self.v_proj(X), "b s (h d) -> b h s d", h=self.n_heads)

        # Compute the scaled dot-product attention
        QK = t.einsum("b h i d, b h j d -> b h i j", Q, K)
        QK = QK / t.sqrt(t.tensor(self.d_head))
        if mask is not None:
            QK = QK.masked_fill(mask, -1e9)
        QK = t.nn.functional.softmax(QK, dim=-1)

        # Compute the output
        Y = t.einsum("b h i j, b h j d -> b h i d", QK, V)
        Y = einops.rearrange(Y, "b h s d -> b s (h d)")

        # Apply the output projection
        Y = self.out_proj(Y)

        return Y


class MLPBlock(InfluenceCalculable, t.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear = t.nn.Linear(input_dim, hidden_dim)
        self.relu = t.nn.ReLU()
        self.linear2 = t.nn.Linear(hidden_dim, output_dim)
        self.input = None
        # Save gradient of loss wrt output of linear layer (Ds_l, where s_l = self.linear(a_l_minus_1))
        def hook_fn(module, grad_input, grad_output):
            self.d_s_l = grad_output[0]
        self.linear.register_full_backward_hook(hook_fn)

    def forward(self, x):
        self.input = x
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

    def get_a_l_minus_1(self):
        # Return the input to the linear layer as a homogenous vector (batch_size, seq_len, input_dim + 1)
        return t.cat([self.input, t.ones((self.input.shape[0], self.input.shape[1], 1)).to(self.input.device)], dim=-1)

    def get_d_s_l(self):
        # Return the gradient of the loss wrt the output of the linear layer
        return self.d_s_l
    
    def get_dims(self):
        # Return the dimensions of the weights - (output_dim, input_dim)
        return self.linear.weight.shape
    
    def get_d_w_l(self):
        # Return the gradient of the loss wrt the weights
        w_grad = self.linear.weight.grad
        b_grad = self.linear.bias.grad
        return t.cat([w_grad.view(-1), b_grad.view(-1)])


class TransformerBlock(t.nn.Module):
    def __init__(self, d_model, n_heads, d_mlp):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_mlp = d_mlp
        self.attn = MultiHeadMaskedAttention(d_model, n_heads)
        self.mlp = MLPBlock(d_model, d_mlp, d_model)
        self.layer_norm1 = t.nn.LayerNorm(d_model)
        self.layer_norm2 = t.nn.LayerNorm(d_model)

    def forward(self, X, mask=None):
        attn_output = self.attn(X, mask)
        X = self.layer_norm1(X + attn_output)
        mlp_output = self.mlp(X)
        Y = self.layer_norm2(X + mlp_output)
        return Y


class DecoderTransformer(t.nn.Module):
    def __init__(self, d_model, n_heads, d_mlp, n_layers, vocab_size, max_seq_len=5):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_mlp = d_mlp
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.embed_input = t.nn.Embedding(vocab_size, d_model)
        self.blocks = t.nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_mlp) for _ in range(n_layers)]
        )
        self.out_proj = t.nn.Linear(d_model, vocab_size)
        self.position_embeddings = t.nn.Embedding(max_seq_len, d_model)
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")

    def forward(self, X):
        seq_len = X.size(-1)
        mask = t.triu(t.ones(seq_len, seq_len), diagonal=1).bool().to(self.device)
        X = self.embed_input(X)
        positions = t.arange(0, seq_len, device=X.device).unsqueeze(0)
        X = X + self.position_embeddings(positions)

        for block in self.blocks:
            X = block(X, mask)
        Y = self.out_proj(X)
        return Y


def train_loop(model, data_loader, optimizer, num_epochs=2):
    model.train()
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model.to(device)
    print_every = num_epochs // 50
    print_every = 1 if print_every == 0 else print_every
    for epoch in range(num_epochs):
        total_loss = 0
        for model_input, target in data_loader:
            model_input, target = model_input.to(device), target.to(device)
            output = model(model_input)
            loss = autoregressive_loss(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


def train_char_predict(n_epochs=500):
    small_transformer = DecoderTransformer(
        d_model=d_model,
        n_heads=n_heads,
        d_mlp=d_mlp,
        n_layers=n_layers,
        vocab_size=vocab_size,
        max_seq_len=5,
    )
    dataset = CharPredictDataset(length=dataset_length, seq_length=sequence_length)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = Adam(small_transformer.parameters(), lr=0.001)
    train_loop(small_transformer, data_loader, optimizer, num_epochs=n_epochs)
    t.save(small_transformer.state_dict(), "small_transformer.pth")


def calc_influence(model_path):
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    train_dataset = CharPredictDataset(length=dataset_length, seq_length=sequence_length)
    model = DecoderTransformer(d_model, n_heads, d_mlp, n_layers, vocab_size)
    model.load_state_dict(t.load(model_path))
    model.to(device)
    model.train()

    def encode(string):
        return t.tensor([ord(c) for c in string], dtype=t.long).to(device)

    queries = [
        (encode("m2n3"), encode("o")),
        (encode("q6r7"), encode("s")),
        (encode("4f5g"), encode("6")),
    ]

    all_top_training_samples, all_top_influences = influence(
        model,
        [b.mlp for b in model.blocks],
        queries,
        train_dataset,
        device,
    )

    def decode(token_ids):
        try:
            return "".join([chr(i) for i in token_ids])
        except:
            return chr(token_ids)

    for i, (top_samples, top_influences) in enumerate(
        zip(all_top_training_samples, all_top_influences)
    ):
        print(f"Query: Input {decode(queries[i][0])} Target: {decode(queries[i][1])}")
        print("Top 10 training samples and their influences:")
        for s, i in zip(top_samples, top_influences):
            s = s.item()
            print(
                f"Sample: {decode(train_dataset[s][0])} {decode(train_dataset[s][1])} Influence: {i}"
            )


def run_model(model_path):
    d_model = 120
    n_heads = 4
    d_mlp = 256
    n_layers = 2
    vocab_size = 128

    model = DecoderTransformer(d_model, n_heads, d_mlp, n_layers, vocab_size)
    model.load_state_dict(t.load(model_path))

    model.eval()
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model.to(device)

    while True:
        user_input = input("Enter a string: ")
        if user_input == "exit":
            return
        if len(user_input) > 4:
            user_input = user_input[-4:]
        token_ids = t.tensor([[ord(c) for c in user_input]], dtype=t.long).to(
            device
        )
        model_output = model(token_ids)
        last_token = model_output[0, -1, :]
        topk = t.topk(last_token, 1)
        topk_tokens = [chr(int(i)) for i in topk.indices.tolist()]
        print(topk_tokens[0])


if __name__ == "__main__":
    # train_char_predict()
    # run_model("small_transformer.pth")
    calc_influence("small_transformer.pth")
