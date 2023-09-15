import torch as t
from abc import ABC, abstractmethod
from typing import List
import einops


class InfluenceCalculable(ABC):
    @abstractmethod
    def get_a_l_minus_1(self):
        # Return the input to the linear layer
        pass

    @abstractmethod
    def get_d_s_l(self):
        # Return the gradient of the loss wrt the output of the linear layer
        pass

    @abstractmethod
    def get_dims(self):
        # Return the dimensions of the weights - (output_dim, input_dim)
        pass

    @abstractmethod
    def get_d_w_l(self):
        # Return the gradient of the loss wrt the weights
        pass


def autoregressive_loss(output, target):
    output = einops.rearrange(output, "b s v -> (b s) v")
    target = einops.rearrange(target, "b s -> (b s)")
    loss = t.nn.functional.cross_entropy(output, target)
    return loss


def get_ekfac_factors_and_train_grads(
    model, dataset, mlp_blocks: List[InfluenceCalculable], device
):
    kfac_input_covs = [
        t.zeros((b.get_dims()[1] + 1, b.get_dims()[1] + 1)).to(device)
        for b in mlp_blocks
    ]
    kfac_grad_covs = [
        t.zeros((b.get_dims()[0], b.get_dims()[0])).to(device) for b in mlp_blocks
    ]
    train_grads = [[] for _ in range(len(mlp_blocks))]
    tot = 0
    for data, target in dataset:
        model.zero_grad()
        data = data.to(device)
        target = target.to(device)
        if len(data.shape) == 1:
            data = data.unsqueeze(0)
            target = target.unsqueeze(0)
        output = model(data)
        loss = autoregressive_loss(output, target)
        for i, block in enumerate(mlp_blocks):
            a_l_minus_1 = block.get_a_l_minus_1()
            input_covs = t.einsum("...ti,...tj->tij", a_l_minus_1, a_l_minus_1)
            kfac_input_covs[i] += input_covs.mean(dim=0)
        loss.backward()
        for i, block in enumerate(mlp_blocks):
            d_s_l = block.get_d_s_l()
            grad_cov = t.einsum("...ti,...tj->tij", d_s_l, d_s_l)
            kfac_grad_covs[i] += grad_cov.mean(dim=0)
            train_grads[i].append(block.get_d_w_l())
        tot += 1
    kfac_input_covs = [A / tot for A in kfac_input_covs]
    kfac_grad_covs = [S / tot for S in kfac_grad_covs]
    return kfac_input_covs, kfac_grad_covs, train_grads


def compute_lambda_ii(train_grads, q_a, q_s):
    """Compute Lambda_ii values for a block."""
    n_examples = len(train_grads)
    squared_projections_sum = 0.0
    for j in range(n_examples):
        dtheta = train_grads[j]
        result = (q_s @ dtheta @ q_a.T).view(-1)
        squared_projections_sum += result**2
    lambda_ii_avg = squared_projections_sum / n_examples
    return lambda_ii_avg


def get_ekfac_ihvp(
    query_grads, kfac_input_covs, kfac_grad_covs, train_grads, damping=0.001
):
    """Compute EK-FAC inverse Hessian-vector products."""
    ihvp = []
    for i in range(len(query_grads)):
        q = query_grads[i]
        # Performing eigendecompositions on the input and gradient covariance matrices
        q_a, _, q_a_t = t.svd(kfac_input_covs[i])
        q_s, _, q_s_t = t.svd(kfac_grad_covs[i])
        lambda_ii = compute_lambda_ii(train_grads[i], q_a, q_s)
        ekfacDiag_damped_inv = 1.0 / (lambda_ii + damping)
        ekfacDiag_damped_inv = ekfacDiag_damped_inv.reshape((q.shape[0], q.shape[1]))
        intermediate_result = q_s @ (q @ q_a_t)
        result = intermediate_result / ekfacDiag_damped_inv
        ihvp_component = q_s_t @ (result @ q_a)
        ihvp.append(ihvp_component.reshape(-1))
    # Concatenating the results across blocks to get the final ihvp
    return t.cat(ihvp)


def get_query_grads(
    model, query, target, mlp_blocks: List[InfluenceCalculable], device
):
    model.zero_grad()
    query = query.to(device)
    target = target.to(device)
    if len(query.shape) == 1:
        # Add batch dimension
        query = query.unsqueeze(0)
        target = target.unsqueeze(0)
    output = model(query)
    loss = autoregressive_loss(output, target)
    loss.backward()
    grads = []
    for block in mlp_blocks:
        grads.append(block.get_d_w_l())
    return grads


def get_influences(ihvp, train_grads):
    """
    Compute influences using precomputed ihvp and train_grads.
    """
    influences = []
    for example_grads in zip(*train_grads):
        influences.append(
            t.dot(ihvp, t.cat([g.view(-1) for g in example_grads])).item()
        )
    return influences


def influence(
    model,
    mlp_blocks: List[InfluenceCalculable],
    test_dataset,
    train_dataset,
    topk,
    device,
):
    kfac_input_covs, kfac_grad_covs, train_grads = get_ekfac_factors_and_train_grads(
        model, train_dataset, mlp_blocks, device
    )

    all_top_training_samples = []
    all_top_influences = []

    for query, target in test_dataset:
        query_grads = get_query_grads(model, query, target, mlp_blocks, device)
        ihvp = get_ekfac_ihvp(query_grads, kfac_input_covs, kfac_grad_covs, train_grads)
        top_influences = get_influences(ihvp, train_grads)
        top_influences, top_samples = t.topk(t.tensor(top_influences), topk)
        all_top_training_samples.append(top_samples)
        all_top_influences.append(top_influences)

    return all_top_training_samples, all_top_influences
