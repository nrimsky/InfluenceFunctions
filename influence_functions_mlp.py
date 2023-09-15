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
        target = t.tensor(target).to(device)
        if len(data.shape) == 1:
            data = data.unsqueeze(0)
        if len(target.shape) == 0:
            target = target.unsqueeze(0)
        output = model(data)
        loss = t.nn.functional.cross_entropy(output, target)
        for i, block in enumerate(mlp_blocks):
            input_cov = t.einsum(
                "...i,...j->ij", block.get_a_l_minus_1(), block.get_a_l_minus_1()
            )
            kfac_input_covs[i] += input_cov
        loss.backward()
        for i, block in enumerate(mlp_blocks):
            grad_cov = t.einsum("...i,...j->ij", block.get_d_s_l(), block.get_d_s_l())
            kfac_grad_covs[i] += grad_cov
            train_grads[i].append(block.get_d_w_l().cpu())
        tot += 1
    kfac_input_covs = [A.cpu() / tot for A in kfac_input_covs]
    kfac_grad_covs = [S.cpu() / tot for S in kfac_grad_covs]
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


def get_ekfac_ihvp(kfac_input_covs, kfac_grad_covs, train_grads, damping=0.001):
    """Compute EK-FAC inverse Hessian-vector products."""
    ihvp = []
    for i in range(len(train_grads)):
        V = train_grads[i]
        stacked = t.stack(V)
        # Performing eigendecompositions on the input and gradient covariance matrices
        q_a, _, q_a_t = t.svd(kfac_input_covs[i])
        q_s, _, q_s_t = t.svd(kfac_grad_covs[i])
        lambda_ii = compute_lambda_ii(train_grads[i], q_a, q_s)
        ekfacDiag_damped_inv = 1.0 / (lambda_ii + damping)
        ekfacDiag_damped_inv = ekfacDiag_damped_inv.reshape((stacked.shape[-2], stacked.shape[-1]))
        intermediate_result = t.einsum("bij,jk->bik", stacked, q_a_t)
        intermediate_result = t.einsum("ji,bik->bjk", q_s, intermediate_result)
        result = intermediate_result / ekfacDiag_damped_inv.unsqueeze(0)
        ihvp_component = t.einsum("bij,jk->bik", result, q_a)
        ihvp_component = t.einsum("ji,bik->bjk", q_s_t, ihvp_component)
        # flattening the result except for the batch dimension
        ihvp_component = einops.rearrange(ihvp_component, "b j k -> b (j k)")
        ihvp.append(ihvp_component)
    # Concatenating the results across blocks to get the final ihvp
    return t.cat(ihvp, dim=-1)


def get_query_grads(
    model, query, target, mlp_blocks: List[InfluenceCalculable], device
):
    query = query.to(device)
    target = t.tensor(target).to(device)
    if len(query.shape) == 1:
        # Add batch dimension
        query = query.unsqueeze(0)
    if len(target.shape) == 0:
        target = target.unsqueeze(0)
    output = model(query)
    loss = t.nn.functional.cross_entropy(output, target)
    model.zero_grad()
    loss.backward()
    grads = []
    for block in mlp_blocks:
        grads.append(block.get_d_w_l())
    return grads


def get_influences(ihvp, query_grads):
    """
    Compute influences using precomputed ihvp and train_grads.
    """
    query_grads = t.cat([q.view(-1) for q in query_grads]).cpu()
    return t.einsum("ij,j->i", ihvp, query_grads)


def influence(
    model,
    mlp_blocks: List[InfluenceCalculable],
    test_dataset,
    train_dataset,
    device,
):
    kfac_input_covs, kfac_grad_covs, train_grads = get_ekfac_factors_and_train_grads(
        model, train_dataset, mlp_blocks, device
    )

    all_top_training_samples = []
    all_top_influences = []

    for query, target in test_dataset:
        query_grads = get_query_grads(model, query, target, mlp_blocks, device)
        ihvp = get_ekfac_ihvp(kfac_input_covs, kfac_grad_covs, train_grads)
        top_influences = get_influences(ihvp, query_grads)
        top_influences, top_samples = t.topk(top_influences, 10)
        all_top_training_samples.append(top_samples)
        all_top_influences.append(top_influences)

    return all_top_training_samples, all_top_influences
