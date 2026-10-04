"""
===========================================
Various functions used across the notebooks
===========================================
"""

import numpy as np
import torch
from torch.func import jvp, vjp, functional_call
from tqdm import tqdm

def dict_to_vector(tensor_dict, reference_model):
    """
    Turn a dictionary of tensors (from model.named_parameters())
    into a D x 1 column vector   

    Parameters
    ----------
    tensor_dict : dict
        The output of model.named_parameters()
    reference_model : Net
        The neural network.

    Returns
    -------
    Tensor
        Network weights reshaped into Dx1 tensor

    """
    # retrieve the tensors in the same order as they appear in the network
    tensors = [tensor_dict[name] for name, _ in reference_model.named_parameters()]
    flat_vector = torch.nn.utils.parameters_to_vector(tensors)
    return flat_vector.unsqueeze(1)

def vector_to_dict(vector, reference_model):
    """
    Does the opposite of dict_to_vector
    """
    tensor_dict = {}
    pointer = 0
    # Loop door de originele parameters om de shapes en namen te achterhalen
    for name, param in reference_model.named_parameters():
        numel = param.numel()
        # Snijd het juiste deel uit de platte vector en geef het de originele vorm
        tensor_dict[name] = vector[pointer:pointer + numel].view(param.shape)
        pointer += numel
    return tensor_dict

def batch_compute_Gv(model, params, v_dict, x, N, batch_size = 10):
    """
    Compute the GGN-vector product over mini batches in a matrix-free manner using 
    autodiff.

    Parameters
    ----------
    model : Net
        The neural network
    params : dict
        The connection weights in the form {'name': tensor}, obtained
        from dict(model.named_parameters()).
    v_dict : dict
        The vector v reshaped to match the shapes of the
        weight tensors, in the form {'name': tensor}.
    x : tensor
        The input training data
    N : integer
        The number of training data.
    Returns
    -------
    Gv : tensor
        The D x 1 tensor Gv.

    """

    batches = torch.split(x, batch_size)
    Gv = 0.

    print(f'Computing Gv over {len(batches)} batches...')
    for i, x_i in enumerate(batches):

        def functional_forward_x(model_params):
            # run the model forward at the current batch
            return  functional_call(model, model_params, (x_i,))

        # compute w = Jv using forward autodiff
        _, w = jvp(functional_forward_x, (params,), (v_dict,))

        _, vjp_fn = vjp(functional_forward_x, params)

        Gv_dict = vjp_fn(w)[0]

        # List of tensors
        Gv_tensors = [1 / N * Gv_dict[name] for name, _ in model.named_parameters()]

        # Flatten into D x 1 vector
        Gv += torch.nn.utils.parameters_to_vector(Gv_tensors).unsqueeze(1)

    return Gv.detach()

def batch_compute_Gv_classification(model, params, v_dict, x, mask, N, batch_size = 128):
    """
    Compute the GGN-vector product over mini batches in a matrix-free manner using 
    autodiff.

    Parameters
    ----------
    params : dict
        The connection weights in the form {'name': tensor}, obtained
        from dict(model.named_parameters()).
    v_dict : dict
        The vector v reshaped to match the shapes of the
        weight tensors, in the form {'name': tensor}.

    Returns
    -------
    Gv : tensor
        The D x 1 tensor Gv.

    """

    batches = torch.split(x, batch_size)
    mask_x_i = torch.split(mask, batch_size)
    Gv = 0.

    print(f'Computing Gv over {len(batches)} batches...')
    for i, x_i in enumerate(batches):

        def functional_forward_x(model_params):
            logits, _ =  functional_call(model, model_params, (x_i, None))
            return logits

        # compute w = Jv using forward autodiff
        logits, w = jvp(functional_forward_x, (params,), (v_dict,))

        # For the GGN, treat H_l at theta_0 as a fixed output-space Hessian.
        logits = logits.detach()
        w = w.detach()

        # softmax probabilities
        p = torch.softmax(logits, dim=-1)
        # compute HJv
        pw = (p * w).sum(dim=-1, keepdim=True)  # [B, T, 1]
        Hw = p * (w - pw)                       # [B, T, K]
        Hw = Hw * mask_x_i[i].unsqueeze(-1)     # remove padding
        Hw = Hw / N                             # mean over valid tokens
        Hw = Hw.detach()

        # compute J^{T}HJv  using backward autodiff
        _, vjp_fn = vjp(functional_forward_x, params)
        Gv_dict = vjp_fn(Hw)[0]

        # List of tensors
        Gv_tensors = [Gv_dict[name] for name, _ in model.named_parameters()]

        # Flatten into D x 1 vector
        Gv += torch.nn.utils.parameters_to_vector(Gv_tensors).unsqueeze(1)

    return Gv.detach()

def lanczos(m, D, model, params, x, N, batch_size = 10, classification = False, mask = None, dtype=torch.float32):
    """
    The Lanczos algorithm. Computes an m x m matrix

    T = Q^T G Q

    T shares the dominant eigenvalues of G, and the dominant
    eigenvectors are computed using QV, V being the eigenvectors
    of T.

    Parameters
    ----------
    m : integer
        The number of Krylov basis functions.
    D : integer
        The number of connection weights.
    model : Net
        The neural network.
    params : dict
        The connection weights in the form {'name': tensor}, obtained
        from dict(model.named_parameters()).
    x : tensor
        The input training data.
    N : integer
        The number of training data.
    batch_size : integer
        The batch size used to compute Gv products.
    classification : boolean
        Flag for regression / classification. The default is False (regression).
    mask : tensor
        A mask that flags padded entries, same shape as x, used for classification.
        The default is None.
    dtype : torch datatype, optional
        The default is torch.float32.

    Returns
    -------
    Q : tensor (D, m)
        The orthogonal Krylov basis.
    T : tensor (m, m)
        The low rank matrix used to approximate dominant G eigenspace.

    """
    Q = torch.zeros((D, m), dtype=dtype)
    alpha = torch.zeros(m, dtype=dtype)
    beta = torch.zeros(m, dtype=dtype)

    v_vector = torch.randn(D, 1, dtype=dtype)
    q = v_vector / torch.linalg.norm(v_vector)
    # use detach, we only want the value of q, not its derivative,
    # without detach it can cause the computer to hang
    # q = q.detach()
    q_prev = q.clone()

    print('Running Lanczos iterations...')
    for k in tqdm(range(m)):
        Q[:, k] = q.squeeze(1).clone().detach()

        # Matrix-vector product via jouw Gv routine
        q_dict = vector_to_dict(q, model)
        if classification:
            z = batch_compute_Gv_classification(model, params, q_dict, x, mask, N, batch_size = batch_size).detach()
        else:
            z = batch_compute_Gv(model, params, q_dict, x, N, batch_size = batch_size).detach()

        # Diagonal term
        alpha[k] = (q.T @ z).item()

        # Orthogonalize 
        if k == 0:
            z = z - alpha[k] * q
        else:
            z = z - alpha[k] * q - beta[k-1] * q_prev

        # Full reorthogonalization
        for j in range(k):
            coeff = (Q[:, j] @ z.squeeze(1)).item()
            z -= coeff * Q[:, j].unsqueeze(1)

        beta[k] = torch.linalg.norm(z)

        if beta[k] < 1e-10:
            print("Early termination at step", k)
            return Q[:, :k+1], build_tridiag(alpha[:k+1], beta[:k])

        q_prev = q.clone().detach()
        q = (z / beta[k]).clone().detach()

    T = build_tridiag(alpha, beta[:-1])

    return Q, T

def build_tridiag(alpha, beta):
    """
    Build the tridiagonal matrix T comprised of alpha, beta 
    coefficients of the Lanczos algorihm.

    Parameters
    ----------
    alpha : tensor (m, )
        The alpha coefficients of the Lanczos algorithm
    beta : tensor (m, )
        The beta coefficients of the Lanczos algorithm

    Returns
    -------
    T : tensor, (m , m)
        The tridiagonal matrix with alpha, beta entries.

    """
    m = len(alpha)
    T = torch.zeros((m, m))
    for i in range(m):
        T[i, i] = alpha[i]
        if i < m - 1:
            T[i, i+1] = beta[i]
            T[i+1, i] = beta[i]
    return T

def solve_sigma0_lowrank(eigvals, w_norm_sq, d, beta, tol=1e-8):
    """
    Root finding algorithm for alpha = sigam_0^{-2}, solving:

    ||w_0||2_2 = \sum_{i=1}^d 1/alpha - 1/(beta * lambda_i + alpha)

    Uses the bisection method.

    Parameters
    ----------
    eigvals : tensor
        The d dominant eigenvalues of G.
    w_norm_sq : float
        The (projected) parameter norm ||w_0||^2_2.
    d : integer
        DESCRIPTION.
    beta : float
        The temperature parameter.
    tol : float, optional
        Tolerance of the root finding. The default is 1e-8.

    Raises
    ------
    RuntimeError
        If the root cannot be bracketed before the bisection starts.

    Returns
    -------
    sigma^2_0 : float
        The prior variance.

    """
    eigvals = eigvals[:d].detach().cpu().numpy()

    def f(alpha, beta):
        term1 = np.sum(1/alpha - 1/(beta*eigvals + alpha))
        return term1 - w_norm_sq

    # bracket the root
    lo = 1e-8
    hi = 1.0

    # increase alpha_high until sign changes
    while f(hi, beta) > 0:
        hi *= 2.0
        if hi > 1e8:
            raise RuntimeError("Failed to bracket root")

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if f(mid, beta) > 0:
            lo = mid
        else:
            hi = mid

        if abs(f(mid, beta)) < tol:
            print(f'Converged to root within {f(mid, beta):.2e}')
            break

    alpha = mid
    print(f'alpha = {alpha}')
    return 1.0 / alpha

def sample(x, model, theta_0, theta_0_dict, sigma_post, eigvecs, restore = True):
    """
    Posterior sampling using the generalized Laplace Active Subspace.

    Parameters
    ----------
    x : tensor
        input features.
    model : Net
        The neural network.
    theta_0 : tensor (D, 1)
        The pretrained connection weights.
    sigma_post : tensor (d, )
        The posterior standard deviations.
    eigvecs : tensor (D, d)
        The dominant eigenvectors.
    restore : boolean, optional
        After sampling, restore the pretrained weights.
    Returns
    -------
    f : tensor
        The posterior prediction. 
    """

    d = eigvecs.shape[1]
    z1 = torch.randn(d, dtype=torch.float32) * sigma_post  # sigma_j
    low_rank = eigvecs @ z1.unsqueeze(1)

    theta_sample = theta_0 + low_rank
    theta_dict = vector_to_dict(theta_sample, model)

    # overwrite weights
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in theta_dict:
                param.copy_(theta_dict[name])

    f = model(x).squeeze(1) #* y_std + y_mean

    # restore weights to pretrained values
    if restore:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in theta_0_dict:
                    param.copy_(theta_0_dict[name])

    return f

def sample_reinvent(batch_size, model, theta_0, theta_0_dict, sigma_post, eigvecs, 
                    device, vocabulary, tokenizer, max_sequence_length=256,
                    laplace=False, d = 1, sampling_mode='random'):
    """
    Generative molecule sampling with generalized Laplace Active
    Subspaces.

    Modified from REINVENT model._sample, see
    reinvent/models/reinvent/model/model.py
    """

    START_TOKEN = '^'

    if laplace:
        z1 = torch.randn(d) * sigma_post  # sigma_j
        low_rank = eigvecs[:, 0:d] @ z1.unsqueeze(1)

        theta_sample = theta_0 + low_rank
        theta_dict = vector_to_dict(theta_sample, model.network)

        # overwrite weights
        with torch.no_grad():
            for name, param in model.network.named_parameters():
                if name in theta_dict:
                    param.copy_(theta_dict[name])

    # NOTE: the first token never gets added in the loop so initialize with the start token
    sequences = [
        torch.full(
            (batch_size, 1),
            vocabulary[START_TOKEN],
            dtype=torch.long,
            device=device,
        )
    ]
    input_vector = torch.full(
        (batch_size,), vocabulary[START_TOKEN], dtype=torch.long, device=device
    )
    hidden_state = None
    nlls = torch.zeros(batch_size, device=device)

    with torch.inference_mode():
        for _ in range(max_sequence_length - 1):
            logits, hidden_state = model.network(input_vector.unsqueeze(1), hidden_state)
            logits = logits.squeeze(1)  # 2D
            log_probs = logits.log_softmax(dim=1)  # 2D
            probabilities = logits.softmax(dim=1)  # 2D
            if sampling_mode == 'random':
                input_vector = torch.multinomial(probabilities, num_samples=1).view(-1)  # 1D
            elif sampling_mode == 'max': 
                input_vector = probabilities.argmax(dim=1).view(-1)
            else:
                raise ValueError(f"Invalid sampling mode: {sampling_mode}")
            sequences.append(input_vector.view(-1, 1))
            nlls += model._nll_loss(log_probs, input_vector)

            if input_vector.sum() == 0:
                break

    concat_sequences = torch.cat(sequences, dim=1)
    seqs = concat_sequences.detach()

    smiles = [
        tokenizer.untokenize(model.vocabulary.decode(seq)) for seq in seqs.cpu().numpy()
    ]

    if laplace:
        # restore weights to pretrained values
        with torch.no_grad():
            for name, param in model.network.named_parameters():
                if name in theta_0_dict:
                    param.copy_(theta_0_dict[name])

    return smiles, seqs, nlls

def linear_laplace(xx, model, theta_0_dict, eigvecs, sigma_post, d):

    # mean at the MAP prediction
    mean_f = model(xx)

    def functional_forward_xx(model_params):
        # run the model forward at locations given by xx
        return functional_call(model, model_params, xx)

    # compute the pointwise linear Laplace variance
    var_f = torch.zeros_like(mean_f)
    for i in range(d):
        # compute J(xx) * p_i for i = 1,...,d
        p_i_dict = vector_to_dict(eigvecs[:,i], model)
        _, Jp_i = jvp(functional_forward_xx, (theta_0_dict,), (p_i_dict,))
        # compute point wise variance
        var_f += sigma_post[i] ** 2 * Jp_i ** 2
    std_f = var_f.sqrt()

    return mean_f, std_f

