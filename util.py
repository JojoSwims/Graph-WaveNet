import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """Standard z-score normalization for per-node features."""

    def __init__(self, mean, std, eps=1e-6):
        self.mean = np.asarray(mean)
        self.std = np.maximum(np.asarray(std), eps)
        self.num_nodes = self.mean.shape[-1]

    def transform(self, data):
        mean, std = self._prepare_params(data)
        return (data - mean) / std

    def inverse_transform(self, data):
        mean, std = self._prepare_params(data)
        return (data * std) + mean

    def _prepare_params(self, data):
        node_axis = self._get_node_axis(data)
        if isinstance(data, torch.Tensor):
            mean = torch.as_tensor(self.mean, dtype=data.dtype, device=data.device)
            std = torch.as_tensor(self.std, dtype=data.dtype, device=data.device)
            view_shape = [1] * data.dim()
        else:
            array = np.asarray(data)
            mean = np.asarray(self.mean, dtype=array.dtype)
            std = np.asarray(self.std, dtype=array.dtype)
            view_shape = [1] * array.ndim
        view_shape[node_axis] = self.num_nodes
        mean = mean.reshape(view_shape)
        std = std.reshape(view_shape)
        return mean, std

    def _get_node_axis(self, data):
        shape = data.shape if not isinstance(data, torch.Tensor) else tuple(data.size())
        for axis, size in enumerate(shape):
            if size == self.num_nodes:
                return axis
        raise ValueError("Unable to determine node axis for data shape {}.".format(shape))


class LogZScoreScaler:
    """Apply log1p followed by z-score normalization per node."""

    def __init__(self, mean, std, eps=1e-6):
        self.mean = np.asarray(mean)
        self.std = np.maximum(np.asarray(std), eps)
        self.num_nodes = self.mean.shape[-1]

    def transform(self, data):
        log_data = self._log1p(data)
        mean, std = self._prepare_params(log_data)
        return (log_data - mean) / std

    def inverse_transform(self, data):
        mean, std = self._prepare_params(data)
        log_data = (data * std) + mean
        return self._expm1(log_data)

    def _prepare_params(self, data):
        node_axis = self._get_node_axis(data)
        if isinstance(data, torch.Tensor):
            mean = torch.as_tensor(self.mean, dtype=data.dtype, device=data.device)
            std = torch.as_tensor(self.std, dtype=data.dtype, device=data.device)
            view_shape = [1] * data.dim()
        else:
            array = np.asarray(data)
            dtype = array.dtype
            mean = np.asarray(self.mean, dtype=dtype)
            std = np.asarray(self.std, dtype=dtype)
            view_shape = [1] * array.ndim
        view_shape[node_axis] = self.num_nodes
        mean = mean.reshape(view_shape)
        std = std.reshape(view_shape)
        return mean, std

    def _get_node_axis(self, data):
        shape = data.shape if not isinstance(data, torch.Tensor) else tuple(data.size())
        for axis, size in enumerate(shape):
            if size == self.num_nodes:
                return axis
        raise ValueError("Unable to determine node axis for data shape {}.".format(shape))

    @staticmethod
    def _log1p(data):
        if isinstance(data, torch.Tensor):
            return torch.log1p(data)
        return np.log1p(data)

    @staticmethod
    def _expm1(data):
        if isinstance(data, torch.Tensor):
            return torch.expm1(data)
        return np.expm1(data)



def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj


def load_dataset(
        dataset_dir,
        batch_size,
        valid_batch_size=None,
        test_batch_size=None,
        scaler_type='log1z'):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    target_train = data['x_train'][..., 0]
    if scaler_type == 'log1z':
        log_train = np.log1p(target_train)
        mean = log_train.mean(axis=(0, 1))
        std = log_train.std(axis=(0, 1))
        scaler = LogZScoreScaler(mean=mean, std=std)
        transform = scaler.transform
    elif scaler_type == 'standard':
        mean = target_train.mean(axis=(0, 1))
        std = target_train.std(axis=(0, 1))
        std = np.maximum(std, 1e-6)
        scaler = StandardScaler(mean=mean, std=std)
        transform = scaler.transform
    else:
        raise ValueError("Unsupported scaler_type '{}'.".format(scaler_type))

    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = transform(data['x_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse


def masked_huber(preds, labels, null_val=np.nan, delta=1.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    abs_diff = torch.abs(preds - labels)
    delta_tensor = torch.tensor(delta, dtype=abs_diff.dtype, device=abs_diff.device)
    quadratic = torch.where(abs_diff < delta_tensor, 0.5 * abs_diff ** 2 / delta_tensor, torch.zeros_like(abs_diff))
    linear = torch.where(abs_diff >= delta_tensor, abs_diff - 0.5 * delta_tensor, torch.zeros_like(abs_diff))
    loss = quadratic + linear
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


