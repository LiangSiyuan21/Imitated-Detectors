import torch

class NLL_OHEM(torch.nn.NLLLoss):
    """ Online hard example mining.
    Needs input from nn.LogSotmax() """

    def __init__(self, ratio):
        super(NLL_OHEM, self).__init__(None, True)
        self.ratio = ratio

    def forward(self, x, y, ratio=None):
        if ratio is not None:
            self.ratio = ratio
        num_inst = x.size(0)
        num_hns = int(self.ratio * num_inst)
        x_ = x.clone()
        inst_losses = torch.zeros(num_inst)
        for idx, label in enumerate(y.data):
            inst_losses[idx] = -x_.data[idx, label]
            # loss_incs = -x_.sum(1)
        _, idxs = inst_losses.topk(num_hns)
        x_hn = x.index_select(0, idxs)
        y_hn = y.index_select(0, idxs)
        return torch.nn.functional.nll_loss(x_hn, y_hn)

def hard_example_mining(dist_mat, labels, return_inds=False):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
            .copy_(torch.arange(0, N).long())
            .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an

def hard_aware_point_2_set_mining(dist_mat, labels, weighting='poly', coeff=10):

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t()).type(torch.uint8)
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t()).type(torch.uint8)
        
        # is_pos = is_pos.type_as(torch.uint8)
        # Exclude selfs for positive samples
        device = labels.device
        v = torch.zeros(N).to(device).type(is_pos.dtype)
        mask = torch.diag(torch.ones_like(v)).to(device).type(is_pos.dtype)

        is_pos = mask * torch.diag(v) + (1. - mask) * is_pos

        # `dist_ap` means distance(anchor, positive)
        dist_ap = dist_mat[is_pos.type(torch.bool)].contiguous().view(N, -1)
        # `dist_an` means distance(anchor, negative)
        dist_an = dist_mat[is_neg.type(torch.bool)].contiguous().view(N, -1)
        # Weighting scheme
        if weighting == 'poly':
            w_ap = torch.pow(dist_ap + 1, coeff)
            w_an = torch.pow(dist_an + 1, -2 * coeff)
        else:
            w_ap = torch.exp(dist_ap / coeff)
            w_an = torch.exp(-dist_an / coeff)

        dist_ap = torch.sum(dist_ap * w_ap, dim=1) / torch.sum(w_ap, dim=1)
        dist_an = torch.sum(dist_an * w_an, dim=1) / torch.sum(w_an, dim=1)
        return dist_ap, dist_an