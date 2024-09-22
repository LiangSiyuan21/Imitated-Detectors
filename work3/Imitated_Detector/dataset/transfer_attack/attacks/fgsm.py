import torch
import torch.nn as nn

from ..attack import Attack


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]
    
    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (DEFAULT: 0.007)
    
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.FGSM(model, eps=0.007)
        >>> adv_images = attack(images, labels)
        
    """
    def __init__(self, model, eps=0.007, ub=1, lb=0):
        super(FGSM, self).__init__("FGSM", model)
        self.eps = eps * (ub-lb)

    def forward(self, data, gts):
        r"""
        Overridden.
        """
        images = data['img'][0].clone().detach().to(self.device)
        # labels = labels.clone().detach().to(self.device)
        # labels = self._transform_label(images, labels)
        new_data = {}
        adv_images = images.clone().detach()
        new_data['img_metas'] = data['img_metas'][0]      
        adv_images.requires_grad = True
        data['img'][0] = adv_images
    
        new_data['img'] = adv_images
        losses = self.model(return_loss=True, **new_data, gt_bboxes=gts[0].unsqueeze(0), gt_labels=gts[1].unsqueeze(0))

        self.model.zero_grad()
        losses['loss_cls'] = losses['loss_cls'] * (-1.0)
        losses['loss_cls'].backward()
        grad = adv_images.grad.data

        adv_images = images - self.eps*grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
