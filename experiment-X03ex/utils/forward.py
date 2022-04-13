import torch
import torch.nn.functional as F
from utils.criterion import SymKlCriterion

###
# SMART function
###

""" Generate and embed noise with same size variance epsilon
    that conforms to the normal distribution  """
def generate_noise(embed, mask, epsilon=0.01):
    noise = embed.data.new(embed.size()).normal_(0, 1) *  epsilon
    noise.detach()
    noise.requires_grad_()
    return noise

def stable_kl(logit, target, epsilon=1e-6, reduce=True):
    logit = logit.view(-1, logit.size(-1)).float()
    target = target.view(-1, target.size(-1)).float()
    bs = logit.size(0)
    p = F.log_softmax(logit, 1).exp()
    y = F.log_softmax(target, 1).exp()
    rp = -(1.0/(p + epsilon) -1 + epsilon).detach().log()
    ry = -(1.0/(y + epsilon) -1 + epsilon).detach().log()
    if reduce:
        return (p* (rp- ry) * 2).sum() / bs
    else:
        return (p* (rp- ry) * 2).sum()

def norm_grad(grad, eff_grad=None, norm_p='inf', epsilon=0.01, sentence_level=False):
    if norm_p == 'l2':
        if sentence_level:
            direction = grad / (torch.norm(grad, dim=(-2, -1), keepdim=True) + epsilon)
        else:
            direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + epsilon)
    elif norm_p == 'l1':
        direction = grad.sign()
    else:
        if sentence_level:
            direction = grad / (grad.abs().max((-2, -1), keepdim=True)[0] + epsilon)
        else:
            direction = grad / (grad.abs().max(-1, keepdim=True)[0] + epsilon)
            eff_direction = eff_grad / (grad.abs().max(-1, keepdim=True)[0] + epsilon)
    return direction, eff_direction

"""
Some difference between MT-DNN model and BERTfor
------------------------------------------------
MT-DNN uses SANBertNetwork() as the base
forward \
(self, input_ids, token_type_ids, attention_mask, premise_mask=None, hyp_mask=None, task_id=0, fwd_type=0, embed=None)

## if fwd_type == 2:
##     assert embed is not None
##     last_hidden_state, all_hidden_states = self.encode(None, token_type_ids, attention_mask, embed) 
## elif fwd_type == 1:
##     return self.embed_encode(input_ids, token_type_ids, attention_mask)
## else:
##     last_hidden_state, all_hidden_states = self.encode(input_ids, token_type_ids, attention_mask)

IndoNLU uses BertForSequenceClassification() as the base
forward \
(input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None)
"""
def adv_training(model, logits,
                input_ids,
                attention_mask,
                token_type_ids,
                labels,
                step_size=1e-3,
                noise_var=0.01,
                norm_level=False,
                K=1,
                pairwise=1):
    
    """
    P.S.
    logits == classification score before softmax
    step_size --> specify how much you move while trying to go downhill (when doing gradient descent)
    """
    # adv training
    embed = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)[1]
    noise = generate_noise(embed, attention_mask, epsilon=noise_var)
    """
    Debug:
    ```````````````````````````````````````````````````````````````
    embed
    tensor([[ 0.3791, -0.4102,  0.3413],
            [ 0.6053, -0.2459, -0.0767],
            [ 0.0130, -0.1101,  0.0268],
            [ 0.2802, -0.0589,  0.1838],
            [ 0.1338, -0.5715,  0.2976],
            [ 0.1656, -0.1947,  0.1821],
            [ 0.0688,  0.0546, -0.0813],
            [ 0.2794, -0.0862, -0.0063]], grad_fn=<AddmmBackward>)
    ----------------------------
    noise
    tensor([[ 0.0082,  0.0298,  0.0057],
            [ 0.0050, -0.0215,  0.0152],
            [-0.0073, -0.0126,  0.0010],
            [ 0.0052, -0.0010, -0.0162],
            [ 0.0012,  0.0016,  0.0003],
            [ 0.0086,  0.0022,  0.0030],
            [-0.0082,  0.0124, -0.0065],
            [-0.0039,  0.0127, -0.0045]], requires_grad=True)
    ----------------------------
    """
    for step in range(0, K):
        # adv_logits is already in tensor form, so there's no need to force it into the model
        adv_logits = embed + noise
        adv_loss = stable_kl(adv_logits, logits.detach(), reduce=False)
        delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True, retain_graph=False)
        norm = delta_grad.norm()
        """
        Debug:
        ```````````````````````````````````````````````````````````````
        adv_logits
        tensor([[ 0.3873, -0.3804,  0.3470],
                [ 0.6103, -0.2673, -0.0615],
                [ 0.0057, -0.1227,  0.0278],
                [ 0.2854, -0.0599,  0.1677],
                [ 0.1351, -0.5699,  0.2978],
                [ 0.1743, -0.1925,  0.1851],
                [ 0.0607,  0.0670, -0.0878],
                [ 0.2755, -0.0735, -0.0109]], grad_fn=<AddBackward0>)
        ----------------------------
        adv_loss
        tensor(0.2333, grad_fn=<SumBackward0>)
        ----------------------------
        delta_grad
        tensor([[ 0.3363, -0.4371,  0.1009],
                [-0.0657, -0.1646,  0.2303],
                [-0.0221, -0.0564,  0.0785],
                [-0.0768,  0.0660,  0.0109],
                [ 0.2763, -0.3721,  0.0958],
                [-0.2051,  0.3138, -0.1087],
                [ 0.3157, -0.2007, -0.1151],
                [ 0.0554,  0.0368, -0.0922]])
        ----------------------------
        norm
        tensor(0.9805)
        ----------------------------
        """   
        if (torch.isnan(norm) or torch.isinf(norm)):
            return 0
        
        eff_delta_grad = delta_grad * step_size
        delta_grad = noise + delta_grad * step_size
        noise, eff_noise = norm_grad(delta_grad, eff_grad=eff_delta_grad, epsilon=0.01, sentence_level=norm_level)
        noise = noise.detach()
        noise.requires_grad_()
    
    # adv_logits is already in tensor form, so there's no need to force it into the model
    adv_logits = embed + noise
    adv_lc = SymKlCriterion()
    adv_loss = adv_lc.forward(logits, adv_logits, ignore_index=-1)
    return adv_loss, embed.detach().abs().mean(), eff_noise.detach().abs().mean()

###
# Forward Function
###

def generate_list(logits, i2w, label):
    # generate prediction & label list
    list_hyp = []
    list_label = []
    hyp = torch.topk(logits, 1)[1]
    for j in range(len(hyp)):
        list_hyp.append(i2w[hyp[j].item()])
        list_label.append(i2w[label[j][0].item()])
    return list_hyp, list_label

""" Forward function for sequence classification """
def forward_model(model=None, batch_data=None, i2w=None, device=None, use_adv_train=True, **kwargs):
    if model is None: raise Exception('No model was given')
    if device is None: raise Exception('No device was given')
    if batch_data is None: raise Exception('No data was given')
    if i2w is None: raise Exception('No index-label was given')
    
    # Unpack batch data
    if len(batch_data) == 3:
        (subword_batch, mask_batch, label_batch) = batch_data
        token_type_batch = None
    elif len(batch_data) == 4:
        (subword_batch, mask_batch, token_type_batch, label_batch) = batch_data
    
    # Prepare input & label
    subword_batch = torch.LongTensor(subword_batch)
    mask_batch = torch.FloatTensor(mask_batch)
    token_type_batch = torch.LongTensor(token_type_batch) if token_type_batch is not None else None
    label_batch = torch.LongTensor(label_batch)
    
    """
    Debug:
    ```````````````````````````````````````````````````````````````
    subword_batch
    tensor([[    2,  8579,  6951,   515,    34,  2040,    90, 17085,  1229,    98,
             18862,   626, 30470,  5314,    92,   515,   724,  1305,    41,  2816,
              1516,  1107,   186,  1305, 30470,   469,    34,   310,   321,  3067,
               951, 30468,   209,    41,  1614,  1614,   119,   166,  2834,   521,
              3346,   321,    26, 29918,   967,  7467,    34,  1004,  1234, 14571,
              5528, 30470,    41,   186,    26,  1436, 26179,  1386,  6234,    41,
               321,  8176,  1057,  2382, 30470,     3],
            [    2,  2124,  7884,  4067,  2965,  2174,  3311, 30477,   259,   388,
               823,    92,   599,  2453,  1974,  3182,  1107,  2279,  9571,  1558,
              5178,  2154, 30470,   209,   119,  3000,    79,    92,     3,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0],
            [    2,  3027,  2486,   186, 30470,  3854,    34,  3318,  1107, 19498,
             30469, 19498,   137,    26,  5652,    32,   377, 30468,  3089,   521,
               988, 13603,     3,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0],
            [    2,  3539, 14313,  5259,    79,   628, 12219,  1107,  4942,  3107,
              2516,   712,  1622,  1107, 30470,   515,    34,  4805,  6240,   295,
             15194,  8161,   536, 30470,   176,  2875,  1137, 30470,     3,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0],
            [    2,  2313,  7835,  1103,  1684,   727,   344,    26,     3,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0],
            [    2,   807,  1634,    34,  5823, 30468,  1532,    92,   259,   176,
              1098,  3043, 12432,  2822,    41,   407,   515,  6710,  6359, 30374,
               232,   295,  2822,    26,  1574,  1098,  3043,  1248, 30470,  1622,
               955,    34,  6358,  1336,   421,   154,   955,  2968, 30470,  6266,
              1107,  6092, 12571,  2822,  7719,  1248, 30470,   405,   955,  4096,
              3303,   469,   786,    79,  1214,   955,    41,  1753,    34,  1591,
             30470,     3,     0,     0,     0,     0],
            [    2,   486,  1095,    43,  2377,   722,  1515,  1622,  1107,   271,
               684,  1214,  3325,    92,   280, 30094,   186,  3514,  3107,    34,
               955,  6673, 16753, 30468, 12659, 30468,   500,    34,  1396,  1063,
              1063,  1522,   137,  3107,     3,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0],
            [    2,  2774,    26, 17188, 30470,   321,   464,  1752,    98,   555,
              9532,     5,    70,   626,  1574,    43,  8579,  6951,   776,   111,
               727,  3449,  1815, 30470,   955,   119,  3107,   405,  3303,   211,
               119, 14170,    79,  1753,    41,  1214,   955, 30470,  1753,  2632,
               684, 30470,   486,   137,   321,  5307,  3210,  5259,   368,  1343,
                90,  8794,    34,  3016,    26, 29608,   638, 10125,   119,   422,
             11475,  1107,    43, 29608, 30470,     3]])
    ----------------------------
    mask_batch
    tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
    ----------------------------
    token_type_batch
    None
    ----------------------------
    label_batch
    tensor([[0],
            [2],
            [2],
            [0],
            [1],
            [0],
            [0],
            [2]])
    ----------------------------
    """
    if device == "cuda":
        subword_batch = subword_batch.cuda()
        mask_batch = mask_batch.cuda()
        token_type_batch = token_type_batch.cuda() if token_type_batch is not None else None
        label_batch = label_batch.cuda()

    # Forward model
    """ 
    BertForSequenceClassification.forward(input_ids=None,         <-- (subword_batch)
                                        attention_mask=None,      <-- (mask_batch)
                                        token_type_ids=None,      <-- (token_type_batch)
                                        position_ids=None,
                                        head_mask=None,
                                        inputs_embeds=None,
                                        labels=None,              <-- (label_batch)
                                        output_attentions=None,
                                        output_hidden_states=None,
                                        return_dict=None)
    """
    outputs = model(subword_batch, attention_mask=mask_batch, token_type_ids=token_type_batch, labels=label_batch)
    loss, logits = outputs[:2]
    
    # adv training
    if use_adv_train:
        adv_alpha = 1
        adv_loss, emb_val, eff_perturb = adv_training(model, logits, subword_batch, mask_batch, token_type_batch, label_batch)
        loss = loss + adv_alpha * adv_loss
        """
        Debug:
        ```````````````````````````````````````````````````````````````
        loss
        tensor(0.9633, grad_fn=<AddBackward0>)
        ----------------------------
        adv_loss
        tensor(0.0099, grad_fn=<MulBackward0>)
        ----------------------------
        loss + adv_alpha * adv_loss
        tensor(0.9732, grad_fn=<AddBackward0>)
        ----------------------------
        """
#   batch_size = batch_data[0].size(0)

    list_hyp, list_label = generate_list(logits, i2w=i2w, label=label_batch)
    
#   return [loss,adv_loss,emb_val,eff_perturb], list_hyp, list_label
    return loss, list_hyp, list_label
