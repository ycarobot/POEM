"""
Copyright to POEM Authors
built upon on DEYO code.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torchvision
import math
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

class POEMv2(nn.Module):
    """POEM online adapts a model by entropy minimization with entropy and PLPD filtering & reweighting during testing.
    Once POEMed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, args, optimizer, branch_optimizer,steps=1, episodic=False, POEM_margin=0.5*math.log(1000), margin_e0=0.4*math.log(1000)):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.branch_optimizer=branch_optimizer
        self.args = args
        self.steps = steps
        self.episodic = episodic
        # self.model_state, self.optimizer_state,self.branch_optimizer_state,self.branch_state ,self.norm_state,self.head_state= \
        #     copy_model_and_optimizer(self.model, self.optimizer,self.branch_optimizer)
        args.counts = [1e-6,1e-6,1e-6,1e-6]
        args.correct_counts = [0,0,0,0]
        self.ema = None
        self.ema_branch=None
        self.POEM_margin = POEM_margin
        self.margin_e0 = margin_e0

    def forward(self, x, iter_, targets=None, flag=True, group=None):
        if self.episodic:
            self.reset()
        
        if targets is None:
            for _ in range(self.steps):
                if flag:
                    outputs, backward, final_backward,ema,ema_branch,reset_flag,reset_flag_branch = forward_and_adapt_POEMv2(x, iter_, self.model, self.args,
                                                                              self.optimizer,self.branch_optimizer,self.ema, self.ema_branch,self.POEM_margin,
                                                                              self.margin_e0, targets, flag, group)
                    # if reset_flag:
                    #     self.reset()
                        
                    # if reset_flag_branch:
                    #     self.resetbranch()
 
                else:
                    outputs = forward_and_adapt_POEMv2(x, iter_, self.model, self.args,
                                                    self.optimizer,self.branch_optimizer, self.POEM_margin,
                                                    self.margin_e0, targets, flag, group)
        else:
            for _ in range(self.steps):
                if flag:
                    outputs, backward, final_backward, corr_pl_1, corr_pl_2 = forward_and_adapt_POEMv2(x, iter_, self.model, 
                                                                                                    self.args, 
                                                                                                    self.optimizer,self.branch_optimizer, 
                                                                                                    self.POEM_margin,
                                                                                                    self.margin_e0,
                                                                                                    targets, flag, group)
                else:
                    outputs = forward_and_adapt_POEMv2(x, iter_, self.model, 
                                                    self.args, self.optimizer,self.branch_optimizer, 
                                                    self.POEM_margin,
                                                    self.margin_e0,
                                                    targets, flag, group, self)
        
        if targets is None:
            if flag:
                return outputs, backward, final_backward,0,0
            else:
                return outputs
        else:
            if flag:
                return outputs, backward, final_backward, corr_pl_1, corr_pl_2
            else:
                return outputs



    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,self.branch_optimizer,
                                 self.model_state, self.optimizer_state,self.branch_optimizer_state)
        self.ema = None
        
    def resetbranch(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_branch_and_optimizer(self.model,self.branch_state,self.branch_optimizer,self.branch_optimizer_state,self.state_dict,self.head_state)
        self.ema = None

def copy_model_and_optimizer(model, optimizer,branch_optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    branch_optimizer_state=deepcopy(branch_optimizer.state_dict())
    branch_state=deepcopy(model.stagebranch.state_dict())
    norm_state=deepcopy(model.normbranch.state_dict())
    head_state=deepcopy(model.headbranch.state_dict())
    return model_state, optimizer_state,branch_optimizer_state,branch_state,norm_state,head_state

def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data
        
@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    #temprature = 1.1 #0.9 #1.2
    #x = x ** temprature #torch.unsqueeze(temprature, dim=-1)
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def branch_update_first(branch_optimizer,filter_branch,outputs_branch,features):
    #criterion=nn.NLLLoss()
    criterion=nn.CrossEntropyLoss()
    select_Data=features[filter_branch]
    select_output=outputs_branch[filter_branch]
    # To gain the probabilities
    probabilities = torch.nn.functional.softmax(select_output, dim=1) 
    # To generate pseudo-labels
    _, pseudo_labels = torch.max(probabilities, dim=1)
    loss=criterion(select_output,pseudo_labels)
    #loss=criterion(probabilities,pseudo_labels)
    loss.backward()
    branch_optimizer.step()
    branch_optimizer.zero_grad()

def branch_update_second(branch_optimizer,filter_branch,outputs_branch,features):
    #criterion=nn.NLLLoss()
    criterion=nn.CrossEntropyLoss()
    #select_Data=features[filter_branch]
    select_output=outputs_branch[filter_branch]
    # To gain the probabilities
    probabilities = torch.nn.functional.softmax(select_output, dim=1) 
    # Generate pseudo-labels
    _, pseudo_labels = torch.max(probabilities, dim=1)
    loss=criterion(select_output,pseudo_labels)
    #loss=criterion(probabilities,pseudo_labels)
    return loss 

def plpd_filter(args,x,outputs,filter_ids_1,model,entropys):
    x_prime = x[filter_ids_1]
    x_prime = x_prime.detach()
    if args.aug_type=='occ':
        first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
        final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
        occlusion_window = final_mean.expand(-1, -1, args.occlusion_size, args.occlusion_size)
        x_prime[:, :, args.row_start:args.row_start+args.occlusion_size,args.column_start:args.column_start+args.occlusion_size] = occlusion_window
    elif args.aug_type=='patch':
        resize_t = torchvision.transforms.Resize(((x.shape[-1]//args.patch_len)*args.patch_len,(x.shape[-1]//args.patch_len)*args.patch_len))
        resize_o = torchvision.transforms.Resize((x.shape[-1],x.shape[-1]))
        x_prime = resize_t(x_prime)
        x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=args.patch_len, ps2=args.patch_len)
        perm_idx = torch.argsort(torch.rand(x_prime.shape[0],x_prime.shape[1]), dim=-1)
        x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1),perm_idx]
        x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=args.patch_len, ps2=args.patch_len)
        x_prime = resize_o(x_prime)
    elif args.aug_type=='pixel':
        x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
        x_prime = x_prime[:,:,torch.randperm(x_prime.shape[-1])]
        x_prime = rearrange(x_prime, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=x.shape[-1], ps2=x.shape[-1])
    with torch.no_grad():
        outputs_prime,outputs_branch_nograd,feature = model(x_prime)
    
    prob_outputs = outputs[filter_ids_1].softmax(1)
    mixoutputs_prime=0.5*outputs_prime+0.5*outputs_branch_nograd
    prob_outputs_prime = mixoutputs_prime.softmax(1)

    cls1 = prob_outputs.argmax(dim=1)

    plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1,1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1,1))
    plpd = plpd.reshape(-1)
    
    if args.filter_plpd:
        filter_ids_2 = torch.where(plpd > args.plpd_threshold)
    else:
        filter_ids_2 = torch.where(plpd >= -2.0)
    entropys = entropys[filter_ids_2]
    return entropys,filter_ids_2


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_POEMv2(x, iter_, model, args, optimizer,branch_optimizer,ema,ema_branch, POEM_margin, margin, targets=None, flag=True, group=None):
    """Forward and adapt model input data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    a=0.5
    reset_flag = False
    reset_flag_branch=False
    branch_margin=POEM_margin

    'Forward'
    outputs,outputs_branch,feature = model(x)
    mix_outputs=a*outputs+(1-a)*outputs_branch
    optimizer.zero_grad()
    branch_optimizer.zero_grad()
    'entropy cal'
    entropys = softmax_entropy(mix_outputs)
    # entropys = softmax_entropy(outputs)

    'entropy filter'
    if args.filter_ent:
        filter_ids_1 = torch.where((entropys < POEM_margin))
    else:    
        filter_ids_1 = torch.where((entropys <= math.log(1000)))
    entropys = entropys[filter_ids_1]
    backward = len(entropys)

    
    if backward==0:
        loss = softmax_entropy(mix_outputs).mean(0)
        # loss = softmax_entropy(outputs).mean(0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if targets is not None:
            return mix_outputs,0, 0,ema,ema_branch,reset_flag,reset_flag_branch
        return mix_outputs, 0, 0,ema,ema_branch,reset_flag,reset_flag_branch

    'PLPD filter(DEYO)'
    x_prime = x[filter_ids_1]
    x_prime = x_prime.detach()
    if args.aug_type=='occ':
        first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
        final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
        occlusion_window = final_mean.expand(-1, -1, args.occlusion_size, args.occlusion_size)
        x_prime[:, :, args.row_start:args.row_start+args.occlusion_size,args.column_start:args.column_start+args.occlusion_size] = occlusion_window
    elif args.aug_type=='patch':
        resize_t = torchvision.transforms.Resize(((x.shape[-1]//args.patch_len)*args.patch_len,(x.shape[-1]//args.patch_len)*args.patch_len))
        resize_o = torchvision.transforms.Resize((x.shape[-1],x.shape[-1]))
        x_prime = resize_t(x_prime)
        x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=args.patch_len, ps2=args.patch_len)
        perm_idx = torch.argsort(torch.rand(x_prime.shape[0],x_prime.shape[1]), dim=-1)
        x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1),perm_idx]
        x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=args.patch_len, ps2=args.patch_len)
        x_prime = resize_o(x_prime)
    elif args.aug_type=='pixel':
        x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
        x_prime = x_prime[:,:,torch.randperm(x_prime.shape[-1])]
        x_prime = rearrange(x_prime, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=x.shape[-1], ps2=x.shape[-1])
    with torch.no_grad():
        outputs_prime,outputs_branch_nograd,feature = model(x_prime)
        mix_outputs1=0.5*outputs_prime+0.5*outputs_branch_nograd
    #prob_outputs = mix_outputs[filter_ids_1].softmax(1)
    prob_outputs = outputs[filter_ids_1].softmax(1)
    prob_outputs_prime = outputs_prime.softmax(1)

    cls1 = prob_outputs.argmax(dim=1)

    plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1,1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1,1))
    plpd = plpd.reshape(-1)
    
    if args.filter_plpd:
        filter_ids_2 = torch.where(plpd > args.plpd_threshold)
    else:
        filter_ids_2 = torch.where(plpd >= -2.0)
    entropys = entropys[filter_ids_2]
    final_backward = len(entropys)
    
    if targets is not None:
        corr_pl_1 = (targets[filter_ids_1] == prob_outputs.argmax(dim=1)).sum().item()
        
    if final_backward==0:
        del x_prime
        del plpd
        
        if targets is not None:
            return mix_outputs, 0, 0,ema,ema_branch,reset_flag,reset_flag_branch
        return mix_outputs, 0, 0,ema,ema_branch,reset_flag,reset_flag_branch
        
    plpd = plpd[filter_ids_2]
    
    if targets is not None:
        corr_pl_2 = (targets[filter_ids_1][filter_ids_2] == prob_outputs[filter_ids_2].argmax(dim=1)).sum().item()

    if args.reweight_ent or args.reweight_plpd:
        coeff = (args.reweight_ent * (1 / (torch.exp(((entropys.clone().detach()) - margin)))) +
                 args.reweight_plpd * (1 / (torch.exp(-1. * plpd.clone().detach())))
                )            
        entropys = entropys.mul(coeff)
        
    loss = entropys.mean(0)
    
    if not np.isnan(loss.item()):
        ema = update_ema(ema, loss.item())
    if ema is not None:
        if ema < 0.2:
            #print("ema < 0.2, now reset the model")
            reset_flag = True
            return outputs,0, 0,ema,ema_branch,reset_flag,reset_flag_branch
        
    'shallow layers update(entropy mini)'
    if final_backward != 0:
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()
    del x_prime
    del plpd
    
    'Forward to avoid graph local change'        
    outputs,outputs_branch,feature = model(x)
    mix_outputs=0.5*outputs+0.5*outputs_branch
    entropys1 = softmax_entropy(mix_outputs)
 
    filter_ids_1 = torch.where((entropys1 < margin))
    if len(filter_ids_1[0])==0:
        return outputs,0, 0,ema,ema_branch,reset_flag,reset_flag_branch
    
    plpd,filter_ids_2=plpd_filter(args,x,outputs,filter_ids_1,model,entropys1)
    #plpd = plpd[filter_ids_2[0]]
    finalfilter=filter_ids_1[0][filter_ids_2[0]]

    'Branch update'  
    if len(finalfilter)!=0:
        branch_update_first(branch_optimizer,finalfilter,outputs_branch,feature)
    else:
        return outputs,0, 0,ema,ema_branch,reset_flag,reset_flag_branch
        
    filter_ids_before=[[]]
    filter_ids_after=finalfilter
    #iteration times
    for i in range(1):
        filter_ids_before=filter_ids_after
        'Forward to search if PRS exist'
        outputs_third,outputs_branch_third,feature3=model(x)
        mix_outputs=0.5*outputs_third+0.5*outputs_branch_third
        entropys1 = softmax_entropy(mix_outputs)
        filter_ids_1 = torch.where(entropys1 < margin)[0]
        if len(filter_ids_1)==0:
            return outputs,0, 0,ema,ema_branch,reset_flag,reset_flag_branch
        entropys,filter_ids_2=plpd_filter(args,x,outputs,filter_ids_1,model,entropys1)
        finalfilter=filter_ids_1[filter_ids_2[0]]
        
        # filter_ids_3=filter_ids_1[filter_ids_2]

        'when using compare the length is same as issame method in tentPOEM'
        if len(filter_ids_before)!=len(finalfilter):
            if len(finalfilter) != 0 :
                entropys1=entropys1[finalfilter]

                loss = entropys1.mean(0)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                filter_ids_after=finalfilter
                break

            outputs_third,outputs_branch_third,feature3=model(x)
            mix_outputs=0.5*outputs_third+0.5*outputs_branch_third
            entropys1 = softmax_entropy(mix_outputs)
            filter_ids_1 = torch.where(entropys1 < margin)[0]
            if len(filter_ids_1)==0:
                return outputs,0, 0,ema,ema_branch,reset_flag,reset_flag_branch
            entropys,filter_ids_2=plpd_filter(args,x,mix_outputs,filter_ids_1,model,entropys1)
            finalfilter=filter_ids_1[filter_ids_2[0]]
            
            # filter_ids_2=filter_ids_1[filter_ids_2]
            if len(finalfilter) != 0 :
                loss_branch=branch_update_second(branch_optimizer,finalfilter,mix_outputs,feature3)
                loss_branch.backward()
                branch_optimizer.step()
                branch_optimizer.zero_grad()
                
            else:
                filter_ids_after=filter_ids_2
                break
                       
            filter_ids_after=filter_ids_2
        
        else:
            break    
    
    outputs_third,outputs_branch_third,feature3=model(x)
    mixoutputs3=0.5*outputs_third+0.5*outputs_branch_third
    
    if targets is not None:
        outputs_third,outputs_branch_third,feature3=model(x)
        mixoutputs3=a*outputs_third+(1-a)*outputs_branch_third
        return mixoutputs3, 0, 0,ema,ema_branch,reset_flag,reset_flag_branch
    
    return mixoutputs3, 0, 0,ema,ema_branch,reset_flag,reset_flag_branch


def collect_params(model):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if'stages.3'in nm:
            continue
        if'stagebranch'in nm:
            continue
        if'normbranch'in nm:
            continue
        if'headbranch'in nm:
            continue
        if 'norm.' in nm:
            continue
        if'model.norm'in nm:
            continue
        if'blockbranch'in nm:
            continue
        if'normbranch'in nm:
            continue
        if'branch'in nm:
            continue
        if nm in ['norm']:
            continue

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names


def load_branch_and_optimizer(model,branch_state ,branch_optimizer,branch_optimizer_state,norm_state,head_state):
    """Restore the model and optimizer states from copies."""
    model.stagebranch.load_state_dict(branch_state, strict=True)
    branch_optimizer.load_state_dict(branch_optimizer_state)

    model.normbranch.load_state_dict(norm_state, strict=True)
    model.headbranch.load_state_dict(head_state, strict=True)

def load_model_and_optimizer(model, optimizer, branch_optimizer,model_state, optimizer_state,branch_optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    branch_optimizer.load_state_dict(branch_optimizer_state)

def configure_model(model):
    """Configure model for use with POEM."""
    # train mode, because POEM optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what POEM updates
    model.requires_grad_(False)
    # configure norm for POEM updates: enable grad + force batch statisics (this only for BN models)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model

