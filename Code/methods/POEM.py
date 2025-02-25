"""
Copyright to POEM Authors
built upon on DEYO code.
"""

from copy import deepcopy
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.jit
import torchvision
import math
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

class POEM(nn.Module):
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
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        args.counts = [1e-6,1e-6,1e-6,1e-6]
        args.correct_counts = [0,0,0,0]
        self.ema = None
        self.POEM_margin = POEM_margin
        self.margin_e0 = margin_e0
        self.current_model_probs = None

    def forward(self, x, iter_, targets=None, flag=True, group=None):
        if self.episodic:
            self.reset()
        
        if targets is None:
            for _ in range(self.steps):
                if flag:
                    outputs, backward, final_backward,ema,reset_flag,updated_probs = forward_and_adapt_POEM(x, iter_, self.model, self.args,
                                                                              self.optimizer,self.branch_optimizer,self.current_model_probs,self.ema, self.POEM_margin,
                                                                              self.margin_e0, targets, flag, group)
                    self.reset_model_probs(updated_probs)
                    if reset_flag:
                        self.reset()
                        self.ema = ema
                
                else:
                    outputs = forward_and_adapt_POEM(x, iter_, self.model, self.args,
                                                    self.optimizer,self.branch_optimizer, self.POEM_margin,
                                                    self.margin_e0, targets, flag, group)
        else:
            for _ in range(self.steps):
                if flag:
                    outputs, backward, final_backward, corr_pl_1, corr_pl_2 = forward_and_adapt_POEM(x, iter_, self.model, 
                                                                                                    self.args, 
                                                                                                    self.optimizer,self.branch_optimizer, 
                                                                                                    self.POEM_margin,
                                                                                                    self.margin_e0,
                                                                                                    targets, flag, group)
                else:
                    outputs = forward_and_adapt_POEM(x, iter_, self.model, 
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


    def reset_model_probs(self, probs):
        self.current_model_probs = probs

    

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

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
    criterion=nn.NLLLoss()
    #criterion=nn.CrossEntropyLoss()
    select_Data=features[filter_branch]
    select_output=outputs_branch[filter_branch]
    # To gain the probabilities
    probabilities = torch.nn.functional.softmax(select_output, dim=1) 
    # To generate pseudo-labels.
    _, pseudo_labels = torch.max(probabilities, dim=1)
    #loss=criterion(select_output,pseudo_labels)
    loss=criterion(probabilities,pseudo_labels)
    loss.backward()
    branch_optimizer.step()

def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)
            
def branch_update_second(branch_optimizer,filter_branch,outputs_branch,features):
    criterion=nn.NLLLoss()
    #criterion=nn.CrossEntropyLoss()
    #select_Data=features[filter_branch]
    select_output=outputs_branch[filter_branch]
    #  
    probabilities = torch.nn.functional.softmax(select_output, dim=1) 
    # To generate pseudo-labels
    _, pseudo_labels = torch.max(probabilities, dim=1)
    #loss=criterion(select_output,pseudo_labels)
    loss=criterion(probabilities,pseudo_labels)
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
    return entropys,filter_ids_2


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_POEM(x, iter_, model, args, optimizer,branch_optimizer,current_model_probs,ema, POEM_margin, margin, targets=None, flag=True, group=None):
    """Forward and adapt model input data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    reset_flag = False
    branch_margin=margin
    outputs,outputs_branch,feature = model(x)
    mix_outputs=0.5*outputs+0.5*outputs_branch
    optimizer.zero_grad()
    entropys = softmax_entropy(mix_outputs)
    if args.filter_ent:
        filter_ids_1 = torch.where((entropys < margin))
    else:    
        filter_ids_1 = torch.where((entropys <= math.log(1000)))
    entropys = entropys[filter_ids_1]
    backward = len(entropys)
    if backward==0:
        # loss = softmax_entropy(mix_outputs).mean(0)
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()
        updated_probs=current_model_probs
        return mix_outputs, backward, 0,ema,reset_flag,updated_probs


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
    prob_outputs_prime = outputs_prime.softmax(1)

    cls1 = prob_outputs.argmax(dim=1)

    plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1,1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1,1))
    plpd = plpd.reshape(-1)
    
    if args.filter_plpd:
        filter_ids_2 = torch.where(plpd > args.plpd_threshold)
    else:
        filter_ids_2 = torch.where(plpd >= -2.0)

    
    entropys = entropys[filter_ids_2[0]]
    filter_ids_3=filter_ids_1[0][filter_ids_2[0]]
    finalfilter=filter_ids_3
    filter_ids_4=None
    if current_model_probs is not None: 
        cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0), mix_outputs[finalfilter].softmax(1), dim=1)
        filter_ids_4 = torch.where(torch.abs(cosine_similarities) < 0.2)[0]  
        finalfilter=finalfilter[filter_ids_4]
        updated_probs = update_model_probs(current_model_probs, mix_outputs[finalfilter].softmax(1))
    else:
        updated_probs = update_model_probs(current_model_probs, mix_outputs[finalfilter].softmax(1))

    final_backward = len(finalfilter)

    
    if targets is not None:
        corr_pl_1 = (targets[filter_ids_1] == prob_outputs.argmax(dim=1)).sum().item()
        
    if final_backward==0:
        del x_prime
        del plpd
        
        if targets is not None:
            return outputs, backward, 0, 0, 0
        return mix_outputs, backward, 0,ema,reset_flag,updated_probs

    
    if filter_ids_4 is not None:
        plpd = plpd[filter_ids_2[0]][filter_ids_4]
        entropys = entropys[filter_ids_4]
    else:
        plpd = plpd[filter_ids_2[0]]
    
   
    if args.reweight_ent or args.reweight_plpd:
        coeff = (args.reweight_ent * (1 / (torch.exp(((entropys.clone().detach()) - margin)))) +
                 args.reweight_plpd * (1 / (torch.exp(-1. * plpd.clone().detach())))
                )            
        entropys = entropys.mul(coeff)
        
    loss = entropys.mean(0)
    
    if not np.isnan(loss.item()):
        ema = update_ema(ema, loss.item())
    if final_backward != 0:
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()
    del x_prime
    del plpd
    if ema is not None:
        if ema < 0.2:
            #print("ema < 0.2, now reset the model")
            reset_flag = True
            return mix_outputs, 0, 0,ema,reset_flag,updated_probs
            
    outputs,outputs_branch,feature = model(x)
    mix_outputs=0.5*outputs+0.5*outputs_branch
    entropys1 = softmax_entropy(mix_outputs)
    filter_ids_1 = torch.where((entropys1 < margin))
    if len(filter_ids_1[0])==0:
        return mix_outputs, backward, 0,ema,reset_flag,updated_probs
    
    plpd,filter_ids_2=plpd_filter(args,x,outputs,filter_ids_1,model,entropys)
    #plpd = plpd[filter_ids_2[0]]
    filter_ids_3=filter_ids_1[0][filter_ids_2[0]]
    finalfilter=filter_ids_3
    filter_ids_4=None
    if current_model_probs is not None: 
        cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0), mix_outputs[finalfilter].softmax(1), dim=1)
        filter_ids_4 = torch.where(torch.abs(cosine_similarities) < 0.2)[0]  
        finalfilter=finalfilter[filter_ids_4]
        updated_probs = update_model_probs(current_model_probs, mix_outputs[finalfilter].softmax(1))
    else:
        updated_probs = update_model_probs(current_model_probs, mix_outputs[finalfilter].softmax(1))
        return mix_outputs, backward, 0,ema,reset_flag,updated_probs

    
    if len(finalfilter)!=0:
        branch_update_first(branch_optimizer,finalfilter,outputs_branch,feature)
    else:
        return mix_outputs, backward, 0,ema,reset_flag,updated_probs
        
    filter_ids_before=[[]]
    filter_ids_after=finalfilter
    #iteration times
    for i in range(1):
        filter_ids_before=filter_ids_after
        outputs_third,outputs_branch_third,feature3=model(x)
        mix_outputs=0.5*outputs_third+0.5*outputs_branch_third
        entropys1 = softmax_entropy(mix_outputs)
        filter_ids_1 = torch.where(entropys1 < margin)[0]
        if len(filter_ids_1)==0:
            return mix_outputs, backward, 0,ema,reset_flag,updated_probs
        
        #plpd filter
        entropys,filter_ids_2=plpd_filter(args,x,outputs,filter_ids_1,model,entropys1)
        finalfilter=filter_ids_1[filter_ids_2[0]]

        #redundant filter
        cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0), mix_outputs[finalfilter].softmax(1), dim=1)
        filter_ids_4 = torch.where(torch.abs(cosine_similarities) < 0.2)[0]  
        finalfilter=finalfilter[filter_ids_4]
        updated_probs = update_model_probs(current_model_probs, mix_outputs[finalfilter].softmax(1))
        # filter_ids_3=filter_ids_1[filter_ids_2]

        #when using compare the length is same as issame method in tentPOEM
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
                return mixoutputs3, backward, 0,ema,reset_flag,updated_probs
            entropys,filter_ids_2=plpd_filter(args,x,mix_outputs,filter_ids_1,model,entropys1)
            finalfilter=filter_ids_1[filter_ids_2[0]]
            cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0), outputs_third[finalfilter].softmax(1), dim=1)
            filter_ids_4 = torch.where(torch.abs(cosine_similarities) < 0.2)[0]  
            finalfilter=finalfilter[filter_ids_4]
            updated_probs = update_model_probs(current_model_probs, mix_outputs[finalfilter].softmax(1))
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
    
    #outputs
    outputs_third,outputs_branch_third,feature3=model(x)
    mixoutputs3=0.5*outputs_third+0.5*outputs_branch_third
    
    if targets is not None:
        outputs_third,outputs_branch_third,feature3=model(x)
        mixoutputs3=0.5*outputs_third+0.5*outputs_branch_third
        return mix_outputs, backward, 0,ema,reset_flag,updated_probs
    
    return mixoutputs3, backward, 0,ema,reset_flag,updated_probs

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


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


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
        # Assuming the branch is an attribute of the model, enable gradients for it
        # if hasattr(model, 'branch'):
        #     branch = model.branch
        #     for param in branch.parameters():
        #         param.requires_grad = True
    return model

