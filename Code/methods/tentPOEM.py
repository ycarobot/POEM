# https://github.com/DequanWang/tent/blob/master/tent.py

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import math
import numpy as np

def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data

class tentPOEM(nn.Module):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, branchoptimizer,steps=1, episodic=False,reset_constant_em=0.2):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.branchoptimizer=branchoptimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.reset_constant_em = reset_constant_em  # threshold e_m for model recovery scheme
        self.ema = None  # to record the moving average of model output entropy, as model recovery criteria

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state ,self.branchoptimizer_state= \
            copy_model_and_optimizer(self.model, self.optimizer,self.branchoptimizer)


    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer,self.branchoptimizer, self.reset_constant_em, self.ema)
            # if reset_flag:
            #     self.reset()
            # self.ema = ema  # update moving average value of loss
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,self.branchoptimizer,
                                 self.model_state, self.optimizer_state,self.branchoptimizer_state)
        self.ema = None

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def branch_update_first(branch_optimizer,filter_branch,outputs_branch,features):
    #criterion=nn.NLLLoss()
    criterion=nn.CrossEntropyLoss()
    select_Data=features[filter_branch]
    select_output=outputs_branch[filter_branch]
    # To gain the probabilities
    probabilities = torch.nn.functional.softmax(select_output, dim=1) 
    # Generate pseudo-labels
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


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer,branch_optimizer, reset_constant, ema):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    a=0.5
    branch_optimizer.zero_grad()
    optimizer.zero_grad()
    branch_margin=math.log(1000)*0.40
    margin=math.log(1000)*0.40
    
    # forward
    outputs,outputs_branch,feature = model(x)
    mix_outputs=a*outputs+(1-a)*outputs_branch
    # adapt
    entropys = softmax_entropy(mix_outputs)
    filter_ids = torch.where(entropys < margin)[0]
    loss=entropys[filter_ids].mean(0)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    
    outputs,outputs_branch,feature = model(x)
    mix_outputs=a*outputs+(1-a)*outputs_branch
    entropys1 = softmax_entropy(mix_outputs)
    filter_ids_1 = torch.where(entropys1 < margin)[0]
    if len(filter_ids_1)> (len(x) //10) :
        branch_update_first(branch_optimizer,filter_ids_1,outputs_branch,feature)
    else:
        return mix_outputs
    
    filter_ids_before=filter_ids
    filter_ids_after=filter_ids_1
    
    for i in range(2):
        
        if not issame(filter_ids_before, filter_ids_after):
            if len(filter_ids_1) != 0 :
                outputs,outputs_branch,feature = model(x)
                mix_outputs=a*outputs+(1-a)*outputs_branch
                # adapt
                entropys = softmax_entropy(mix_outputs)
                filter_ids = torch.where((entropys < margin))[0]
                loss=entropys[filter_ids].mean(0)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                filter_ids_after=filter_ids_1
                
                #Update the Adapt Branch network
                outputs_third,outputs_branch_third,feature3=model(x)
                mix_outputs=a*outputs_third+(1-a)*outputs_branch_third
                entropys1 = softmax_entropy(mix_outputs)
                filter_ids_1 = torch.where(entropys1 < margin)[0]
                if len(filter_ids_1) != 0 :
                    loss_branch=branch_update_second(branch_optimizer,filter_ids_1,mix_outputs,feature3)
                    loss_branch.backward()
                    branch_optimizer.step()
                    branch_optimizer.zero_grad()
                    filter_ids_after=filter_ids_1
                else:
                    break
            else:
                break
        
        else:
            break    
        
    # outputs_third,outputs_branch_third,feature3=model(x)
    # mix_outputs=a*outputs_third+(1-a)*outputs_branch_third
   
              
    return mix_outputs

def issame(tensor1, tensor2):
    """
    Compare two tensors to check if they contain the same elements, ignoring order.
    Args:
    - tensor1 (torch.Tensor): The first tensor to compare.
    - tensor2 (torch.Tensor): The second tensor to compare.
    
    Returns:
    - bool: True if tensors contain the same elements, False otherwise.
    """
    # Check if shapes are the same
    if tensor1.shape != tensor2.shape:
        return False
    
    # Sort tensors and then compare
    sorted_tensor1, _ = torch.sort(tensor1)
    sorted_tensor2, _ = torch.sort(tensor2)
    
    return torch.equal(sorted_tensor1, sorted_tensor2)



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



def copy_model_and_optimizer(model, optimizer,branchoptimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    branchoptimizer_state=deepcopy(branchoptimizer.state_dict())
    return model_state, optimizer_state,branchoptimizer_state


def load_model_and_optimizer(model, optimizer, branchoptimizer,model_state, optimizer_state,branchoptimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    branchoptimizer.load_state_dict(branchoptimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"