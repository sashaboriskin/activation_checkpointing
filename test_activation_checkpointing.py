import pytest
import torch
import torch.nn.functional as F
from model import Model, ModelCheckpoint


class State:
    def __init__(self):
        self.device = torch.device("cuda")
        self.B = 1
        self.L = 512
        
        self.model_baseline = Model(
            in_dim=256,
            hidden_dim=256, 
            ff_dim=512,
            num_layers=2,
            head_dim=64,
        ).to(self.device).to(torch.bfloat16)
        
        self.model_checkpoint = ModelCheckpoint(
            in_dim=256,
            hidden_dim=256, 
            ff_dim=512,
            num_layers=2,
            head_dim=64,
            use_rng_state=True,
        ).to(self.device).to(torch.bfloat16)
        
        self.initial_state_dict = self.model_baseline.state_dict()
        self.model_checkpoint.load_state_dict(self.initial_state_dict)
        
        self.optimizer_baseline = torch.optim.AdamW(self.model_baseline.parameters(), lr=1e-4)
        self.optimizer_checkpoint = torch.optim.AdamW(self.model_checkpoint.parameters(), lr=1e-4)
        
        self.cu = torch.arange(0, (self.B + 1) * self.L, self.L, dtype=torch.int32, device=self.device)
        self.x = torch.randn(self.B * self.L, 256, device=self.device, dtype=torch.bfloat16) 
        
        self.optimizer_baseline.zero_grad()
        self.output_baseline = self.model_baseline(self.x, self.cu)
        self.loss_baseline = F.mse_loss(self.output_baseline, self.x.detach())
        self.loss_baseline.backward()
        self.optimizer_baseline.step()
        
        self.optimizer_checkpoint.zero_grad()
        self.output_checkpoint = self.model_checkpoint(self.x, self.cu)
        self.loss_checkpoint = F.mse_loss(self.output_checkpoint, self.x.detach())
        self.loss_checkpoint.backward()
        self.optimizer_checkpoint.step()

        self.gradients_baseline = {}
        self.gradients_checkpoint = {}
        
        for name, param in self.model_baseline.named_parameters():
            if param.grad is not None:
                self.gradients_baseline[name] = param.grad.clone()
        
        for name, param in self.model_checkpoint.named_parameters():
            if param.grad is not None:
                self.gradients_checkpoint[name] = param.grad.clone()
        
        self.final_weights_baseline = {
            name: param.clone() for name, param in self.model_baseline.named_parameters()
        }
        self.final_weights_checkpoint = {
            name: param.clone() for name, param in self.model_checkpoint.named_parameters()
        }


@pytest.fixture(scope="session")
def test_state():
    state = State()
    return state
    
def test_model_outputs_identical(test_state):
    assert torch.allclose(test_state.output_baseline, test_state.output_checkpoint, rtol=1e-3, atol=1e-4)

def test_loss_identical(test_state):
    assert torch.allclose(test_state.loss_baseline, test_state.loss_checkpoint, rtol=1e-3, atol=1e-4)

def test_model_gradients_identical(test_state):
    all_grads_close = True
    
    for name in test_state.gradients_baseline.keys():
        if name in test_state.gradients_checkpoint:
            grad_b = test_state.gradients_baseline[name]
            grad_c = test_state.gradients_checkpoint[name]
            
            is_close = torch.allclose(grad_b, grad_c, rtol=1e-3, atol=1e-4)
            all_grads_close = all_grads_close and is_close
    
    assert all_grads_close
