"""Check offload preserves the double derivatives needed for force fitting."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from modules.module_c_mlips.umof_mace_train import cpu_offloaded_step


def force_step(*, device):
    positions = torch.linspace(-1, 1, 24, dtype=torch.float64, device=device).reshape(8, 3)
    positions.requires_grad_()
    weight = torch.tensor([0.3, -0.7, 1.1], dtype=torch.float64, device=device, requires_grad=True)
    energy = torch.sin(positions @ weight).square().sum()
    forces = -torch.autograd.grad(energy, positions, create_graph=True)[0]
    loss = energy.square() + (forces - 0.2).square().sum()
    loss.backward()
    return loss.detach().cpu(), weight.grad.detach().cpu(), forces.detach().cpu()


class CpuOffloadTests(unittest.TestCase):
    def test_small_graph_fast_path_keeps_large_or_dense_graphs_offloaded(self):
        def fake_step(**kwargs):
            return 'step'
        step = cpu_offloaded_step(fake_step, gpu_only_max_edges=8000)
        with patch('torch.autograd.graph.save_on_cpu') as hooks:
            small = SimpleNamespace(num_nodes=100, edge_index=torch.empty((2, 7000)))
            self.assertEqual(step(device='cuda', batch=small), 'step')
            hooks.assert_not_called()
            large = SimpleNamespace(num_nodes=400, edge_index=torch.empty((2, 7000)))
            step(device='cuda', batch=large)
            dense = SimpleNamespace(num_nodes=100, edge_index=torch.empty((2, 9000)))
            step(device='cuda', batch=dense)
            self.assertEqual(hooks.call_count, 2)

    def test_cpu_training_does_not_use_offload_hooks(self):
        expected = force_step(device='cpu')
        with patch('torch.autograd.graph.save_on_cpu') as hooks:
            actual = cpu_offloaded_step(force_step)(device='cpu')
        hooks.assert_not_called()
        for reference, value in zip(expected, actual):
            torch.testing.assert_close(reference, value, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for CPU offload')
    def test_cuda_force_loss_and_parameter_gradients_are_preserved(self):
        expected = force_step(device='cuda')
        with patch('torch.autograd.graph.save_on_cpu', wraps=torch.autograd.graph.save_on_cpu) as hooks:
            actual = cpu_offloaded_step(force_step)(device='cuda')
        hooks.assert_called_once_with(pin_memory=False)
        for reference, value in zip(expected, actual):
            torch.testing.assert_close(reference, value, rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
