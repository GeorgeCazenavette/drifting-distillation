"""Evaluations of distilled image sets (see each module's docstring for the recipe).

From-scratch ResNet-18 training:
    train_soft     soft labels (alpha 0.9) / hard labels (--alpha 0), ImageNet-1k
    train_kd       EDC knowledge distillation (ImageNet-1k, ImageNette, ImageWoof)
    train_dc       hard labels on ImageNette / ImageWoof (Minimax Diffusion protocol)
Frozen feature spaces:
    eval_coverage  coverage@5 and 1-NN on an augmented bank
    eval_linear    augmented linear probe
"""
