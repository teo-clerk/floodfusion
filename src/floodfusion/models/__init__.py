"""Model components: encoders, cross-attention fusion, decoder, segmenter, losses.

The architecture is derived in ``docs/07_fusion_architectures.md``; the losses in
``docs/08_training_recipe_and_vram.md`` section 2. Read both before implementing.

The single structural invariant
-------------------------------
The fused representation must reduce **exactly** to the SAR representation when the
optical branch is absent::

    F_fused = F_sar + g * CrossAttn(Q=F_opt, K=F_sar, V=F_sar)

so that ``g -> 0`` gives ``F_fused == F_sar`` bit for bit. This satisfies operational
requirement R2 (``docs/00`` section 7) *structurally* rather than by hoping the network
learns it, and it is asserted by contract 3 in ``tests/test_fusion_module.py``.

Every module here emits **logits**. Never apply a sigmoid inside the model: the losses
depend on receiving logits for numerical stability (``docs/07`` section 9.2).
"""

from __future__ import annotations

__all__: list[str] = []
