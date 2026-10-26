"""VLFM policy adapter for observations produced by Isaac Sim.

The original real-world adapter eagerly loads a monocular depth model for the
Spot hand camera.  Isaac Sim already provides metric depth, so loading that
model wastes several gigabytes of GPU memory.  This subclass keeps the tested
RealityMixin observation/action contract while skipping the unused model.
"""

from typing import Any

from vlfm.policy.itm_policy import ITMPolicyV2
from vlfm.policy.reality_policies import RealityMixin


class IsaacSimITMPolicyV2(RealityMixin, ITMPolicyV2):
    """Continuous VLFM policy for an Isaac Sim differential-drive robot."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ITMPolicyV2.__init__(self, sync_explored_areas=True, *args, **kwargs)
        self._object_map.use_dbscan = False

    def _infer_depth(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("IsaacSimITMPolicyV2 expects metric depth from Isaac Sim")
