from .kitti_dataset import KITTIDataset
from .sceneflow_dataset import SceneFlowDatset
from .drivingstereo_dataset import DrivingStereoDataset
from .em_dataset import EMDataset
from .eth3d_dataset import Eth3DDataset
from .middlebury_dataset import MiddleburyDataset

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    "drivingstereo": DrivingStereoDataset,
    "em": EMDataset,
    "eth3d": Eth3DDataset,
    "middlebury": MiddleburyDataset
}
