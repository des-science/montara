try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("montara")
except PackageNotFoundError:
    # package is not installed
    pass

from . import des_tile  # noqa
from . import input_desstar  # noqa
from . import catalogsampler  # noqa
from . import catchffterror_stamp  # noqa
from . import utils  # noqa
from . import z_slice_shear  # noqa
from . import mixed_scene_postop  # noqa
from . import badpixfromfits  # noqa
from . import eastlake_step  # noqa
