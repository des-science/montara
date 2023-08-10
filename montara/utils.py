# Some misc utility functions for pipeline code
from __future__ import print_function
import errno
import os
import numpy as np


def safe_rm(pth, verbose=False):
    try:
        os.remove(pth)
    except Exception as e:
        if verbose:
            print("removing file %s failed w/ error %r" % (pth, e))
        pass


def safe_mkdir(d):
    try:
        os.makedirs(d)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e


def add_field(a, descr, arrays):
    """Return a new array that is like "a", but has additional fields.

    Arguments:
      a     -- a structured numpy array
      descr -- a numpy type description of the new fields

    The contents of "a" are copied over to the appropriate fields in
    the new array, whereas the new fields are uninitialized.  The
    arguments are not modified.

    >>> sa = numpy.array([(1, 'Foo'), (2, 'Bar')], \
                         dtype=[('id', int), ('name', 'S3')])
    >>> sa.dtype.descr == numpy.dtype([('id', int), ('name', 'S3')])
    True
    >>> sb = add_field(sa, [('score', float)])
    >>> sb.dtype.descr == numpy.dtype([('id', int), ('name', 'S3'), \
                                       ('score', float)])
    True
    >>> numpy.all(sa['id'] == sb['id'])
    True
    >>> numpy.all(sa['name'] == sb['name'])
    True
    """
    if a.dtype.fields is None:
        raise ValueError("`A' must be a structured numpy array")
    b = np.empty(a.shape, dtype=a.dtype.descr + descr)
    for name in a.dtype.names:
        b[name] = a[name]
    for d, c in zip(descr, arrays):
        b[d[0]] = c
    return b


def get_truth_from_image_file(image_file, tilename):
    """Get the truth catalog path from the image path and tilename.
    Parameters
    ----------
    image_file : str
        The path to the image file.
    tilename : str
        The name of the coadd tile.
    Returns
    -------
    truth_path : str
        The path to the truth file.
    """
    return os.path.join(
        os.path.dirname(image_file),
        "truth_%s_%s.dat" % (tilename, os.path.basename(image_file)))
