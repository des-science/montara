import tempfile
import os

from montara.eastlake_step import read_galsim_truth_file

import pytest


@pytest.mark.parametrize("head", [True, False])
@pytest.mark.parametrize("ndata", [0, 1, 2])
def test_read_file(head, ndata):
    letters = "abc"
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "blah.dat")
        with open(fname, "w") as fp:
            if head:
                fp.write("# a band c\n")
            for nd in range(ndata):
                fp.write("%d %s %f\n" % (nd, letters[nd], nd*3.14159))

        if not head and ndata > 0:
            with pytest.raises(RuntimeError) as e:
                read_galsim_truth_file(fname)

            assert "No header line found for truth file" in str(e.value)
        else:
            d = read_galsim_truth_file(fname)
            if ndata == 0:
                assert d is None
            else:
                assert d.dtype.descr == [("a", "i8"), ("b", "U1"), ("c", "f8")]
                assert d.shape[0] == ndata
