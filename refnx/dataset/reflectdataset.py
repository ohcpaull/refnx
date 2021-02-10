import string
import time
import re

# from datetime import datetime

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import os.path
from refnx.dataset import Data1D
from refnx._lib import possibly_open_file

import numpy as np

_template_ref_xml = """<?xml version="1.0"?>
<REFroot xmlns="">
<REFentry time="$time">
<Title>$title</Title>
<User>$user</User>
<REFsample>
<ID>$sample</ID>
</REFsample>
<REFdata axes="Qz" rank="1" type="POINT" spin="UNPOLARISED" dim="$numpoints">
<Run filename="$datafilenumber" preset="" size="">
</Run>
<R uncertainty="dR">$_ydata</R>
<Qz uncertainty="dQz" units="1/A">$_xdata</Qz>
<dR type="SD">$_ydataSD</dR>
<dQz type="_FWHM" units="1/A">$_xdataSD</dQz>
</REFdata>
</REFentry>
</REFroot>"""


class ReflectDataset(Data1D):
    """
    A 1D Reflectivity dataset.
    """

    def __init__(self, data=None, **kwds):
        """
        Initialise a reflectivity dataset.

        Parameters
        ----------
        data : str, file-like or tuple of np.ndarray, optional
            `data` can be a string or file-like object referring to a File to
            load the dataset from.

            Alternatively it is a tuple containing the data from which the
            dataset will be constructed. The tuple should have between 2 and 4
            members.

                - data[0] - Q
                - data[1] - R
                - data[2] - dR
                - data[3] - dQ

            `data` must be at least two long, `Q` and `R`.
            If the tuple is at least 3 long then the third member is `dR`.
            If the tuple is 4 long then the fourth member is `dQ`.
            All arrays must have the same shape.
        """
        super(ReflectDataset, self).__init__(data=data, **kwds)
        self.datafilenumber = list()
        self.sld_profile = None

    def __repr__(self):
        msk = self._mask
        if np.all(self._mask):
            msk = None

        d = {"filename": self.filename, "msk": msk, "data": self.data}
        if self.filename is not None:
            return "ReflectDataset(data={filename!r}," " mask={msk!r})".format(
                **d
            )
        else:
            return "ReflectDataset(data={data!r}," " mask={msk!r})".format(**d)

    def save_xml(self, f, start_time=0):
        """
        Saves the reflectivity data to an XML file.

        Parameters
        ----------
        f : str or file-like
            The file to write the spectrum to, or a str that specifies the file
            name
        start_time: int, optional
            Epoch time specifying when the sample started
        """
        s = string.Template(_template_ref_xml)
        self.time = time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(start_time)
        )
        # self.time = time.strftime(
        # datetime.fromtimestamp(start_time).isoformat()
        # filename = 'c_PLP{:07d}_{:d}.xml'.format(self._rnumber[0], 0)

        self._ydata = repr(self.y.tolist()).strip(",[]")
        self._xdata = repr(self.x.tolist()).strip(",[]")
        self._ydataSD = repr(self.y_err.tolist()).strip(",[]")
        self._xdataSD = repr(self.x_err.tolist()).strip(",[]")

        thefile = s.safe_substitute(self.__dict__)

        with possibly_open_file(f, "wb") as g:
            if "b" in g.mode:
                thefile = thefile.encode("utf-8")

            g.write(thefile)

    def load(self, f):
        """
        Load a dataset from file. Can either be 2-4 column ascii or XML file.

        Parameters
        ----------
        f : str or file-like
            The file to load the spectrum from, or a str that specifies the
            file name
        """
        if hasattr(f, "name"):
            fname = f.name
        else:
            fname = f
        try:
            tree = ET.ElementTree()
            tree.parse(f)

            delim = ", | |,"
            qtext = re.split(delim, tree.find(".//Qz").text)
            rtext = re.split(delim, tree.find(".//R").text)
            drtext = re.split(delim, tree.find(".//dR").text)
            dqtext = re.split(delim, tree.find(".//dQz").text)

            qvals = [float(val) for val in qtext if len(val)]
            rvals = [float(val) for val in rtext if len(val)]
            drvals = [float(val) for val in drtext if len(val)]
            dqvals = [float(val) for val in dqtext if len(val)]

            self.filename = fname
            self.name = os.path.splitext(os.path.basename(fname))[0]
            self.data = (qvals, rvals, drvals, dqvals)
        except ET.ParseError:
            super(ReflectDataset, self).load(fname)


def simulreflec_format(ref1, ref2, data_folder=None):
    """
    Function that converts individual .dat files to simulreflec format.
    For R++ and R--, produces a .txt file with 5 columns corresponding
    to Q, R++, R++_err, R--, R--_err. See the simulreflec documentation at
    `http://www-llb.cea.fr/prism/programs/simulreflec/simulreflec.html`

    Parameters
    ----------
    ref1 : Data1D or str
        This is the R++ reflectivity file
    ref2 : Data1D or str
        This is the R-- reflectivity file

    Returns
    -------
    valid : bool
        If the input files are valid, then will write file and return
        True, otherwise returns ValueError.
    """
    if all(isinstance(d, Data1D) for d in [ref1, ref2]):
        data1 = ref1
        data2 = ref2

    elif all(type(d) is str for d in [ref1, ref2]):
        data1 = Data1D(os.path.join(data_folder, ref1))
        data2 = Data1D(os.path.join(data_folder, ref2))
    else:
        raise ValueError(
            "Input should either be Data1D or string of reduced .dat file"
        )

    # Convert Q data from inverse Angstroms to inverse nanometres
    x = data1.x * 10
    y1 = data1.y
    y1_err = data1.y_err
    y2 = data2.y
    y2_err = data2.y_err
    data = np.array([x, y1, y1_err, y2, y2_err])

    fname = str(data1.name + "_" + data2.name + ".txt")

    with open(fname, "w") as f:
        comment = "# Comment = {}_{}\n".format(data1.name, data2.name)
        radiation = "# Particles = neutrons\n"
        polarisation = "# Polarisation = polarised\n"
        Qvals = "# Abscisses = nm-1\n"
        tof = "# TimeOfFlight = False\n"
        f.write(comment)
        f.write(radiation)
        f.write(polarisation)
        f.write(Qvals)
        f.write(tof)
        np.savetxt(f, data.T, delimiter="\t")

    return True
