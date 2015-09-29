import unittest
from refnx.dataset import ReflectDataset
import numpy as np
from numpy.testing import assert_equal
import os.path


path = os.path.dirname(os.path.abspath(__file__))

class TestReflectDataset(unittest.TestCase):

    def setUp(self):
        pass
             
    def test_load(self):
        '''
            test reflectivity calculation
            with values generated from Motofit
        
        '''
        dataset = ReflectDataset()
        with open(os.path.join(path, 'c_PLP0000708.xml')) as f:
            dataset.load(f)
        
        assert_equal(dataset.npoints, 90)
        assert_equal(90, np.size(dataset.x))
        
        dataset1 = ReflectDataset()
        with open(os.path.join(path, 'c_PLP0000708.dat')) as f:
            dataset1.load(f)
        
        assert_equal(dataset1.npoints, 90)
        assert_equal(90, np.size(dataset1.x))
        
        
if __name__ == '__main__':
    unittest.main()