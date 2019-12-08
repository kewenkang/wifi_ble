# util unit test
import unittest
from util import *
import pandas as pd
from pandas.testing import assert_frame_equal


class TestUtil(unittest.TestCase):
    def test_acf_norm(self):
        data1 = np.array([6, 7, 8, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 6, 7, 8], dtype=np.complex)
        data2 = np.array([6, 7, 8, 1, 2, 3, 1.1, 2, 2.9, 1.2, 1.8, 3, 1, 2, 3.3, 6, 7, 8], dtype=np.complex)
        # print(moving_avg(data1, window_size=6))
        # print(acf_norm(data1, ndelay=3, nwindow=6))
        # print(acf_norm(data2, ndelay=3, nwindow=6))

    def test_convert_seq2start_len(self):
        data_indices = [2,3,4,7,8,9]
        result = pd.DataFrame({'start_index': [2,7], 'length': [3,3]})
        assert_frame_equal(result, convert_seq2start_len(data_indices))

if __name__=='__main__':
    unittest.main()