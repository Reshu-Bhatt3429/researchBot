import unittest
import numpy as np
import pandas as pd
import research
import breakout_research

class ResearchTests(unittest.TestCase):
    def test_nonoverlap(self):
        np.testing.assert_array_equal(research.nonoverlap(np.array([1,2,5,6,10]),4),np.array([1,5,10]))

    def test_breakout_discards_ambiguous_dual_trigger(self):
        d=pd.DataFrame({"open":[100]*5,"high":[100,101,100,100,100],"low":[100,99,100,100,100],"close":[100]*5})
        self.assertTrue(breakout_research.simulate(d,np.array([0]),3,1,5,"continuation").empty)

    def test_hac_zero_mean(self):
        _,t,p1,p2=research.hac_inference(np.array([1.,-1.]*20))
        self.assertAlmostEqual(t,0.0)
        self.assertAlmostEqual(p1,0.5)
        self.assertAlmostEqual(p2,1.0)

if __name__=="__main__":unittest.main()
