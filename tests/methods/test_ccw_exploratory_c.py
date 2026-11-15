"""Contract tests for isolated research-only inference and paired evaluation."""
import unittest
import numpy as np
from experiments.ccw_v6.model import fit,infer,METHODS
from experiments.ccw_v6.exploration.evaluate import metric
from experiments.ccw_v6.exploration.run import dag


class ExploratoryContract(unittest.TestCase):
    def test_predictions_do_not_require_labels_or_groups(self):
        rng=np.random.default_rng(71);p=rng.uniform(.1,1,(48,15,2));p/=p.sum(-1,keepdims=True)
        d=rng.uniform(.4,1.6,p.shape);y=np.arange(48)%2;ids=np.array([str(i) for i in range(48)]);heads=np.array([str(i) for i in range(15)])
        fitted=fit(p,d,y,ids,heads)
        for method in METHODS:
            a=infer(p,d,fitted,heads,method)
            np.testing.assert_allclose(a['probs'].sum(-1),1,atol=1e-12)
            np.testing.assert_allclose(a['weights'].sum(-1),1,atol=1e-12)
            self.assertEqual(a['weights'].shape,(48,15))
            self.assertFalse(any(k in a for k in ['y','group']))
        with self.assertRaises(ValueError):infer(p,d,fitted,heads[::-1],'CCW_FULL')

    def test_all_inference_before_any_ccw_evaluation(self):
        d=dag();inferences=[k for k in d if k.startswith('infer:')]
        self.assertEqual(len(inferences),45)
        self.assertEqual(set(d['inference_complete']),set(inferences))
        self.assertTrue(all(v==['inference_complete'] for k,v in d.items() if k.startswith('eval:')))
        self.assertFalse(any('selection' in k or 'promotion' in k for k in d))

    def test_wga_is_minimum_not_average(self):
        y=np.array([0,0,0,0,1,1,1,1]);g=np.repeat(np.arange(4),2);pred=y.copy();pred[4]=0
        p=np.eye(2)[pred];r=metric(p,y,g)
        self.assertEqual(r['AA'],.875);self.assertEqual(r['WGA'],.5);self.assertEqual(r['WCA'],.75)
        self.assertEqual(r['group_count'],[2,2,2,2])


if __name__=='__main__':unittest.main()
