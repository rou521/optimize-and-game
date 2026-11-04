"""必要数值检查：秩亏/零梯度、参考公式、梯度、下降界、神经网络反传。"""
import unittest
import numpy as np
from muon_study.optimizers import Muon, AdamW, newton_schulz, polar_svd
from muon_study.problems import make_problem, mlp_initial, mlp_value_grad
from muon_study.experiments import convergence_run


class TestCore(unittest.TestCase):
    def test_polar_duality_rectangular_and_rank_deficient(self):
        rng = np.random.default_rng(7)
        for shape in ((7, 4), (4, 7), (4, 4)):
            g = rng.normal(size=shape)
            g[:, -1] = g[:, 0]  # 显式制造列相关。
            p = polar_svd(g)
            singular = np.linalg.svd(g, compute_uv=False)
            self.assertAlmostEqual(np.sum(g*p), singular.sum(), places=11)
            self.assertLessEqual(np.linalg.norm(p, 2), 1+1e-12)
            self.assertAlmostEqual(np.linalg.norm(p, "fro")**2, np.linalg.matrix_rank(g), places=10)

    def test_zero_and_rank_one(self):
        np.testing.assert_array_equal(polar_svd(np.zeros((5, 3))), 0)
        np.testing.assert_array_equal(newton_schulz(np.zeros((5, 3))), 0)
        g = np.outer([1., 2., 3.], [4., 5.])
        np.testing.assert_allclose(polar_svd(g), g/np.linalg.norm(g), atol=1e-14)

    def test_ns_matches_scalar_polynomial(self):
        s = np.array([1.0, .3, .01])
        expected = s/(np.linalg.norm(s)+1e-7)
        for _ in range(5):
            expected = 3.4445*expected - 4.7750*expected**3 + 2.0315*expected**5
        np.testing.assert_allclose(np.diag(newton_schulz(np.diag(s))), expected, atol=1e-12)

    def test_ns_transpose_consistency(self):
        g = np.random.default_rng(2).normal(size=(9, 4))
        np.testing.assert_allclose(newton_schulz(g).T, newton_schulz(g.T), atol=1e-13)

    def test_reference_momentum_and_nonmutation(self):
        rng = np.random.default_rng(3)
        optimizer = Muon(beta=.95)
        m = np.zeros((3, 5))
        for _ in range(3):
            g = rng.normal(size=m.shape)
            original = g.copy()
            m = .95*m + .05*g
            expected = newton_schulz(.05*g+.95*m)
            np.testing.assert_allclose(optimizer.direction(g), expected, atol=1e-12)
            np.testing.assert_array_equal(g, original)

    def test_shape_scale_and_weight_decay(self):
        g = np.arange(1., 19.).reshape(6, 3)
        w = np.ones_like(g)
        expected = .99*w-.1*np.sqrt(2)*newton_schulz(g)
        opt = Muon(beta=0, weight_decay=.1, scale_mode="jordan")
        np.testing.assert_allclose(opt.step(w, g, .1), expected, atol=1e-12)

    def test_matrix_gradients(self):
        for kind in ("quadratic", "nonconvex"):
            problem = make_problem(kind, 3)
            w = problem.initial.copy()
            gradient = problem.value_grad(w)[1]
            direction = np.random.default_rng(4).normal(size=w.shape)
            eps = 1e-5
            fd = (problem.value_grad(w+eps*direction)[0]-problem.value_grad(w-eps*direction)[0])/(2*eps)
            self.assertAlmostEqual(fd, np.sum(gradient*direction), places=6)

    def test_convergence_bounds(self):
        for kind in ("quadratic", "nonconvex"):
            for beta in (0, .9):
                for nesterov in (False, True):
                    result = convergence_run(make_problem(kind, 0), 64, beta, nesterov)
                    self.assertLessEqual(result["ratio"], 1+1e-9)

    def test_neural_backprop_all_parameter_groups(self):
        p = mlp_initial(0)
        rng = np.random.default_rng(4)
        x, y = rng.normal(size=(7, 10)), rng.normal(size=(7, 3))
        _, gradients = mlp_value_grad(p, x, y)
        for name in p:
            direction = rng.normal(size=p[name].shape)
            plus, minus = dict(p), dict(p)
            plus[name] = p[name]+1e-6*direction
            minus[name] = p[name]-1e-6*direction
            fd = (mlp_value_grad(plus,x,y)[0]-mlp_value_grad(minus,x,y)[0])/2e-6
            self.assertAlmostEqual(fd, np.sum(gradients[name]*direction), places=7)

    def test_adamw_first_update(self):
        p = {"w": np.array([1., -2.])}
        g = {"w": np.array([.2, -.4])}
        result = AdamW(weight_decay=.1).step(p, g, .01)
        np.testing.assert_allclose(result["w"], .999*p["w"]-.01*g["w"]/(np.abs(g["w"])+1e-8))

    def test_reject_invalid_inputs(self):
        with self.assertRaises(ValueError):
            Muon(beta=1)
        with self.assertRaises(ValueError):
            polar_svd(np.array([1., 2.]))
        with self.assertRaises(ValueError):
            newton_schulz(np.array([[np.nan]]))


if __name__ == "__main__":
    unittest.main()
