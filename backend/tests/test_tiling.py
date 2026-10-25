import unittest
import numpy as np
from app.core.tiling import Window, iter_tiles, tile_origins, blend_weights


class TilingTests(unittest.TestCase):
    def test_window_ops(self):
        a, b = Window(0, 0, 10, 10), Window(5, 5, 20, 20)
        self.assertEqual(a.intersect(b), Window(5, 5, 10, 10))
        self.assertIsNone(a.intersect(Window(10, 0, 12, 5)))
        self.assertEqual(Window(-5, -5, 3, 3).clip(10, 10), Window(0, 0, 3, 3))
        with self.assertRaises(ValueError):
            Window(5, 0, 1, 1)

    def test_origins_end_on_edge(self):
        o = tile_origins(1000, 256, 192)
        self.assertEqual(o[0], 0)
        self.assertEqual(o[-1] + 256, 1000)
        self.assertEqual(tile_origins(100, 256, 192), [0])

    def test_keep_regions_partition_image_exactly(self):
        for (w, h, t, ov) in [(1000, 777, 256, 64), (513, 300, 128, 0), (100, 90, 256, 32), (2049, 2049, 512, 100)]:
            cover = np.zeros((h, w), dtype=np.int32)
            for spec in iter_tiles(w, h, t, ov):
                # keep must lie inside the model's window
                self.assertEqual(spec.keep.intersect(spec.window), spec.keep)
                cover[spec.keep.slices()] += 1
            self.assertTrue((cover == 1).all(), f"gaps/overlaps for {(w, h, t, ov)}")

    def test_blend_weights_shape_and_range(self):
        wgt = blend_weights(64, 48, 16)
        self.assertEqual(wgt.shape, (64, 48))
        self.assertTrue((wgt > 0).all() and wgt.max() <= 1.0)
        self.assertAlmostEqual(float(wgt[32, 24]), 1.0)
        self.assertLess(float(wgt[0, 0]), float(wgt[32, 24]))


if __name__ == "__main__":
    unittest.main()
