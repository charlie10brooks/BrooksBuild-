import multiprocessing as mp
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.core.rasterstore import TiledRaster, EditTooLarge
from app.core.tiling import Window


def _worker(root, value, n):
    r = TiledRaster(root)
    for i in range(n):
        with r.edit("w"):
            r.write(Window(i, 0, i + 1, 1), value)
    r.close()


class RasterStoreTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.r = TiledRaster.create(self.dir / "mask", 1000, 700, tile=128)

    def tearDown(self):
        self.r.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_empty_raster_is_sparse_and_reads_fill(self):
        self.assertEqual(list(self.r.iter_existing_tiles()), [])
        self.assertEqual(int(self.r.read(Window(0, 0, 1000, 700)).sum()), 0)
        self.assertEqual(self.r.histogram(), {0: 700000})

    def test_write_across_tile_boundaries_roundtrip(self):
        rng = np.random.default_rng(1)
        data = rng.integers(0, 10, size=(300, 400), dtype=np.uint8)
        win = Window(90, 60, 490, 360)  # spans many tiles, unaligned
        self.r.write(win, data)
        np.testing.assert_array_equal(self.r.read(win), data)
        # persists across a fresh handle
        r2 = TiledRaster(self.dir / "mask")
        np.testing.assert_array_equal(r2.read(win), data)
        r2.close()

    def test_histogram_matches_ground_truth_after_edits(self):
        rng = np.random.default_rng(2)
        for _ in range(6):
            x, y = int(rng.integers(0, 800)), int(rng.integers(0, 500))
            w = Window(x, y, x + 150, y + 120)
            m = rng.random(w.shape) > 0.5
            self.r.paint(w, m, int(rng.integers(1, 6)))
        full = self.r.read(Window(0, 0, 1000, 700))
        truth = {int(v): int(c) for v, c in zip(*np.unique(full, return_counts=True))}
        self.assertEqual(self.r.histogram(), truth)
        self.assertEqual(self.r.rebuild_histogram(), truth)

    def test_undo_redo_restore_exact_pixels_and_counts(self):
        full = Window(0, 0, 1000, 700)
        self.r.paint(Window(100, 100, 400, 300), np.ones((200, 300), bool), 1)
        snap1 = self.r.read(full)
        self.r.paint(Window(200, 150, 600, 450), np.ones((300, 400), bool), 2)
        snap2 = self.r.read(full)
        self.assertNotEqual(self.r.histogram(), {0: 700000})

        self.assertIsNotNone(self.r.undo())
        np.testing.assert_array_equal(self.r.read(full), snap1)
        self.assertIsNotNone(self.r.undo())
        np.testing.assert_array_equal(self.r.read(full), np.zeros((700, 1000), np.uint8))
        self.assertEqual(self.r.histogram(), {0: 700000})
        self.assertEqual(list(self.r.iter_existing_tiles()), [], "undo to empty should leave no tiles")
        self.assertIsNone(self.r.undo())

        self.r.redo()
        self.r.redo()
        np.testing.assert_array_equal(self.r.read(full), snap2)
        self.assertIsNone(self.r.redo())

    def test_new_edit_discards_redo_branch(self):
        self.r.paint(Window(0, 0, 10, 10), np.ones((10, 10), bool), 1)
        self.r.paint(Window(20, 0, 30, 10), np.ones((10, 10), bool), 2)
        self.r.undo()
        self.assertTrue(self.r.can_redo())
        self.r.paint(Window(40, 0, 50, 10), np.ones((10, 10), bool), 3)
        self.assertFalse(self.r.can_redo())

    def test_noop_edit_is_not_recorded(self):
        self.r.paint(Window(0, 0, 10, 10), np.ones((10, 10), bool), 1)
        v = self.r.version
        self.r.paint(Window(0, 0, 10, 10), np.ones((10, 10), bool), 1)   # same value again
        self.assertEqual(self.r.version, v)
        self.r.undo()
        self.assertFalse(self.r.can_undo())

    def test_exception_in_edit_rolls_back_everything(self):
        with self.assertRaises(RuntimeError):
            with self.r.edit("boom"):
                self.r.write(Window(0, 0, 300, 300), 7)
                raise RuntimeError("fail")
        self.assertEqual(int(self.r.read(Window(0, 0, 300, 300)).sum()), 0)
        self.assertEqual(self.r.histogram(), {0: 700000})

    def test_edit_size_guard(self):
        r = TiledRaster.create(self.dir / "small", 4096, 4096, tile=64, max_edit_tiles=10)
        with self.assertRaises(EditTooLarge):
            r.write(Window(0, 0, 4096, 4096), 1)
        r.close()

    def test_bulk_mode_resets_history_and_keeps_counts(self):
        self.r.paint(Window(0, 0, 50, 50), np.ones((50, 50), bool), 1)
        with self.r.edit("ai", record_history=False):
            self.r.write(Window(0, 0, 1000, 700), 4)
        self.assertFalse(self.r.can_undo())
        self.assertEqual(self.r.histogram(), {4: 700000})
        self.assertEqual(self.r.rebuild_histogram(), {4: 700000})

    def test_out_of_bounds_window_reads_fill_and_clips_writes(self):
        self.r.write(Window(-20, -20, 30, 30), 5)
        out = self.r.read(Window(-20, -20, 30, 30))
        self.assertEqual(int(out[0, 0]), 0)
        self.assertEqual(int(out[25, 25]), 5)
        self.assertEqual(self.r.histogram()[5], 900)

    def test_crash_journal_triggers_histogram_rebuild(self):
        self.r.paint(Window(0, 0, 10, 10), np.ones((10, 10), bool), 3)
        (self.dir / "mask" / "pending.json").write_text("{}")
        # corrupt the cached totals to simulate an interrupted commit
        import json
        meta = json.loads((self.dir / "mask" / "meta.json").read_text())
        meta["hist"] = {"0": 1}
        (self.dir / "mask" / "meta.json").write_text(json.dumps(meta))
        r2 = TiledRaster(self.dir / "mask")
        self.assertEqual(r2.histogram(), {0: 699900, 3: 100})
        self.assertFalse((self.dir / "mask" / "pending.json").exists())
        r2.close()

    def test_concurrent_writers_do_not_lose_updates(self):
        ctx = mp.get_context("fork")
        ps = [ctx.Process(target=_worker, args=(str(self.dir / "mask"), v, 40)) for v in (1, 2)]
        # distinct pixels per worker are not guaranteed; assert only integrity
        for p in ps:
            p.start()
        for p in ps:
            p.join()
            self.assertEqual(p.exitcode, 0)
        r = TiledRaster(self.dir / "mask")
        full = r.read(Window(0, 0, 1000, 700))
        truth = {int(v): int(c) for v, c in zip(*np.unique(full, return_counts=True))}
        self.assertEqual(r.histogram(), truth)
        self.assertEqual(int((full[0, :40] > 0).sum()), 40)
        r.close()

    def test_float_raster_supported(self):
        f = TiledRaster.create(self.dir / "conf", 300, 300, tile=100, dtype="float32", fill=-9999)
        f.write(Window(10, 10, 20, 20), np.full((10, 10), 3.5, np.float32))
        self.assertAlmostEqual(float(f.read(Window(12, 12, 13, 13))[0, 0]), 3.5)
        self.assertEqual(float(f.read(Window(0, 0, 1, 1))[0, 0]), -9999.0)
        f.close()


if __name__ == "__main__":
    unittest.main()
