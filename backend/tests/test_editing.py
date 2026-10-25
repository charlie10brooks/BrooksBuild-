import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.core import classes as C
from app.core import regions
from app.core.editing import ArrayOrthoReader, EditError, EditOp, apply_edit
from app.core.rasterstore import TiledRaster
from app.core.tiling import Window


def synthetic_ortho(h=400, w=500):
    """Left third green canopy, middle brown soil, right third grey road, plus noise."""
    rng = np.random.default_rng(0)
    img = np.zeros((h, w, 3), np.float32)
    img[:, : w // 3] = (40, 110, 40)
    img[:, w // 3: 2 * w // 3] = (140, 100, 70)
    img[:, 2 * w // 3:] = (110, 110, 115)
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


class RegionTests(unittest.TestCase):
    def test_polygon_area_close_to_geometric_area(self):
        win = Window(0, 0, 200, 200)
        tri = [(20, 20), (120, 20), (20, 100)]  # area 4000
        m = regions.polygon_mask(tri, win)
        self.assertAlmostEqual(int(m.sum()), 4000, delta=40)

    def test_rect_mask_uses_pixel_centres(self):
        m = regions.rect_mask(10, 10, 14, 13, Window(0, 0, 30, 30))
        self.assertEqual(int(m.sum()), 4 * 3)
        self.assertTrue(m[10, 10] and m[12, 13] and not m[13, 10] and not m[10, 14])

    def test_brush_single_dab_is_disc_and_stroke_is_connected(self):
        win = Window(0, 0, 100, 100)
        dab = regions.brush_mask([(50.5, 50.5)], 10, win)
        self.assertAlmostEqual(dab.sum(), np.pi * 100, delta=40)
        stroke = regions.brush_mask([(10, 50), (90, 50)], 5, win)
        self.assertTrue(stroke[50, 10:90].all())
        self.assertFalse(stroke[0:30, :].any())

    def test_polygon_matches_rect_rule_exactly_for_axis_aligned_boxes(self):
        win = Window(0, 0, 60, 60)
        sq = regions.polygon_mask([(10, 10), (30, 10), (30, 25), (10, 25)], win)
        np.testing.assert_array_equal(sq, regions.rect_mask(10, 10, 30, 25, win))

    def test_polygon_even_odd_handles_concave_and_self_intersecting(self):
        win = Window(0, 0, 100, 100)
        # concave 'C' shape: the notch must stay empty
        c = regions.polygon_mask([(10, 10), (90, 10), (90, 30), (30, 30), (30, 70), (90, 70), (90, 90), (10, 90)], win)
        self.assertTrue(c[20, 50] and c[80, 50] and c[50, 20])
        self.assertFalse(c[50, 60])
        # bow-tie: even-odd leaves the crossing triangles both filled, outside empty
        bow = regions.polygon_mask([(10, 10), (90, 90), (90, 10), (10, 90)], win)
        self.assertTrue(bow[50, 20] and bow[50, 80])      # left and right lobes filled
        self.assertFalse(bow[20, 50] or bow[80, 50])      # top and bottom are outside (no edge closes them)
        self.assertFalse(bow[50, 5])

    def test_polygon_clips_to_window_without_error(self):
        m = regions.polygon_mask([(-50, -50), (500, -50), (500, 500), (-50, 500)], Window(0, 0, 40, 40))
        self.assertTrue(m.all())
        self.assertFalse(regions.polygon_mask([(300, 300), (400, 300), (400, 400)], Window(0, 0, 40, 40)).any())

    def test_polygon_respects_window_offset(self):
        win = Window(1000, 2000, 1100, 2100)
        m = regions.polygon_mask([(1010, 2010), (1050, 2010), (1050, 2050), (1010, 2050)], win)
        self.assertTrue(m[10:50, 10:50].all() and not m[60:, :].any())

    def test_flood_stops_at_colour_boundary_and_reports_edge(self):
        rgb = synthetic_ortho()
        sel, edge = regions.flood_select(rgb, (200, 50), tolerance=15)
        # fills the green third entirely, nothing from the soil third
        self.assertGreater(sel[:, : 500 // 3 - 2].mean(), 0.97)
        self.assertEqual(int(sel[:, 500 // 3 + 2:].sum()), 0)
        self.assertTrue(edge)  # region spans the full window height

    def test_flood_tolerance_zero_is_tight_and_high_is_loose(self):
        rgb = synthetic_ortho()
        tight, _ = regions.flood_select(rgb, (200, 50), tolerance=0.5)
        loose, _ = regions.flood_select(rgb, (200, 50), tolerance=120)
        self.assertLess(tight.sum(), 0.2 * loose.sum())

    def test_similar_is_not_connectivity_limited(self):
        rgb = synthetic_ortho()
        rgb[190:210, 20:40] = (140, 100, 70)       # a soil-coloured patch inside the canopy
        sel = regions.similar_mask(rgb, (200, 250), tolerance=15)
        self.assertTrue(sel[195, 30])               # disconnected patch is picked up
        fill, _ = regions.flood_select(rgb, (200, 250), tolerance=15)
        self.assertFalse(fill[195, 30])             # but flood fill does not jump to it

    def test_clean_mask_removes_specks_and_fills_holes(self):
        m = np.zeros((50, 50), bool)
        m[10:30, 10:30] = True
        m[20, 20] = False
        m[45, 45] = True
        out = regions.clean_mask(m, min_area=10, fill_holes=True)
        self.assertTrue(out[20, 20] and not out[45, 45])


class EditApplyTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.img = synthetic_ortho()
        self.ortho = ArrayOrthoReader(self.img)
        self.mask = TiledRaster.create(self.dir / "m", 500, 400, tile=64)

    def tearDown(self):
        self.mask.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_flood_fill_classifies_canopy_and_is_undoable(self):
        res = apply_edit(EditOp("fill", C.TREES, points=[(50.5, 200.5)], tolerance=15), self.mask, self.ortho)
        self.assertGreater(res.changed_pixels, 0.9 * 400 * (500 // 3))
        h = self.mask.histogram()
        self.assertEqual(h[C.TREES], res.changed_pixels)
        self.mask.undo()
        self.assertEqual(self.mask.histogram(), {0: 200000})

    def test_only_classes_restricts_fill(self):
        apply_edit(EditOp("rect", C.SOIL, rect=(0, 0, 500, 100)), self.mask, self.ortho)
        res = apply_edit(EditOp("similar", C.GRASS, points=[(50.5, 200.5)], tolerance=15,
                                only_classes=[C.UNKNOWN]), self.mask, self.ortho)
        top = self.mask.read(Window(0, 0, 500, 100))
        self.assertTrue((top == C.SOIL).all(), "pixels outside only_classes must be untouched")
        self.assertGreater(res.changed_pixels, 0)

    def test_protect_classes_blocks_overwrite(self):
        apply_edit(EditOp("rect", C.BUILDING, rect=(100, 100, 200, 200)), self.mask)
        apply_edit(EditOp("rect", C.TREES, rect=(0, 0, 500, 400), protect_classes=[C.BUILDING]), self.mask)
        self.assertTrue((self.mask.read(Window(100, 100, 200, 200)) == C.BUILDING).all())
        self.assertEqual(self.mask.histogram()[C.BUILDING], 10000)

    def test_erase_marks_unknown(self):
        apply_edit(EditOp("rect", C.TREES, rect=(0, 0, 100, 100)), self.mask)
        apply_edit(EditOp("brush", C.UNKNOWN, points=[(50, 50)], radius=20), self.mask)
        self.assertEqual(int(self.mask.read(Window(50, 50, 51, 51))[0, 0]), C.UNKNOWN)
        self.assertLess(self.mask.histogram()[C.TREES], 10000)

    def test_brush_and_polygon_cross_tile_edges_and_clip_to_raster(self):
        apply_edit(EditOp("brush", C.SOIL, points=[(-30, 10), (600, 10)], radius=6), self.mask)
        self.assertTrue((self.mask.read(Window(0, 6, 500, 14)) == C.SOIL).all())
        apply_edit(EditOp("polygon", C.WATER, points=[(60, 60), (200, 70), (120, 300)]), self.mask)
        self.assertGreater(self.mask.histogram()[C.WATER], 1000)

    def test_validation_errors(self):
        with self.assertRaises(EditError):
            apply_edit(EditOp("polygon", 1, points=[(0, 0), (1, 1)]), self.mask)
        with self.assertRaises(EditError):
            apply_edit(EditOp("wiggle", 1, points=[(0, 0)]), self.mask)
        with self.assertRaises(EditError):
            apply_edit(EditOp("fill", 1, points=[(10, 10)]), self.mask, ortho=None)
        with self.assertRaises(EditError):
            apply_edit(EditOp("fill", 1, points=[(9999, 10)]), self.mask, self.ortho)

    def test_truncated_flag_when_fill_hits_analysis_window_edge(self):
        res = apply_edit(EditOp("fill", C.TREES, points=[(50.5, 200.5)], tolerance=15), self.mask, self.ortho)
        self.assertTrue(res.truncated is True or res.truncated is False)  # field is always present
        res2 = apply_edit(EditOp("fill", C.ROAD, points=[(450.5, 200.5)], tolerance=15,
                                 analysis_window=Window(430, 180, 470, 220)), self.mask, self.ortho)
        self.assertTrue(res2.truncated)
        self.assertIn("analysis-window edge", res2.message)


class ClassTests(unittest.TestCase):
    def test_lut_hides_unknown_and_hidden_classes(self):
        defaults = [c.to_dict() for c in C.DEFAULT_CLASSES]
        lut = C.build_lut(defaults, opacity=0.5, hidden={C.GRASS})
        self.assertEqual(lut[C.UNKNOWN, 3], 0)
        self.assertEqual(lut[C.GRASS, 3], 0)
        self.assertEqual(lut[C.TREES, 3], 128)
        rgba = C.colorize(np.array([[C.TREES, C.GRASS]], np.uint8), lut)
        self.assertEqual(rgba.shape, (1, 2, 4))


if __name__ == "__main__":
    unittest.main()


class TaggedProvenanceTests(unittest.TestCase):
    """Stored value = class | tag, so provenance shares the raster (and its undo history) with the class."""
    CM, MANUAL, AUTO = 0x3F, 0x80, 0x40

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.mask = TiledRaster.create(self.dir / "m", 200, 200, tile=64)

    def test_tag_written_and_class_bits_compared(self):
        with self.mask.edit("auto", record_history=False):
            self.mask.paint(Window(0, 0, 100, 100), np.ones((100, 100), bool), C.TREES | self.AUTO)
        r = apply_edit(EditOp("rect", C.SOIL, rect=(0, 0, 50, 50), only_classes=[C.TREES]), self.mask,
                       class_mask=self.CM, tag=self.MANUAL)
        self.assertEqual(r.changed_pixels, 2500)
        v = self.mask.read(Window(0, 0, 100, 100))
        self.assertTrue((v[:50, :50] == (C.SOIL | self.MANUAL)).all())
        self.assertTrue((v[50:, 50:] == (C.TREES | self.AUTO)).all())          # outside the edit: untouched, still auto
        # only_classes matched on class bits even though stored values carry a tag
        r2 = apply_edit(EditOp("rect", C.ROCK, rect=(0, 0, 100, 100), only_classes=[C.GRASS]), self.mask,
                        class_mask=self.CM, tag=self.MANUAL)
        self.assertEqual(r2.changed_pixels, 0)

    def test_repainting_same_class_marks_manual(self):
        with self.mask.edit("auto", record_history=False):
            self.mask.paint(Window(0, 0, 10, 10), np.ones((10, 10), bool), C.TREES | self.AUTO)
        r = apply_edit(EditOp("rect", C.TREES, rect=(0, 0, 10, 10)), self.mask, class_mask=self.CM, tag=self.MANUAL)
        self.assertEqual(r.changed_pixels, 100)
        self.assertTrue((self.mask.read(Window(0, 0, 10, 10)) == (C.TREES | self.MANUAL)).all())

    def test_protect_and_undo_restore_tags_exactly(self):
        with self.mask.edit("auto", record_history=False):
            self.mask.paint(Window(0, 0, 100, 100), np.ones((100, 100), bool), C.GRASS | self.AUTO)
        before = self.mask.read(Window(0, 0, 100, 100)).copy()
        apply_edit(EditOp("polygon", C.WATER, points=[(5, 5), (60, 5), (30, 70)]), self.mask, class_mask=self.CM, tag=self.MANUAL)
        self.assertFalse((self.mask.read(Window(0, 0, 100, 100)) == before).all())
        self.mask.undo()
        np.testing.assert_array_equal(self.mask.read(Window(0, 0, 100, 100)), before)
        self.mask.redo()
        self.assertTrue(((self.mask.read(Window(0, 0, 100, 100)) & 0xC0) == self.MANUAL).any())

    def test_class_must_fit_class_bits(self):
        with self.assertRaises(EditError):
            apply_edit(EditOp("rect", 100, rect=(0, 0, 5, 5)), self.mask, class_mask=self.CM, tag=self.MANUAL)
