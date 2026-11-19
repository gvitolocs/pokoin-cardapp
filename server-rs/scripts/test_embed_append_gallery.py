"""Gallery synchronization tests; no embedder or model required."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from embed_append import gallery_filename, update_gallery


class GalleryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.galleries = [self.root / "cdn_images", self.root / "cdn_images_delta"]
        for gallery in self.galleries: gallery.mkdir()
        self.face = self.root / "recovered.jpg"
        self.face.write_bytes(b"recovered face")

    def test_replace_with_backup(self):
        filename = "original-row-name.jpg"
        for i, gallery in enumerate(self.galleries):
            (gallery / filename).write_bytes(f"placeholder {i}".encode())
        self.assertEqual(update_gallery(self.galleries, [(filename, self.face)], "test-stamp"), (2, 0))
        for i, gallery in enumerate(self.galleries):
            self.assertEqual((gallery / filename).read_bytes(), b"recovered face")
            backup = self.root / "backup-gallery-test-stamp" / gallery.name / filename
            self.assertEqual(backup.read_bytes(), f"placeholder {i}".encode())
            self.assertEqual(list(gallery.iterdir()), [gallery / filename])

    def test_add_to_last_dir(self):
        self.galleries[-1].rmdir()
        self.assertEqual(update_gallery(self.galleries, [("missing.jpg", self.face)], "test"), (0, 1))
        self.assertFalse((self.galleries[0] / "missing.jpg").exists())
        self.assertEqual((self.galleries[-1] / "missing.jpg").read_bytes(), b"recovered face")
        self.assertFalse((self.root / "backup-gallery-test").exists())

    def test_skip_existing_for_append(self):
        for i, gallery in enumerate(self.galleries):
            (gallery / f"existing-{i}.jpg").write_bytes(b"existing face")
        items = [(f"existing-{i}.jpg", self.face) for i in range(2)]
        items += [("new.jpg", self.face), ("new.jpg", self.face)]
        self.assertEqual(update_gallery(self.galleries, [], "test", items), (0, 1))
        for i, gallery in enumerate(self.galleries):
            self.assertEqual((gallery / f"existing-{i}.jpg").read_bytes(), b"existing face")
            self.assertFalse((self.galleries[1-i] / f"existing-{i}.jpg").exists())
        self.assertEqual((self.galleries[-1] / "new.jpg").read_bytes(), b"recovered face")
        self.assertFalse((self.galleries[0] / "new.jpg").exists())
        self.assertFalse((self.root / "backup-gallery-test").exists())

    def test_failed_atomic_replace_preserves_original_and_cleans_temp(self):
        target = self.galleries[0] / "old.jpg"
        target.write_bytes(b"old face")
        with patch("embed_append.os.replace", side_effect=OSError("replace failed")) as replace:
            with self.assertRaisesRegex(OSError, "replace failed"):
                update_gallery(self.galleries, [(target.name, self.face)], "test")
        tmp, dst = replace.call_args.args
        self.assertEqual(Path(tmp).parent, target.parent)
        self.assertTrue(Path(tmp).name.startswith(".old.jpg."))
        self.assertEqual(dst, target)
        self.assertEqual(list(target.parent.iterdir()), [target])
        self.assertEqual(target.read_bytes(), b"old face")
        self.assertEqual((self.root / "backup-gallery-test" / target.parent.name / target.name).read_bytes(), b"old face")

    def test_empty_filename_falls_back_to_face_name(self):
        face = self.root / "123_some-card.jpg"
        face.write_bytes(b"named face")
        for empty in ("", None, "  "):
            self.assertEqual(gallery_filename(empty, face), "123_some-card.jpg")
        self.assertEqual(gallery_filename("kept.jpg", face), "kept.jpg")
        filename = gallery_filename("", face)
        self.assertEqual(update_gallery(self.galleries, [(filename, face)], "test"), (0, 1))
        self.assertEqual((self.galleries[-1] / "123_some-card.jpg").read_bytes(), b"named face")
        self.assertEqual(update_gallery(self.galleries, [(filename, face)], "test2"), (1, 0))
        self.assertEqual(sorted(p.name for p in self.galleries[-1].iterdir()), ["123_some-card.jpg"])

    def test_invalid_filenames_raise_before_any_write(self):
        (self.galleries[0] / "dir.jpg").mkdir()
        for bad in ("", "a/b.jpg", "../x.jpg", "..", "dir.jpg"):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "gallery item"):
                    update_gallery(self.galleries, [("good.jpg", self.face), (bad, self.face)], "test")
                with self.assertRaisesRegex(ValueError, "gallery item"):
                    update_gallery(self.galleries, [], "test", [("good.jpg", self.face), (bad, self.face)])
                self.assertFalse((self.galleries[-1] / "good.jpg").exists())
                self.assertEqual(list(self.galleries[-1].iterdir()), [])
                self.assertEqual(list(self.root.glob("backup-gallery-*")), [])
        self.assertEqual(sorted(p.name for p in self.root.iterdir() if p.parent == self.root and p.name.startswith(".")), [])
        self.assertFalse((self.root / "x.jpg").exists())


class AppEmbedWorstRowsTests(unittest.TestCase):
    def test_three_lowest_of_five_returned_ascending(self):
        from app_embed import worst_rows
        cosines = [0.9900, 0.9455, 0.9999, 0.9300, 0.9800]
        ids = ["10", "20", "30", "40", "50"]
        images = ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"]
        self.assertEqual(worst_rows(cosines, ids, images),
                         [(0.93, "40", "d.jpg"), (0.9455, "20", "b.jpg"), (0.98, "50", "e.jpg")])

    def test_fewer_rows_than_count_returns_all(self):
        from app_embed import worst_rows
        self.assertEqual(worst_rows([0.9, 0.5], ["2", "1"], ["b.jpg", "a.jpg"], count=3),
                         [(0.5, "1", "a.jpg"), (0.9, "2", "b.jpg")])

    def test_ties_on_cosine_ordered_by_row_id(self):
        from app_embed import worst_rows
        result = worst_rows([0.9, 0.9, 0.9], ["1002", "1001", "1003"],
                            ["x.jpg", "y.jpg", "z.jpg"], count=3)
        self.assertEqual(result, [(0.9, "1001", "y.jpg"), (0.9, "1002", "x.jpg"), (0.9, "1003", "z.jpg")])

    def test_input_order_does_not_change_result(self):
        from app_embed import worst_rows
        cosines = [0.97, 0.92, 0.99, 0.93]
        ids = ["4", "1", "3", "2"]
        images = ["d.jpg", "a.jpg", "c.jpg", "b.jpg"]
        forward = worst_rows(cosines, ids, images, count=2)
        backward = worst_rows(cosines[::-1], ids[::-1], images[::-1], count=2)
        self.assertEqual(forward, backward)


if __name__ == "__main__":
    unittest.main()
