"""Gallery synchronization tests; no embedder or model required."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from embed_append import update_gallery


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


if __name__ == "__main__":
    unittest.main()
