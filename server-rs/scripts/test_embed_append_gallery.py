"""Gallery synchronization tests; no embedder or model required."""
import random
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

from embed_append import gallery_filename, update_gallery

CDN = "https://cdn.pokoin.com/"


def face(index):
    """The objects file name row `index` points at, unique per row."""
    return f"{2 * index}_card.jpg"


def app_rows(count):
    """Catalog rows in the shape app_embed reads: a public id and a cdn image URL."""
    return [{"id": str(2 * i), "image": CDN + face(i)} for i in range(count)]


def stored_vectors(count):
    """(count, 128) float16 rows of the unit vector e0, as a catalog embeddings file holds them."""
    return np.tile(np.eye(128, dtype=np.float16)[0], (count, 1))


def matching_fresh(indices):
    """Fake embedder returning e0 per row: a fresh embedding that agrees with the stored vector."""
    return np.tile(np.eye(128, dtype=np.float32)[0], (len(indices), 1))


def perturbed_fresh(indices):
    """Fake embedder returning e1 per row: cosine 0 against every stored vector."""
    return np.tile(np.eye(128, dtype=np.float32)[1], (len(indices), 1))


def fresh_with_one_failure(failing):
    """Fake embedder agreeing with the stored vector for all rows but `failing`, which comes back unrelated."""
    def embed_rows(indices):
        fresh = matching_fresh(indices)
        for at, index in enumerate(indices):
            if index == failing:
                fresh[at] = np.eye(128, dtype=np.float32)[1]
        return fresh
    return embed_rows


def already_unit(vectors):
    """normalize() stand-in for vectors that are already unit length."""
    return vectors


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


class CatalogStampTests(unittest.TestCase):
    """When the catalog's own vectors were computed, read from its stamp sources and never from a file mtime."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def sidecar(self, text):
        (self.root / "embedded-at.json").write_text(text)

    def test_run_start_sidecar_wins(self):
        from app_embed import catalog_stamp
        self.sidecar('{"runStart": "2026-11-19T23:43:58Z"}')
        stamp, source = catalog_stamp(self.root, {"generatedAt": "2026-01-01T00:00:00Z"})
        self.assertEqual(source, "embedded-at.json")
        self.assertEqual(stamp, datetime(2026, 11, 19, 23, 43, 58, tzinfo=timezone.utc).timestamp())

    def test_generated_at_alone_with_an_offset(self):
        from app_embed import catalog_stamp
        stamp, source = catalog_stamp(self.root, {"generatedAt": "2026-11-19T12:00:00+02:00"})
        self.assertEqual((source, stamp),
                         ("index.generatedAt", datetime(2026, 11, 19, 10, 0, 0, tzinfo=timezone.utc).timestamp()))

    def test_unusable_sidecar_falls_back_to_generated_at(self):
        from app_embed import catalog_stamp
        index = {"generatedAt": "2026-11-19T10:00:00Z"}
        expected = (datetime(2026, 11, 19, 10, 0, 0, tzinfo=timezone.utc).timestamp(), "index.generatedAt")
        for text in ("", "not json", "[]", "{}", '{"runStart": 1763581438}', '{"runStart": "yesterday"}'):
            with self.subTest(sidecar=text):
                self.sidecar(text)
                self.assertEqual(catalog_stamp(self.root, index), expected)

    def test_no_usable_source_gives_no_stamp(self):
        from app_embed import catalog_stamp
        self.assertEqual(catalog_stamp(self.root, {}), (None, "no stamp"))
        self.sidecar('{"runStart": "yesterday"}')
        self.assertEqual(catalog_stamp(self.root, {"generatedAt": "long ago"}), (None, "no stamp"))


class ReplacedRowsLeaveTheSampleTests(unittest.TestCase):
    """A picture newer than the catalog stamp leaves the parity sample and is re-embedded instead."""

    STAMP = 1000000.0
    OLD = 999000.0
    NEW = 1000001.0

    def setUp(self):
        from app_embed import image_key
        self.image_key = image_key
        self.rows = app_rows(40)
        self.vectors = stored_vectors(40)
        self.mtimes = {face(i): self.OLD for i in range(40)}

    def parity(self, embed_rows=matching_fresh):
        from app_embed import parity_step
        return parity_step(self.rows, self.vectors, set(), set(), self.image_key, self.mtimes.get, self.STAMP,
                           embed_rows, already_unit)

    def refresh(self):
        from app_embed import refresh_indices
        return refresh_indices(self.rows, self.vectors, set(), set(), self.image_key, self.mtimes.get, self.STAMP)

    def test_replaced_picture_is_skipped_and_refreshed(self):
        self.mtimes[face(7)] = self.NEW
        code, lines, pick, skipped = self.parity()
        self.assertEqual((code, skipped), (0, [7]))
        self.assertEqual(self.refresh(), [7])
        self.assertNotIn(7, pick)

    def test_the_sample_keeps_todays_unchanged_rows_and_tops_up_from_the_rest(self):
        changed = [7, 13, 21]
        for i in changed:
            self.mtimes[face(i)] = self.NEW
        today = random.Random(29).sample(list(range(40)), 30)
        kept = [i for i in today if i not in changed]
        code, _, pick, skipped = self.parity()
        self.assertEqual((code, skipped), (0, changed))
        self.assertEqual(len(pick), 30)
        self.assertEqual(set(pick) & set(changed), set())
        self.assertEqual([i for i in pick if i in set(kept)], kept)
        top_up = [i for i in pick if i not in set(today)]
        self.assertEqual(len(top_up), 30 - len(kept))
        self.assertTrue(set(top_up) <= set(range(40)) - set(today) - set(changed))

    def test_an_unchanged_failure_of_todays_draw_is_still_compared_and_fails(self):
        today = random.Random(29).sample(list(range(40)), 30)
        replaced, failing = today[0], today[1]
        self.mtimes[face(replaced)] = self.NEW
        code, lines, pick, skipped = self.parity(embed_rows=fresh_with_one_failure(failing))
        out = [text for stream, text in lines if stream == "stdout"]
        self.assertEqual((code, skipped, len(pick)), (2, [replaced], 30))
        self.assertIn(failing, pick)
        self.assertNotIn(replaced, pick)
        self.assertEqual(out[1], "parity below 0.995: 1 of 30")
        self.assertEqual(out[2], f"parity row id={self.rows[failing]['id']} cos=0.000000 image={face(failing)}")

    def test_no_stamp_or_a_stamp_after_every_picture_keeps_todays_draw(self):
        from app_embed import parity_step
        expected = random.Random(29).sample(list(range(40)), 30)
        for stamp in (None, self.NEW):
            with self.subTest(stamp=stamp):
                code, lines, pick, skipped = parity_step(self.rows, self.vectors, set(), set(), self.image_key,
                                                         self.mtimes.get, stamp, matching_fresh, already_unit)
                out = [text for stream, text in lines if stream == "stdout"]
                self.assertEqual((code, skipped, pick), (0, [], expected))
                self.assertIn("parity skipped 0 changed rows", out)

    def test_a_stamp_that_skips_rows_still_samples_thirty_and_is_repeatable(self):
        for i in range(6):
            self.mtimes[face(i)] = self.NEW
        code, _, pick, skipped = self.parity()
        self.assertEqual((code, skipped), (0, list(range(6))))
        self.assertEqual(len(pick), 30)
        self.assertEqual(len(set(pick)), 30)
        self.assertEqual(set(pick) & set(range(6)), set())
        self.assertEqual(self.parity()[2], pick)

    def test_picture_as_old_as_the_stamp_is_still_compared(self):
        self.mtimes[face(3)] = self.STAMP
        code, _, pick, skipped = self.parity()
        self.assertEqual((code, skipped), (0, []))
        self.assertEqual(pick, random.Random(29).sample(list(range(40)), 30))

    def test_healthy_run_names_no_skipped_row_and_states_the_rule(self):
        code, lines, pick, skipped = self.parity()
        out = [text for stream, text in lines if stream == "stdout"]
        self.assertEqual((code, skipped, len(pick)), (0, [], 30))
        self.assertIn("parity skipped 0 changed rows", out)
        stamp_line = ("parity stamp none; a row is skipped only when its image file is newer than the stamp,"
                      " an image replaced with its old modification time kept is still compared"
                      " and may still fail")
        self.assertEqual(out[-1], stamp_line)

    def test_at_most_ten_skipped_row_ids_are_named(self):
        from app_embed import parity_step
        rows, vectors = app_rows(60), stored_vectors(60)
        mtimes = {face(i): (self.NEW if i < 12 else self.OLD) for i in range(60)}
        code, lines, pick, skipped = parity_step(rows, vectors, set(), set(), self.image_key, mtimes.get, self.STAMP,
                                                 matching_fresh, already_unit)
        out = [text for stream, text in lines if stream == "stdout"]
        self.assertEqual((code, skipped, len(pick)), (0, list(range(12)), 30))
        self.assertIn("parity skipped 12 changed rows: 0 2 4 6 8 10 12 14 16 18", out)


class PerturbedEmbedderFailsTests(unittest.TestCase):
    """Rows whose picture has not changed are still compared, so a drifted embedder cannot pass unnoticed."""

    def test_fresh_vectors_that_do_not_match_still_return_code_two(self):
        from app_embed import image_key, parity_step
        rows, vectors = app_rows(40), stored_vectors(40)
        mtimes = {face(i): 1000.0 for i in range(40)}
        code, lines, pick, skipped = parity_step(rows, vectors, set(), set(), image_key, mtimes.get, 2000.0,
                                                 perturbed_fresh, already_unit)
        out = [text for stream, text in lines if stream == "stdout"]
        self.assertEqual((code, skipped, len(pick)), (2, [], 30))
        self.assertEqual(out[0], "parity n=30 min=0.000000")
        self.assertIn("parity below 0.995: 30 of 30", out)
        self.assertEqual(sum(1 for text in out if text.startswith("parity row id=")), 3)


class CannotGoQuietTests(unittest.TestCase):
    """Skipping stale rows may never leave the check comparing too little and passing silently."""

    STAMP = 1000000.0
    ROWS = 200

    def catalog(self, comparable, changed):
        """`comparable` rows hold a vector and a picture, `changed` of them newer than STAMP; the rest hold zeros."""
        rows, vectors = app_rows(self.ROWS), stored_vectors(self.ROWS)
        for i in range(comparable, self.ROWS):
            vectors[i] = 0
        mtimes = {face(i): self.STAMP + (1 if i < changed else -1) for i in range(comparable)}
        return rows, vectors, mtimes

    def assert_refused(self, comparable, changed, message):
        from app_embed import image_key, parity_step
        rows, vectors, mtimes = self.catalog(comparable, changed)
        code, lines, pick, skipped = parity_step(rows, vectors, set(), set(), image_key, mtimes.get, self.STAMP,
                                                 matching_fresh, already_unit)
        self.assertEqual((code, lines), (3, [("stderr", message)]))
        self.assertEqual((pick, len(skipped)), ([], changed))

    def test_too_few_unchanged_rows_left_after_skipping(self):
        # 15 skipped of 200 rows is under the 20 % limit, so only the 30-unchanged-rows floor can refuse here.
        self.assert_refused(40, 15, "parity: not enough unchanged rows: 25 eligible, 15 skipped")

    def test_too_many_rows_skipped_while_thirty_unchanged_remain(self):
        # 45 skipped is over 20 % of 200 rows, and 55 unchanged rows remain, so only the fraction can refuse here.
        self.assert_refused(100, 45, "parity: not enough unchanged rows: 55 eligible, 45 skipped")

    def test_too_few_comparable_rows_keeps_todays_refusal(self):
        from app_embed import image_key, parity_step
        rows, vectors, mtimes = self.catalog(20, 0)
        code, lines, pick, skipped = parity_step(rows, vectors, set(), set(), image_key, mtimes.get, self.STAMP,
                                                 matching_fresh, already_unit)
        self.assertEqual((code, lines), (2, [("stderr", "not enough parity rows")]))
        self.assertEqual((pick, skipped), ([], []))


class NothingChangedMeansNothingDifferentTests(unittest.TestCase):
    """With no picture newer than the stamp the draw must be exactly the one today's rule made."""

    def test_the_sample_matches_the_draw_of_the_list_before_the_stamp_rule(self):
        from app_embed import image_key, parity_step
        count = 210
        rows, vectors = app_rows(count), stored_vectors(count)
        for i in range(5, 60, 7):
            vectors[i] = 0
        noimage = {rows[i]["id"] for i in range(3, count, 11)}
        audit = {face(i) for i in range(7, count, 13)}
        missing = {face(i) for i in range(9, count, 17)}
        mtimes = {face(i): 500.0 for i in range(count) if face(i) not in missing}
        # the list drawn before the stamp rule
        old_eligible = [i for i, row in enumerate(rows)
                        if row["id"] not in noimage and np.any(vectors[i])
                        and image_key(row.get("image")) not in audit and image_key(row.get("image"))
                        and mtimes.get(image_key(row.get("image"))) is not None]
        code, lines, pick, skipped = parity_step(rows, vectors, noimage, audit, image_key, mtimes.get, 1000.0,
                                                 matching_fresh, already_unit)
        self.assertGreaterEqual(len(old_eligible), 30)
        self.assertEqual((code, skipped), (0, []))
        self.assertEqual(pick, random.Random(29).sample(old_eligible, 30))


if __name__ == "__main__":
    unittest.main()
