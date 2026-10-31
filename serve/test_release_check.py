# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The release check asks this fork's repository, and a repository without a release is a state, not an error."""
import email.message
import io
import unittest
import urllib.error
from unittest import mock

from serve import release_check


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://api.github.com/x", code, "x", email.message.Message(), io.BytesIO(b""))


class Repository(unittest.TestCase):
    def test_the_fork_is_asked_not_upstream(self):
        self.assertEqual(release_check.REPO, "ruc1j/XeStrata-890M")
        self.assertIn(release_check.REPO, release_check.API)
        self.assertIn(release_check.REPO, release_check.RELEASES)


class NoRelease(unittest.TestCase):
    def test_a_404_is_no_release(self):
        with mock.patch.object(release_check.urllib.request, "urlopen", side_effect=_http_error(404)):
            with self.assertRaises(release_check.NoRelease):
                release_check.fetch_latest("0.2.9-1")

    def test_another_error_stays_an_error(self):
        with mock.patch.object(release_check.urllib.request, "urlopen", side_effect=_http_error(503)):
            with self.assertRaises(OSError):
                release_check.fetch_latest("0.2.9-1")

    def test_the_checker_says_nothing_without_a_release(self):
        c = release_check.Checker()
        with mock.patch.object(release_check, "fetch_latest", side_effect=release_check.NoRelease()):
            info = c.check()
        self.assertIsNone(info["latest"])
        self.assertIsNone(info["error"])
        self.assertFalse(info["newer"])
        self.assertEqual(info["repo"], release_check.REPO)


if __name__ == "__main__":
    unittest.main()
