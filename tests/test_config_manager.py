import os
import tempfile
import unittest

from config_manager import (
    normalize_save_settings, platform_default_recording_directory,
    resolve_recording_directory,
)


class RecordingDirectoryTests(unittest.TestCase):
    def test_default_is_records_beside_application_on_macos(self):
        with tempfile.TemporaryDirectory() as application_directory:
            result = platform_default_recording_directory(
                platform="darwin", application_directory=application_directory)
            self.assertEqual(result, os.path.join(application_directory, "records"))

    def test_default_is_records_beside_application_on_windows(self):
        with tempfile.TemporaryDirectory() as application_directory:
            result = platform_default_recording_directory(
                platform="win32", application_directory=application_directory)
            self.assertEqual(result, os.path.join(application_directory, "records"))

    def test_auto_and_legacy_default_resolve_to_records(self):
        with tempfile.TemporaryDirectory() as application_directory:
            expected = os.path.join(application_directory, "records")
            for configured in ("auto", "./records", ".\\records", "./recordings"):
                self.assertEqual(
                    resolve_recording_directory(
                        configured, application_directory=application_directory),
                    expected)

    def test_explicit_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as explicit_directory:
            self.assertEqual(
                resolve_recording_directory(explicit_directory),
                os.path.abspath(explicit_directory))

    def test_normalize_save_settings_creates_directory(self):
        with tempfile.TemporaryDirectory() as root:
            destination = os.path.join(root, "new", "records")
            location, filename = normalize_save_settings(destination, "sample-01")
            self.assertEqual(location, destination)
            self.assertEqual(filename, "sample-01")
            self.assertTrue(os.path.isdir(destination))

    def test_normalize_save_settings_rejects_path_in_filename(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                normalize_save_settings(root, "nested/sample")


if __name__ == "__main__":
    unittest.main()
