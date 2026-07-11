# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression checks for Linux system headers used with HIP compatibility headers."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_native_file_dialog_loads_glib_before_hip_transitive_headers():
    source = (
        PROJECT_ROOT / "src/visualizer/gui/utils/native_file_dialog.cpp"
    ).read_text(encoding="utf-8")

    gtk_include = source.index("#include <gtk/gtk.h>")
    hip_transitive_include = source.index('#include "io/formats/colmap.hpp"')

    assert gtk_include < hip_transitive_include
