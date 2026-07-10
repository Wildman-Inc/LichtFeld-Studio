# LichtFeld Studio ROCm Arc Viewer

This package is the standalone SDL3/Vulkan PLY point viewer. It does not include
LichtFeld Studio training, editing, Python, plugins, MCP, or export workflows.

## Run

```powershell
.\LichtFeld-Studio-Arc-Viewer.exe C:\data\scene.ply
.\LichtFeld-Studio-Arc-Viewer.exe --gpu 0 C:\data\scene.ply
```

You can also drop a PLY file onto the window. Use `--help` for the complete
command-line and control reference.

The viewer supports PLY 1.0 ASCII and `binary_little_endian` files, explicit RGB
properties, and Gaussian splat `f_dc_0..2` SH colors. `binary_big_endian` is not
supported. Scale, rotation, opacity, and higher-order SH coefficients are not
rendered by this point-viewer target.

Automatic device selection prefers a compatible Intel discrete GPU. Use
`--gpu INDEX` to override it. Intel Arc hardware validation is still pending;
the compatible Vulkan fallback has been tested on AMD hardware.

See `docs/windows-rocm.md` for build and validation details. The project license
is `LICENSE.txt`; dependency notices are under the `licenses` directory.
