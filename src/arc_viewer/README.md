# LichtFeld Arc Viewer (Experimental)

This package is the standalone SDL3/Vulkan PLY point viewer. It does not include
LichtFeld Studio training, editing, Python, plugins, MCP, or export workflows.
It is not a ROCm training target and does not provide a compute training backend.

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
`--gpu INDEX` to override it. No physical Intel Arc GPU has been validated;
only the compatible AMD Vulkan fallback has been validated.

The portable package includes `LICENSE.txt`, `THIRD_PARTY_LICENSES.md`, and the
exact SDL3, GLM, Vulkan loader, and Vulkan headers notices under `licenses`.
