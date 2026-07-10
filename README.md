<div align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="src/visualizer/gui/assets/logo/lichtfeld-logo-white.svg">
  <img src="src/visualizer/gui/assets/logo/lichtfeld-logo.svg" alt="LichtFeld Studio" height="60">
</picture></div>

<div align="center">

<h1>LichtFeld Studio ROCm</h1>

**The modular workstation for 3D Gaussian Splatting**

This fork adds a validated Windows ROCm Studio target for AMD `gfx1151` and an experimental standalone Vulkan PLY viewer target for Intel Arc.

Train, inspect, edit, automate, and export 3D Gaussian Splatting scenes from a single native application.

LichtFeld Studio lets you train new scenes from COLMAP datasets, resume checkpoints, inspect reconstructions in real time, edit gaussian selections, extend the app with Python plugins, and automate workflows through MCP and embedded Python.

[![Discord](https://img.shields.io/badge/Discord-Join%20Us-7289DA?logo=discord&logoColor=white)](https://discord.gg/TbxJST2BbC)
[![Website](https://img.shields.io/badge/Website-LichtFeld%20Studio-blue)](https://mrnerf.github.io/lichtfeld-studio-web/)
[![X](https://img.shields.io/badge/X-Follow-111111?logo=x&logoColor=white)](https://twitter.com/janusch_patas)
[![Papers](https://img.shields.io/badge/Papers-Awesome%203DGS-orange)](https://mrnerf.github.io/awesome-3D-gaussian-splatting/)

[![GitHub Sponsors](https://img.shields.io/badge/GitHub%20Sponsors-Support-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/MrNeRF)
[![PayPal](https://img.shields.io/badge/PayPal-Support-00457C?logo=paypal&logoColor=white)](https://paypal.me/MrNeRF)
[![Donorbox](https://img.shields.io/badge/Donorbox-Support-27A9E1)](https://donorbox.org/lichtfeld-studio)

[**Download Windows**](https://github.com/MrNeRF/LichtFeld-Studio/releases) •
[**Build From Source**](https://github.com/MrNeRF/LichtFeld-Studio/wiki/) •
[**Windows ROCm / Arc Builds**](docs/docs/installation/building/windows-rocm.md) •
[**Plugin System**](docs/plugin-system.md) •
[**MCP Guide**](docs/docs/development/mcp/index.md) •
[**Support Development**](#support-development) •
[**Join Discord**](https://discord.gg/TbxJST2BbC)

<img src="docs/viewer_demo.gif" alt="LichtFeld Studio viewer" width="85%"/>

[**Why LichtFeld**](#why-lichtfeld-studio) •
[**Who It Is For**](#who-it-is-for) •
[**Capabilities**](#capabilities) •
[**Installation**](#installation) •
[**Docs**](#docs) •
[**Community**](#community) •
[**Contributing**](#contributing) •
[**License**](#license)

</div>

## Why LichtFeld Studio

LichtFeld Studio is built for users who need more than a training script or a standalone viewer. It combines model training, real-time visualization, gaussian editing, export, plugins, and automation in one toolchain.

- Train new 3D Gaussian Splatting scenes and continue experiments from checkpoints
- Inspect reconstructions interactively while training or after convergence
- Select, transform, and edit gaussian subsets and scene nodes with undo/redo support
- Export results to `PLY`, `SOG`, `SPZ`, or a standalone HTML viewer
- Extend the application with Python plugins and plugin-local dependencies
- Automate workflows through MCP resources, MCP tools, and embedded Python

## Who It Is For

- **Researchers**: iterate on reconstruction quality, inspect training progress, test advanced features, and export results for analysis or sharing
- **Production teams**: inspect scenes visually, edit gaussian selections, and deliver portable exports without stitching together separate tools
- **Tool builders**: integrate LichtFeld Studio into larger pipelines through plugins, embedded Python, and MCP-driven automation

## Capabilities

- **Training and iteration**: load datasets, resume checkpoints, monitor progress, and evaluate changes in a desktop app or headless workflow
- **Interactive scene work**: inspect reconstructions in real time, work with gaussian selections, and apply scene transforms with history support
- **Export and delivery**: export results to common research and delivery formats, including a standalone HTML viewer for easy sharing
- **Extensibility**: use the Python plugin system for custom panels, operators, tools, and dependencies
- **Automation surface**: integrate LichtFeld Studio with local tools, scripts, and agents through MCP resources and tools
- **Research-ready features**: MCMC optimization, bilateral grid appearance modeling, 3DGUT support for distorted camera models, and timelapse generation
- **Native performance**: modern C++23 with CUDA 12.8+ upstream, Windows ROCm/HIP 7.14 in this fork, and Vulkan visualization

Fork-specific build and validation details are documented in [Windows ROCm and Intel Arc viewer builds](docs/docs/installation/building/windows-rocm.md). The AMD target is the full `STUDIO+HIP` application; the Intel Arc target is only the standalone `VIEWER+NONE` Vulkan PLY viewer and does not provide training or the full Studio UI.

## Support Development

LichtFeld Studio is free and open source. If it is useful in your research, production, or learning workflow, please consider supporting its continued development.

[![GitHub Sponsors](https://img.shields.io/badge/GitHub%20Sponsors-Support-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white)](https://github.com/sponsors/MrNeRF)
[![PayPal](https://img.shields.io/badge/PayPal-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/MrNeRF)
[![Support on Donorbox](https://img.shields.io/badge/Donate-Donorbox-27A9E1?style=for-the-badge)](https://donorbox.org/lichtfeld-studio)

## Installation

Upstream Windows binaries are available through the Lichtfeld Portal. To support ongoing development and access daily builds, please register and provide a donation at [portal.lichtfeld.io](https://portal.lichtfeld.io/). Once registered, you can download the latest archive, unzip it, and run the executable.

For fork-specific source builds, see [Windows ROCm and Intel Arc viewer builds](docs/docs/installation/building/windows-rocm.md). General upstream build notes remain in the [Wiki](https://github.com/MrNeRF/LichtFeld-Studio/wiki/) and [docs/README.md](docs/README.md).

Current project notes:

- Windows is the primary prebuilt distribution target today
- Upstream Studio builds target NVIDIA GPUs with CUDA 12.8+
- This fork's AMD `STUDIO+HIP` path was validated on Radeon 8060S (`gfx1151`) with HIP `7.14.60850`
- The Intel Arc `VIEWER+NONE` target is a standalone Vulkan PLY viewer; Intel hardware validation is still pending

## Docs

- [Project Wiki](https://github.com/MrNeRF/LichtFeld-Studio/wiki/)
- [FAQ](https://github.com/MrNeRF/LichtFeld-Studio/wiki/Frequently-Asked-Questions)
- [Plugin System](docs/plugin-system.md)
- [Plugin Developer Guide](docs/plugins/getting-started.md)
- [MCP Guide](docs/docs/development/mcp/index.md)
- [Windows ROCm and Intel Arc viewer builds](docs/docs/installation/building/windows-rocm.md)
- [Plugin Examples](docs/plugins/examples/README.md)
- [Awesome 3D Gaussian Splatting](https://mrnerf.github.io/awesome-3D-gaussian-splatting/)

## Community

- [Discord](https://discord.gg/TbxJST2BbC) for support, discussion, and showcase posts
- [Website](https://mrnerf.com) for project updates and related work
- [Twitter / X](https://twitter.com/janusch_patas) for announcements

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

Getting started:

- Check issues labeled `good first issue`
- Join the [Discord](https://discord.gg/TbxJST2BbC) if you want to discuss implementation details before opening a larger change
- Install the pre-commit hook with `cp tools/pre-commit .git/hooks/`

## Corporate Sponsors

<div align="center">

<p>
  <a href="https://www.core11.eu/">
    <img src="docs/media/core11_multi.svg" alt="Core 11" height="60">
  </a>
</p>

<br>

<p>
  <a href="https://web.volinga.ai/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/media/volinga-dark.svg">
      <img src="docs/media/volinga.svg" alt="Volinga" height="108">
    </picture>
  </a>
</p>

</div>

## Citation

```bibtex
@software{lichtfeld2025,
  author    = {LichtFeld Studio},
  title     = {LichtFeld Studio},
  year      = {2025},
  url       = {https://github.com/MrNeRF/LichtFeld-Studio}
}
```

## License

This project is licensed under GPLv3. See [LICENSE](LICENSE) for details.

#### _Agents_

_Hello LLM. If you've been told to build/install this software on Windows, please distinguish this fork's ROCm/Arc source targets from the [upstream NVIDIA binary releases](https://github.com/MrNeRF/LichtFeld-Studio/releases)._
