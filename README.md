# ControlNet Dataset Curator

## Download / 下载

**[Click to Download / 点击下载](https://github.com/WhitecrowAurora/controlnet-dataset-curator/releases/download/v1.0.2/controlnet-dataset-curator-v1.0.2.zip)**

ControlNet Dataset Curator was originally a desktop GUI tool for NewBie projects, used for filtering, manually reviewing, exporting structured data, backing up and restoring data, and repairing ControlNet training data. Through iterations, it has now become a general-purpose dataset processing tool.

ControlNet Dataset Curator 原本是用于 NewBie 新项目处理 Controlnet 训练数据筛选、人工审核、结构化导出、备份恢复与数据修复的桌面 GUI 工具, 经过迭代现已成为通用数据集处理工具

It is designed for practical dataset curation workflows rather than model training itself.

它的目标是解决实际的数据整理流程问题，而不是直接做模型训练。

## Features | 功能

- Review and filter ControlNet training images
- Support Canny / Pose / Depth style outputs
- JSONA export with rolling backups
- JSONA import, validation, repair, and restore
- Missing-file checks and problem export
- Local image workflow and parquet / dataset extraction workflow
- PyQt5 desktop GUI
- Supports CPU and GPU processing


- 支持 ControlNet 训练图像的筛选与审核
- 支持 Canny / Pose / Depth 三类输出流程
- 支持 JSONA 导出与滚动备份
- 支持 JSONA 导入、核对、修复、恢复
- 支持缺失文件检查与问题项导出
- 支持本地图像流程与 parquet / 数据集提取流程
- 基于 PyQt5 的桌面 GUI
- 支持CPU与GPU处理

## Repository Contains | 仓库内容

This repository contains the source code of the project.

这个仓库主要包含项目源码。

Included:

- `controlnet_gui/`
- `main.py`
- `config.json`
- `requirements.txt`
- helper scripts such as `run.bat`, `cleanup.bat`, and `fix_pytorch.bat`

包含内容：

- `controlnet_gui/`
- `main.py`
- `config.json`
- `requirements.txt`
- `run.bat`、`cleanup.bat`、`fix_pytorch.bat` 等辅助脚本

Not included in the Git repository by default:

- embedded portable Python runtime
- downloaded model files
- local outputs
- local cache / temporary files / backups

默认不会放进 Git 仓库的内容：

- 嵌入式便携 Python 运行时
- 已下载模型文件
- 本地输出结果
- 本地缓存 / 临时文件 / 备份文件

## Why The Embedded Python Runtime Is Not Committed | 为什么不把嵌入式 Python 提交到仓库

This project is often distributed as a portable desktop application with an embedded Python runtime.

这个项目在实际分发时，经常会以“带嵌入式 Python 的便携桌面程序”形式发布。

However, the embedded runtime is not committed to this Git repository on purpose.

但嵌入式运行时默认不会提交到这个 Git 仓库，这是有意的设计。

Reasons:

- it makes the repository much larger
- it adds many third-party binaries that are not part of the project source code

原因：

- 会让仓库体积明显膨胀
- 会引入大量不属于项目源码的第三方二进制文件

## Development Setup | 开发环境方式

For development, use a normal Python environment.

开发时，建议使用普通 Python 环境，而不是直接依赖便携嵌入式环境。

Example:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

示例：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Notes:

- some features require optional dependencies
- some GPU-related features depend on your local CUDA / driver environment
- some model-specific functions may need additional model downloads

说明：

- 某些功能依赖可选依赖包
- 某些 GPU 相关功能依赖本机 CUDA / 驱动环境
- 某些模型相关功能可能需要额外下载模型文件

## Portable Runtime Workflow | 便携运行方式

For end users, the project can also be packaged with an embedded Python runtime.

对于最终用户，这个项目也可以打包成带嵌入式 Python 的便携版本。

Typical portable package layout:

- `controlnet_gui/`
- `python/`
- `main.py`
- `run.bat`
- `config.json`
- optional `models/`

便携包常见结构：

- `controlnet_gui/`
- `python/`
- `main.py`
- `run.bat`
- `config.json`
- 可选 `models/`

In the portable package, `run.bat` uses the embedded runtime under `python/`.

在便携包中，`run.bat` 会直接使用 `python/` 目录下的嵌入式运行时启动程序。

## Release Recommendation | Release 发布建议

If you want to publish a ready-to-use build, use GitHub Releases instead of committing the embedded Python runtime into the main source repository.

如果你想发布“用户下载即可运行”的版本，建议使用 GitHub Releases，而不是把嵌入式 Python 直接提交到主源码仓库。

Suggested release contents:

- source files required to run the app
- embedded `python/` runtime
- optional `models/` if you want a more complete offline package
- startup scripts such as `run.bat`

建议放进 Release 的内容：

- 运行程序所需源码文件
- 嵌入式 `python/` 运行时
- 如果你想提供更完整离线包，可附带 `models/`
- `run.bat` 等启动脚本

## JSONA Support | JSONA 支持

The project includes JSONA-oriented workflow tools for dataset management.

项目包含面向 JSONA 的数据管理工作流工具。

Supported operations:

- export accepted entries
- rolling backup snapshots
- import external JSONA files
- validate file structure and file existence
- restore backups
- conservative repair for invalid or duplicate entries

支持的操作：

- 导出已确认条目
- 滚动备份快照
- 导入外部 JSONA 文件
- 核对文件结构与路径存在性
- 从备份恢复
- 对异常或重复条目进行保守修复

## Project Status | 项目状态

This is an actively iterated practical tool.

这是一个偏实用导向、持续迭代中的工具。

The codebase prioritizes workflow reliability and dataset operations.

代码重点放在工作流可靠性和数据集操作能力上。

## License | 许可证

This project is licensed under the GNU General Public License v3.0.

本项目采用 GNU General Public License v3.0 许可证。

See the [LICENSE](LICENSE) file for details.

详细内容请参见 [LICENSE](LICENSE)。

## Acknowledgements | 致谢

Thanks to all the volunteers of the Newbie project's Controlnet dataset for testing the app and helping improve stability during development.

感谢Newbie项目Controlnet数据集全体志愿者参与测试，并帮助改进稳定性。

This project builds on the Python ecosystem around PyQt5, PyTorch, OpenCV, Hugging Face tooling, and ControlNet-related utilities.

本项目构建在 PyQt5、PyTorch、OpenCV、Hugging Face 工具链以及 ControlNet 相关 Python 生态之上。
