# 全新 Windows 电脑安装 Codex 前置步骤

更新时间：2026-05-20

本文面向一台全新的 Windows 电脑，从网络准备、`FlClash`、`Node.js` 到 Codex App / CLI / IDE 扩展，整理完整安装包下载入口、前置环境、安装命令和常见问题。

## 先看结论

推荐安装顺序：

1. 更新 Windows，并确认 `winget` 可用。
2. 安装 FlClash，并导入你自己的代理订阅。
3. 打开 FlClash 的系统代理，确认浏览器和终端能访问 OpenAI 相关服务。
4. 安装 Git for Windows。
5. 安装 Node.js 24 LTS，并配置 npm 国内镜像。
6. 安装 Codex Windows App。
7. 安装 Codex CLI。
8. 按需要安装 VS Code / Cursor / JetBrains 的 Codex IDE 扩展。
9. 首次启动 Codex，登录 ChatGPT 账号或 OpenAI API Key。
10. 第一次运行本地 Agent 时，允许 Codex 完成 Windows 沙盒初始化。

## 安装包下载清单

| 用途 | 安装包 / 工具 | 下载方式 | 是否通常不需要代理 | 备注 |
| --- | --- | --- | --- | --- |
| 网络代理客户端 | FlClash Windows 安装包 | [GitHub Releases](https://github.com/chen08209/FlClash/releases/latest)；当前可找 `FlClash-0.8.92-windows-amd64-setup.exe` | GitHub 不稳定 | 新电脑还没有代理时，建议提前从可访问设备下载好安装包拷到 U 盘。 |
| Node.js 运行时 | Node.js 24 LTS Windows x64 MSI | [Node.js 官方下载](https://nodejs.org/en/download)；国内镜像：[npmmirror node binaries](https://npmmirror.com/mirrors/node/) | 是 | Codex CLI 通过 npm 安装，Node.js 必装。2026-05-20 推荐 24 LTS。 |
| Codex 桌面端 | Codex Windows App | [官方 Windows 下载页](https://get.microsoft.com/installer/download/9PLM9XGG6VKS?cid=website_cta_psi) | 是 | OpenAI 官方文档跳转到 Microsoft 下载。适合日常多线程、工作树、Review、自动化。 |
| Codex 命令行 | `@openai/codex` npm 包 | `npm i -g @openai/codex@latest --registry=https://registry.npmmirror.com` | 是 | 适合 PowerShell / Windows Terminal / 脚本使用。 |
| 指定版本 CLI | `@openai/codex@0.132.0` | `npm i -g @openai/codex@0.132.0 --registry=https://registry.npmmirror.com` | 是 | 2026-05-20 官方 changelog 最新版本。 |
| VS Code 扩展 | Codex IDE extension | [Visual Studio Marketplace](https://marketplace.visualstudio.com/) 里搜索 Codex | 多数情况下是 | 官方文档说明支持 VS Code、Cursor、Windsurf 以及 JetBrains。 |
| Chrome 扩展 | Codex Chrome extension | Codex App -> Plugins -> Chrome plugin -> 跟随引导安装 | 不稳定 | 只有需要 Codex 操作已登录的 Chrome 网页时才装。普通本地预览优先用 Codex 内置浏览器。 |

说明：

- 这张表把 Codex 能跑起来所需的外部安装包也放进来了：FlClash 负责网络连通性，Node.js 负责 npm / Codex CLI。
- Codex Windows App 的官方入口在 OpenAI 文档中标为 “Download for Windows”，实际跳转到 `get.microsoft.com`。
- Codex CLI 官方安装命令是 `npm i -g @openai/codex`。在国内装机时建议加 `--registry=https://registry.npmmirror.com`。
- 如果你只是想用桌面 App，理论上可以不单独安装 CLI；但建议装 CLI，因为 App、CLI、IDE 扩展会共享一部分配置和会话体验。

## 电脑系统要求

推荐：

- Windows 11，完整更新到最新补丁。
- Windows 10 也能尽量使用，但官方只按 “best effort” 支持；Windows 10 需要 1809 或更新版本，太旧会缺少现代控制台能力。
- `winget` 应该可用。如果 PowerShell 里输入 `winget --version` 没反应，先更新 Microsoft Store 里的 “应用安装程序 / App Installer”，或通过 Windows 更新补齐。
- 当前 Windows 用户最好有管理员权限。Codex 原生 Windows 沙盒的推荐模式需要管理员批准一次初始化。

检查命令：

```powershell
winver
winget --version
```

## 基础工具安装步骤

### 1. FlClash

用途：让新电脑先具备稳定访问 OpenAI、GitHub、npm 官方源等服务的网络能力。

下载：

- 官方项目：[https://github.com/chen08209/FlClash](https://github.com/chen08209/FlClash)
- 最新发布页：[https://github.com/chen08209/FlClash/releases/latest](https://github.com/chen08209/FlClash/releases/latest)
- Windows x64 安装包文件名通常类似：`FlClash-0.8.92-windows-amd64-setup.exe`

新电脑还没有代理时：

- 优先在另一台已经能访问 GitHub 的电脑上下载 Windows x64 `setup.exe` 安装包，再用 U 盘或局域网拷到新电脑。
- 如果 GitHub 直连能打开，就直接从 Releases 下载。
- 不建议从来路不明的网盘或二次打包站下载代理客户端，除非你能确认文件来源和哈希。

安装和配置：

1. 双击 `FlClash-*-windows-amd64-setup.exe` 安装。
2. 打开 FlClash。
3. 导入你自己的订阅链接或配置文件。
4. 选择可用节点。
5. 打开 `System Proxy` 或 “系统代理”。
6. 浏览器测试能否打开 OpenAI、GitHub、npm 等网站。
7. 如果终端不走系统代理，再单独给 PowerShell 设置代理环境变量。

PowerShell 临时代理示例：

```powershell
$env:HTTP_PROXY="http://127.0.0.1:7890"
$env:HTTPS_PROXY="http://127.0.0.1:7890"
```

端口以 FlClash 设置页显示为准，常见是 `7890`、`7897` 或其他本地端口。

验证：

```powershell
curl.exe https://api.openai.com/v1/models
```

如果返回认证错误但不是连接超时，说明网络已经通了；没有 API Key 时出现 401/403 属于正常现象。



### 2. Git for Windows

用途：让 Codex 能读仓库状态、看 diff、创建分支、提交、使用 worktree。

下载：

- 官方入口：[https://git-scm.com/download/win](https://git-scm.com/download/win)
- 国内镜像入口：[https://registry.npmmirror.com/binary.html?path=git-for-windows/](https://registry.npmmirror.com/binary.html?path=git-for-windows/)

安装时建议：

- 默认编辑器随意，后续可以用 VS Code。
- 勾选把 Git 加入 PATH。
- 终端选择 MinTTY 或 Windows Terminal 都可以；Codex 原生 Windows 推荐 PowerShell / Windows Terminal。

验证：

```powershell
git --version
```

### 3. VS Code 或其他 IDE

用途：配合 Codex IDE 扩展，直接在编辑器里让 Codex 读取打开的文件、选择区和项目上下文。

下载：

- VS Code 官方：[https://code.visualstudio.com/Download](https://code.visualstudio.com/Download)
- VS Code 国内镜像下载页通常会自动走 Microsoft CDN，直连大多可用。

验证：

```powershell
code --version
```

如果 `code` 命令不可用，在 VS Code 里打开命令面板，搜索 `Shell Command: Install 'code' command in PATH`，或重新安装时勾选加入 PATH。

### 4. Windows Terminal

用途：比老版 PowerShell 控制台更稳定，适合跑 Codex CLI。

下载：

- Microsoft Store 搜索 “Windows Terminal”
- 或 PowerShell 执行：

```powershell
winget install --id Microsoft.WindowsTerminal -e
```

### 5. Microsoft Visual C++ Redistributable / Build Tools

Codex CLI 0.132.0 起官方说明 Windows MSVC release binaries 不再依赖额外安装的 VC++ runtime DLL。也就是说，普通使用 Codex CLI 时通常不用单独装 VC++ 运行库。

但如果 Codex IDE 扩展无响应，官方 Windows FAQ 仍建议安装：

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools -e
```

安装器里选择 C++ workload。一般用户先不用装，遇到 IDE 扩展问题再装。

### 6. WSL2，可选

如果你的项目依赖 Linux 工具链，或者原生 Windows 沙盒在公司电脑上被策略拦住，可以使用 WSL2。

安装：

```powershell
wsl --install
```

重启后进入 WSL，再在 WSL 里安装 Node.js 和 Codex：

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash
nvm install 22
npm i -g @openai/codex
codex
```

国内环境下，`curl raw.githubusercontent.com` 可能需要代理。没有 Linux 需求时，优先用 Windows 原生 Codex。

## 安装 Node.js 和 npm

推荐 Node.js 24 LTS。Node.js 官方发布计划显示 24.x 是当前 Active LTS，22.x 是 Maintenance LTS；新电脑装机优先用 24 LTS。

下载方式：

- 官方下载页：[https://nodejs.org/en/download](https://nodejs.org/en/download)
- 国内镜像目录：[https://npmmirror.com/mirrors/node/](https://npmmirror.com/mirrors/node/)

手动下载镜像时，选择类似下面的文件：

```text
v24.x.x/node-v24.x.x-x64.msi
```

也可以用 winget 安装 LTS：

```powershell
winget install --id OpenJS.NodeJS.LTS -e
```

检查：

```powershell
node -v
npm -v
```

设置 npm 国内镜像：

```powershell
npm config set registry https://registry.npmmirror.com
npm config get registry
```

## 安装 Codex Windows App

1. 打开官方 Windows 下载页：

   [https://get.microsoft.com/installer/download/9PLM9XGG6VKS?cid=website_cta_psi](https://get.microsoft.com/installer/download/9PLM9XGG6VKS?cid=website_cta_psi)

2. 下载并安装。
3. 打开 Codex。
4. 使用 ChatGPT 账号登录，或使用 OpenAI API Key 登录。
5. 选择一个项目目录。
6. 第一次运行本地 Agent 时，选择 Local，并允许必要的管理员弹窗完成沙盒设置。

注意：

- 使用 API Key 登录时，部分功能例如 cloud threads 可能不可用。
- Codex App 适合日常使用；CLI 更适合终端、脚本和自动化。

## 安装 Codex CLI

推荐安装最新版：

```powershell
npm i -g @openai/codex@latest --registry=https://registry.npmmirror.com
```

如果要固定到本文核对时的最新版：

```powershell
npm i -g @openai/codex@0.132.0 --registry=https://registry.npmmirror.com
```

验证：

```powershell
codex --version
codex doctor
```

启动：

```powershell
cd D:\your-project
codex
```

升级：

```powershell
npm i -g @openai/codex@latest --registry=https://registry.npmmirror.com
```

如果安装后提示缺少 Windows 可选依赖，先重装并确保 optional dependencies 没被禁用：

```powershell
npm config set optional true
npm uninstall -g @openai/codex
npm i -g @openai/codex@latest --include=optional --registry=https://registry.npmmirror.com
codex --version
```

## 安装 IDE 扩展

### VS Code / Cursor / Windsurf

1. 打开扩展商店。
2. 搜索 `Codex`。
3. 安装 OpenAI 的 Codex 扩展。
4. 重启 IDE。
5. 右侧边栏打开 Codex。
6. 登录 ChatGPT 账号或 API Key。

官方说明 Codex IDE extension 支持 VS Code forks，例如 Cursor 和 Windsurf。

### JetBrains

如果使用 Rider、IntelliJ IDEA、PyCharm、WebStorm 等，按官方文档跳转 JetBrains 安装页安装 Codex 集成。

## 首次使用前的账号准备

二选一：

1. ChatGPT 账号：Plus、Pro、Business、Edu、Enterprise 等计划包含 Codex 使用权限，具体以 OpenAI 当前账号页面为准。
2. OpenAI API Key：在 [OpenAI API Dashboard](https://platform.openai.com/) 创建 API Key。

国内网络提醒：

- 安装包下载可以尽量用 Microsoft、npmmirror、Git 镜像等直连来源。
- 但登录 ChatGPT、调用 OpenAI API、Codex 访问模型服务时，网络连通性仍取决于你所在网络环境。
- 不要把 API Key 发给别人，也不要写进公开仓库。

## Windows 沙盒建议

Codex 原生 Windows 模式有两种沙盒：

- `elevated`：推荐模式。需要管理员批准初始化，隔离更强。
- `unelevated`：兜底模式。不需要完整管理员设置，但隔离弱一些。

配置文件通常在：

```text
%USERPROFILE%\.codex\config.toml
```

推荐配置：

```toml
[windows]
sandbox = "elevated"
```

如果公司电脑策略拦截，临时改成：

```toml
[windows]
sandbox = "unelevated"
```

如果沙盒不能读取某个目录，在 Codex 对话里输入：

```text
/sandbox-add-read-dir C:\absolute\directory\path
```

## 建议的完整装机命令清单

先手动安装 FlClash：

```text
https://github.com/chen08209/FlClash/releases/latest
```

导入订阅、开启系统代理后，再继续下面的步骤。

PowerShell 管理员窗口：

```powershell
winget --version
winget install --id Git.Git -e
winget install --id OpenJS.NodeJS.LTS -e
winget install --id Microsoft.VisualStudioCode -e
winget install --id Microsoft.WindowsTerminal -e
```

普通 PowerShell 窗口：

```powershell
node -v
npm -v
npm config set registry https://registry.npmmirror.com
npm i -g @openai/codex@latest --registry=https://registry.npmmirror.com
codex --version
codex doctor
```

然后安装 Codex Windows App：

```text
https://get.microsoft.com/installer/download/9PLM9XGG6VKS?cid=website_cta_psi
```

## 最小验证流程

新建一个测试目录：

```powershell
mkdir D:\codex-test
cd D:\codex-test
git init
codex
```

在 Codex 里输入：

```text
创建一个 hello.js，运行它，并解释你做了什么。
```

期望结果：

- Codex 能创建文件。
- Codex 能请求或运行 `node hello.js`。
- `git status` 能看到新增文件。
- 如果需要联网安装依赖，Codex 会请求权限，而不是静默越权。

## 常见问题

### `codex` 命令找不到

检查 npm 全局目录：

```powershell
npm prefix -g
npm bin -g
```

确认 npm 全局 bin 目录在系统 PATH 中。修改 PATH 后重开 PowerShell。

### npm 下载很慢或失败

重新设置 registry：

```powershell
npm config set registry https://registry.npmmirror.com
npm cache clean --force
npm i -g @openai/codex@latest --registry=https://registry.npmmirror.com
```

### Codex 沙盒初始化失败

优先处理：

1. 重新启动 Codex。
2. 再次允许管理员弹窗。
3. 如果是公司电脑，确认本机策略允许创建本地用户/组、修改防火墙规则和配置沙盒用户登录权限。
4. 临时使用 `unelevated` 沙盒继续工作。

### IDE 扩展装了但没反应

1. 重启 VS Code / Cursor。
2. 确认 Codex 图标没有被隐藏。
3. 安装 Visual Studio Build Tools C++ workload。
4. 再重启 IDE。

## 参考来源

- OpenAI Codex App 文档：[https://developers.openai.com/codex/app](https://developers.openai.com/codex/app)
- OpenAI Codex CLI 文档：[https://developers.openai.com/codex/cli](https://developers.openai.com/codex/cli)
- OpenAI Codex Windows 文档：[https://developers.openai.com/codex/windows](https://developers.openai.com/codex/windows)
- OpenAI Codex IDE extension 文档：[https://developers.openai.com/codex/ide](https://developers.openai.com/codex/ide)
- OpenAI Codex Chrome extension 文档：[https://developers.openai.com/codex/app/chrome-extension](https://developers.openai.com/codex/app/chrome-extension)
- OpenAI Codex changelog：[https://developers.openai.com/codex/changelog](https://developers.openai.com/codex/changelog)
- Node.js 下载页：[https://nodejs.org/en/download](https://nodejs.org/en/download)
- Node.js 发布计划：[https://nodejs.org/en/about/releases/](https://nodejs.org/en/about/releases/)
- FlClash GitHub 项目：[https://github.com/chen08209/FlClash](https://github.com/chen08209/FlClash)
- FlClash Releases：[https://github.com/chen08209/FlClash/releases](https://github.com/chen08209/FlClash/releases)
