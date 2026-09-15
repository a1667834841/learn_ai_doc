# GitBook.com 发布说明

本仓库已按 GitBook Git Sync 格式整理，可直接导入 GitBook.com。

## 导入步骤

1. 登录 [GitBook](https://app.gitbook.com/)。
2. 创建组织或选择已有组织，创建一个 Documentation site。
3. 在内容来源中选择 GitHub，授权 GitBook GitHub App 访问 `a1667834841/learn_ai_doc`。
4. 选择 `main` 分支，项目目录选择仓库根目录 `./`。
5. GitBook 会读取根目录的 `.gitbook.yaml`、`README.md` 和 `SUMMARY.md`。
6. 发布站点后，在站点的 Customization 中选择主题、品牌色、Logo、明暗模式和 AI/MCP 设置。

## MCP 地址

发布后的站点会自动提供 MCP Server，地址为：

```text
https://你的-GitBook-站点域名/~gitbook/mcp
```

GitBook.com 的 Git Sync 会把仓库中的 Markdown 变更同步到站点；后续建议只在 GitHub 中维护 README、SUMMARY 和章节 Markdown，避免双向编辑冲突。
