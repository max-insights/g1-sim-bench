# 官方库的本地改动（备份）

| 文件 | 官方仓库 | 基线 | 正本在哪 |
|---|---|---|---|
| GR00T-WholeBodyControl_45c03fa_g1-local.patch | NVlabs/GR00T-WholeBodyControl | 45c03fa | 工作站仓库的 `g1-local` 分支 |
| xr_teleoperate_817fb00_worktree.diff | unitreerobotics/xr_teleoperate | 817fb00 | 工作区未提交改动（已不在当前链路） |
| xr_teleoperate_televuer_worktree.diff | televuer 子模块 | 见文件首行 | 工作区未提交改动 |

恢复 GR00T：在干净的 45c03fa 上 `git am GR00T-WholeBodyControl_45c03fa_g1-local.patch`。
