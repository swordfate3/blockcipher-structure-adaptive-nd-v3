# 工具边界

用户训练入口是安装后的 `cipher-experiments` CLI，不是tools目录中的批次脚本。
当前不需要额外运维脚本；Git同步和SSH/SCP结果传输在训练程序外由用户操作。
不要将旧backup中的runner、审批卡或机器专属脚本复制到这里。
