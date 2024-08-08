# 用户设置
git config --global user.name "用户名"      # 设置用户名
git config --global user.email "邮箱"       # 设置邮箱

# 初始化操作
git init                                   # 初始化仓库

# 查看状态
git status                                 # 查看当前状态

# 文件操作
git add 文件名                             # 将文件添加到暂存区
git commit -m "日志信息" 文件名              # 提交暂存区的文件到本地库
rm <file-name>                            # Unix/Linux/macOS 删除文件
del <file-name>                           # Windows 删除文件
git rm <file-name>                        # 从 Git 版本控制中移除文件
git commit -m "删除文件 <file-name>"       # 提交删除操作
git add <file-name>                       # 将新文件添加到暂存区

# 远程仓库操作
git remote add 别名 远程地址                # 添加远程仓库并起别名
git remote -v                              # 查看远程仓库别名
git push 别名 分支                         # 将本地分支推送到远程仓库
git clone 远程地址                         # 克隆远程仓库到本地
git push -u origin <branch-name>         # 推送新分支到远程仓库并设置追踪关系
git pull origin <branch-name>            # 拉取远程分支的最新更改
git fetch origin                         # 从远程获取最新的分支信息

# 分支操作
git branch                                # 查看本地所有分支
git branch -r                             # 查看远程所有分支
git branch -a                             # 查看所有本地和远程分支
git status                                # 查看当前所在分支和状态
git checkout -b <branch-name>              # 创建新分支并切换到该分支
git checkout <branch-name>                 # 切换到已有的分支
git fetch origin                          # 获取远程更新
git checkout -b <branch-name> origin/<branch-name>  # 从远程创建并切换到本地分支

# 分支关系和合并情况
git log --oneline --graph --decorate --all   # 查看分支的提交历史和图形化表示
git branch --merged                       # 查看已合并到当前分支的分支
git branch --no-merged                    # 查看未合并到当前分支的分支
git branch -vv                            # 查看分支的详细信息，包括上游分支
git branch -r --merged                    # 查看远程分支的合并情况

# 处理未解决的冲突
git status                               # 查看当前状态和未解决的冲突
git add <conflicted-file>                # 标记冲突文件为已解决
git commit                               # 提交合并结果

# 推送更改
git push origin <branch-name>            # 将更改推送到远程仓库
