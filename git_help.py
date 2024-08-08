#设置用户
git config --global user.name ""
git config --global user.email ""

#初始化
git init

#查看
git status

#将文件提交至暂存区
git add 文件名

#将暂存区的文件提交至本地库
git commit -m "日志信息" 文件名

#给远程仓库起别名
git remote add 别名 远程地址

#查看别名
git remote -v

#将本地文件上传至Github
git push 别名 分支

#从远端仓库下载文件
git clone 远程地址
