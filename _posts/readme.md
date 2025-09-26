---
layout:     post
title:      read me for 4090
subtitle:   temporarily located in hhb706
date:       2025-09-26
co-author:     AYU
header-img: img/post-bg-2015.jpg
catalog: true
tags:
    - 算力操作文档
    - GPU4090
    - 香港理工大学
    - HHB
---
# Magical document for first time Ubuntu user
## about remove home icons in desktop:
install an extension and remove.`sudo apt install gnome-shell-extension-prefs`
I didn't find any related 

## About SSD and SATA storage
This is the first day I gain access to this ubuntu device. It gets 4TB SSD+SATA waiting to be mounted. If in the future the storage is almost full. Please try to mount the 4TB disk into file explorer.

1. Check the disk information by using `sudo lsblk` or `sudo fdisk -l`

2. Mount the disk into /mnt directory using `sudo mount /dev/sda /mnt`

3. Or unmount it by using `sudo umount /mnt`

4. Or mount by search for the disk app, use `+` button to create your partion (mine for 1024GB) in `/media/user/zhaiyu`.

**But the thing is when I reboot the mounting partion path changed into mnt path with this long ids, do you know why?**

I believe there is no difference between /media and /mnt.


This file could be read by using `cat >readme.txt`. Clicking on `ctrl+D` will allow you to save the file and quit.

[reference site](https://askubuntu.com/questions/1357183/how-to-save-hard-disk-data)

## About SSH server

`sudo apt-get install openssh-server`
`ps -e |grep ssh`

After these two commands you could see sshd which means the server has run.

use `sudo systemctl status ssh` to check status

use `ssh user@153.xxx.xxx.xxx -p 22` to connect to ssh in your windows command

## About VS Code Tunnel

I am sure SSH server could be connected in campus wifi but not in mainland or other network. Therefore you could use `./code tunnel` to login into VS Code tunnel and connect anywhere.


## About cuda and pytorch

Just be careful about these versions.

You can use `???` or go to settings about.

## About auto=sleep, power, suspend and screen lock
`sudo vim /etc/systemd/logind.conf`to open the setting for the rebooting and `lidswitch`.

Delete `#` first, change `HandleLidSwitch` from `suspend` into `ignore`

`ESC+:wq` To save the file and exit.

## About disable auto-start service

`systemctl list-unit-files` check all the current service status

use `systemctl status xxx.service` to check specific service's status

`state`: enable/disable/generated/masked

`preset`: status when install or default status

use `systemctl stop xxx.service` to stop the service

to disable service! use `systemctl disable xxx.service`

**Anydesk is enabled now but if you don't need it no more please disable it.** --2025.09.26

## About downloading huggingface models:

**existing LLM models:**

- Llama-2-7b-chat 
- Llama-3-7b-instruct 
- Lllama-3.1-7b-instruct
- Qwen2-7b-instruct

use `pip install huggingface_hub` to invoke `huggingface-cli`

first use `huggingface-cli login` to paste your huggingface token into the popped up window.

then use `huggingface-cli download <repo-id> --include "*.<include-filetype>" --exclude "*.<exclude-file-type>" --local-dir </path to save the model>`
`huggingface-cli download Qwen/Qwen3-32B --local-dir /media/user/zhaiyu/llms/qwen`
change content in `<>` into your own.

download dataset by `huggingface-cli download HuggingFaceH4/ultrachat_200k --repo-type dataset`

## About CUDA install 

(This website is a good tutorial for cuda and cudann and pytorch installation)[https://www.felixsanz.dev/articles/compatibility-between-pytorch-cuda-and-xformers-versions]

use `nvidia-smi` to check on the right up corner whether cuda is installed!!!

use `nvcc --version` to check whether your cuda has installed

if not installed, (This website is for offical cuda installation)[https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_network]

choose `Linux+x86_64+Ubuntu+24.04+deb(network)+Base Installer`

and then reboot

## About use conda env in jupyter kernel

use `conda install -n <env-name> ipykernel`

and `conda activate <env-name>`

and `python -m ipykernel install --user --name <env-name> --display-name "<env-name>"`

then you can choose your env in the upperfront button of jupyter!

## About scp file download from another server

upload data to server:
`scp -rP <port> <local folder> root@xxxx`

从远程复制到本地，只要将从本地复制到远程的命令的后2个参数调换顺序
`scp root@www.runoob.com:/home/root/others/music /home/space/music/1.mp3 `
`scp -r www.runoob.com:/home/root/others/ /home/space/music/`

如果远程服务器防火墙有为scp命令设置了指定的端口，我们需要使用 -P 参数来设置命令的端口号，命令格式如下： 
`scp -P 4588 remote@www.runoob.com:/usr/local/sin.sh /home/administrator`
+iTAeIVz7Vf0
AUTODL:
scp -rP duankou <>

## About nvidia-smi

You can monitor it dynamicly by using `watch -n 1 nvidia-smi`

## about npm and node

I did not change anything of npm cache and global. you can check the version with `npm -v` and `node -v`

## github and git token

Name:Yuzyzy 

Personal Access Token:
- readnwrite available in 2025.03.16 - 2025.04.14 :
`github_pxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

how to use: change your password for this token.

### How to add and push to the remote?

Steps:
1. under some project folder, command input `git init`

2. in the remote github, create the repo and copy the repo http url.

3. change the code, and command input `git add .` (git add all file)

4. `git commit -m "explain why commit some changes"`

5. `git push -u origin master` or `git push -u origin main`

 - Use `git branch to check for the current branch name`


## OLLAMA_SERVICE

### Inint: 
Run `ollama serves` and `ollama run model_name`, they will load from the downloaded file.

### Download
Download models: `ollama run model_name`

#### Change download path?
TODO: Change the model_cache_path 

#### Run with different model types?
TODO: run the model by using safetensor files?

#### Check the running models
`ollama list` or `ollama ps`

#### TODO: Change the listening port?



In file `\etc\systemd\system\ollama.service` I delete the line `Environment="PATH=/home/user/anaconda3/envs/llm/bin:/home/user/.nvm/versions/node/v16.20.2/bin:/home/user/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin"`


## Remote tunnel singleton locked issue for VSCODE tunnels
find `/home/user/.vscode/cli/tunnel-stable.lock` and temporarily detele this file or cut this file. Everything will work. `https://github.com/microsoft/vscode-remote-release/issues/9806`

## Stop the automatic update of the ubuntu kernel and software:
You can search for software & update and go for update to set the update to never.
You can also set in the command line according to [this link](https://blog.csdn.net/fighting_88412/article/details/150584233) 

` uname -r`

`dpkg --get-selections |grep your-version-number`

`sudo apt-mark hold everything-shows-in-the-command-line`
