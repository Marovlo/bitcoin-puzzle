# 协调者部署指南（2C4G Linux 服务器）

## 前置条件

- Linux x64（Ubuntu 20.04+ / CentOS 7+）
- Go 1.21+（用于编译）
- gcc（SQLite 的 cgo 依赖）

## 安装步骤

```bash
# 1. 安装 Go（如果没有）
wget https://go.dev/dl/go1.22.4.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.22.4.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc

# 2. 安装 gcc（cgo 编译 SQLite 需要）
sudo apt update && sudo apt install -y build-essential  # Ubuntu/Debian
# 或 sudo yum groupinstall -y "Development Tools"       # CentOS

# 3. 拉取代码
git clone https://github.com/Marovlo/bitcoin-puzzle.git
cd bitcoin-puzzle/coordinator

# 4. 编译（静态链接 SQLite，无运行时依赖）
CGO_ENABLED=1 go build -o puzzle_coordinator .

# 5. 验证
./puzzle_coordinator --help  # 如果没有 --help 也没关系，直接运行即可
```

## 配置（环境变量）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PUZZLE_NUM` | 71 | 目标 puzzle 编号 |
| `PORT` | 8080 | HTTP 监听端口 |
| `DB_PATH` | puzzle_pool.db | SQLite 数据库文件路径 |
| `PUZZLES_FILE` | ../puzzles.json | puzzles 数据文件路径 |

## 运行

```bash
# 前台运行（测试）
PUZZLE_NUM=71 PUZZLES_FILE=../puzzles.json ./puzzle_coordinator

# 后台运行（生产）
nohup env PUZZLE_NUM=71 PUZZLES_FILE=../puzzles.json PORT=8080 \
  ./puzzle_coordinator > coordinator.log 2>&1 &

# 或使用 systemd（推荐）
```

## Systemd 服务（推荐）

```bash
sudo tee /etc/systemd/system/puzzle-pool.service << 'EOF'
[Unit]
Description=Bitcoin Puzzle Pool Coordinator
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/bitcoin-puzzle/coordinator
Environment=PUZZLE_NUM=71
Environment=PUZZLES_FILE=/home/ubuntu/bitcoin-puzzle/puzzles.json
Environment=PORT=8080
ExecStart=/home/ubuntu/bitcoin-puzzle/coordinator/puzzle_coordinator
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable puzzle-pool
sudo systemctl start puzzle-pool
sudo systemctl status puzzle-pool
```

## 验证

```bash
# 健康检查
curl http://localhost:8080/health

# 查看状态
curl http://localhost:8080/api/stats | python3 -m json.tool

# 如果服务器有公网 IP，Worker 用这个连接：
# ./puzzle_worker --url http://YOUR_PUBLIC_IP:8080
```

## 防火墙

```bash
# 如果用 ufw
sudo ufw allow 8080/tcp

# 如果用 iptables
sudo iptables -A INPUT -p tcp --dport 8080 -j ACCEPT
```

## 监控

```bash
# 查看日志
tail -f coordinator.log
# 或
journalctl -u puzzle-pool -f

# 数据库大小
ls -lh puzzle_pool.db
```

## 注意事项

1. **不要**直接暴露在公网无认证。当前版本无鉴权，任何人都能获取和提交任务。
   生产环境建议加 nginx 反向代理 + Basic Auth 或 token。
2. 数据库文件 `puzzle_pool.db` 是持久化的。重启服务不会丢失进度。
3. 如需切换目标 puzzle，停止服务 → 删除 db → 修改 PUZZLE_NUM → 重启。
