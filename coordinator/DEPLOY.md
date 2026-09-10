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
| `EMAIL_TO` | 359207423@qq.com | 各类通知收件人（逗号分隔可多个） |
| `SMTP_HOST` | smtp.qq.com | SMTP 服务器 |
| `SMTP_PORT` | 587 | SMTP 端口 |
| `SMTP_USER` | （空） | SMTP 账号；为空则不发邮件，仅记日志 |
| `SMTP_PASS` | （空） | SMTP 授权码 |
| `REPORT_AT_UTC` | 09:00 | 日报发送时刻（UTC，`HH:MM`） |
| `SOLVE_CHECK_MINUTES` | 10 | 轮询目标地址是否已被解出的间隔（分钟） |

## 告警行为

三类事件都会发邮件，但保证方式不同：

| 事件 | 触发条件 | 及时性 | 兜底机制 |
|------|---------|--------|---------|
| **找到密钥** | worker 提交 `found=true` 的**当下** | 秒级 | 日报会一直重复列出"未确认送达"的 key，直到发信成功 |
| **每日日报** | 每天 `REPORT_AT_UTC`（默认 09:00 UTC） | 定点；每次重新对齐墙钟，长期运行不漂移 | — |
| **目标被解出** | 每 `SOLVE_CHECK_MINUTES` 轮询链上余额 | 默认 10 分钟内 | 自动切到下一个未解出的 puzzle，并补发一封确认邮件 |

实现要点：

- `found_keys` 表有 `notified` 列，**告警邮件确认发出后才置 1**。因此"找到了却没收到邮件"最多延迟到下一次日报，不会永久丢失；
- 日报里"未确认送达"的 key **不按当前 puzzle 过滤**——否则一旦自动换题，刚找到的 key 就再也不会出现在邮件里；
- 通知走独立 goroutine，SMTP 阻塞不会卡住任务分配（分配路径的互斥锁内绝不发信）；
- 网络查询失败一律按"未解出"处理，绝不误报解出；切换新 puzzle 前先校验地址，校验失败则保持原目标并告警。

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

实际部署使用 `puzzle-coordinator.service`，以下为当前生产配置：

```bash
sudo tee /etc/systemd/system/puzzle-coordinator.service << 'EOF'
[Unit]
Description=Bitcoin Puzzle Pool Coordinator
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/bitcoin-puzzle/coordinator
Environment=SMTP_HOST=smtp.qq.com
Environment=SMTP_PORT=587
Environment=SMTP_USER=your_email@qq.com
Environment=SMTP_PASS=your_smtp_auth_code
Environment=EMAIL_TO=your_email@qq.com
ExecStart=/root/bitcoin-puzzle/coordinator/puzzle_coordinator
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable puzzle-coordinator
sudo systemctl start puzzle-coordinator
sudo systemctl status puzzle-coordinator
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
# 查看日志（systemd）
journalctl -u puzzle-coordinator -f

# 若用 nohup 前台/后台启动，日志在文件中：
tail -f coordinator.log

# 数据库大小
ls -lh puzzle_pool.db
```

## 注意事项

1. **不要**直接暴露在公网无认证。当前版本无鉴权，任何人都能获取和提交任务。
   生产环境建议加 nginx 反向代理 + Basic Auth 或 token。
2. 数据库文件 `puzzle_pool.db` 是持久化的。重启服务不会丢失进度。
3. 如需切换目标 puzzle，停止服务 → 删除 db → 修改 PUZZLE_NUM → 重启。
4. Monitor 会每小时调用 mempool.space / blockstream.info 检查 puzzle 是否已被网络解出。
   国内服务器访问海外 API 偶发超时属正常现象，代码已做 IPv4 强制 + 多端点兜底 + 重试；
   若日志中仍频繁出现 `API error`，通常是临时网络抖动，不影响主流程。
5. 重启 / 更新二进制后，systemd 会自动拉起服务（`Restart=always`），
   进行中的任务会在 10 分钟后被自动回收并重新分配，无需手动处理。
