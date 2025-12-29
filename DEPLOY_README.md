# 🚀 XGenerator EC2 部署快速指南

快速部署 XGenerator API 到 AWS EC2 (Amazon Linux 2023) 使用 Docker。

## 📋 前置需求

- ✅ EC2 instance (Amazon Linux 2023)
- ✅ DNS 設定: `api.xgenerators.net` → EC2 IP
- ✅ Security Group: 開放 22, 80, 443
- ✅ Let's Encrypt SSL 憑證已設定
- ✅ GitHub public repository

## ⚡ 一鍵部署

### 步驟 1: SSH 連接到 EC2

```bash
ssh -i your-key.pem ec2-user@api.xgenerators.net
```

### 步驟 2: 執行部署腳本

```bash
# 下載並執行
curl -O https://raw.githubusercontent.com/YOUR_USERNAME/XGenerator/main/deploy.sh
chmod +x deploy.sh
./deploy.sh
```

> ⚠️ 記得替換 `YOUR_USERNAME` 為您的 GitHub 用戶名

### 步驟 3: 設定環境變數

腳本會在沒有 `.env` 時暫停，此時：

```bash
cd ~/apps/XGenerator
nano .env
```

貼上（記得替換真實的 API key）:

```bash
OPENAI_API_KEY=sk-proj-your-actual-key-here
OPENAI_MODEL=gpt-3.5-turbo
REDIS_URL=redis://redis:6379/0
TESTING=false
```

儲存後再次執行：

```bash
docker-compose up -d --build
```

### 步驟 4: 設定 Nginx

```bash
sudo nano /etc/nginx/conf.d/api.xgenerators.net.conf
```

參考 `projectHint_deploy.txt` 中的完整 Nginx 設定。

測試並重新載入：

```bash
sudo nginx -t
sudo systemctl reload nginx
```

## ✅ 驗證部署

從本地測試：

```bash
# 健康檢查
curl https://api.xgenerators.net/health

# API 文檔
# 瀏覽器開啟: https://api.xgenerators.net/docs
```

## 📚 完整文檔

詳細的部署步驟、故障排除和維護指令請參考：
- **[projectHint_deploy.txt](./projectHint_deploy.txt)** - 完整部署指南

## 🔧 常用指令

```bash
# 查看狀態
cd ~/apps/XGenerator
docker-compose ps

# 查看日誌
docker-compose logs -f api

# 重啟服務
docker-compose restart

# 更新程式碼
git pull origin main
docker-compose up -d --build
```

## 🐛 故障排除

### API 無法訪問
```bash
# 檢查容器
docker-compose ps

# 檢查日誌
docker-compose logs api

# 檢查 Nginx
sudo systemctl status nginx
sudo nginx -t
```

### 502 Bad Gateway
```bash
# 測試內部連接
curl http://localhost:8000/health

# 檢查 port
docker-compose ps | grep 8000
```

更多問題請參考 `projectHint_deploy.txt` 的故障排除章節。

## 📦 Docker 服務

此部署包含 3 個 Docker 容器：

| 容器 | 用途 | Port |
|------|------|------|
| `xgenerator_api` | FastAPI 服務 | 8000 |
| `xgenerator_worker` | Celery 背景任務 | - |
| `xgenerator_redis` | 訊息佇列 | 6379 |

## 🔒 安全性

- ✅ HTTPS 強制啟用
- ✅ API Key 認證
- ✅ Rate limiting
- ✅ Security headers (Nginx)
- ✅ 敏感 ports 不對外開放

## 📞 支援

遇到問題請檢查：
1. `projectHint_deploy.txt` - 完整指南
2. Docker logs: `docker-compose logs -f`
3. GitHub Issues

---

**部署文檔版本**: v1.0  
**最後更新**: 2025-12-29
