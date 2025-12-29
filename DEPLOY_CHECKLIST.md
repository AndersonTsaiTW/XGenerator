# XGenerator Docker Deployment - Pre-deployment Checklist

## ⚠️ 部署前必讀

在執行 `deploy.sh` 之前，請完成以下檢查：

### 1. 修改 deploy.sh 中的 GitHub URL

打開 `deploy.sh`，找到這一行（第 13 行左右）：

```bash
GITHUB_REPO="https://github.com/YOUR_USERNAME/XGenerator.git"
```

**替換成您的實際 GitHub 用戶名**，例如：

```bash
GITHUB_REPO="https://github.com/johndoe/XGenerator.git"
```

### 2. 確認所有檔案都已推送到 GitHub

在本地執行：

```bash
git status
git push origin main
```

確認這些檔案都在 GitHub 上：
- ✅ `deploy.sh`
- ✅ `docker-compose.yml`
- ✅ `Dockerfile`
- ✅ `.env.example`
- ✅ `requirements.txt`
- ✅ `app/` 目錄

### 3. 確認 .gitignore 正確

以下檔案**不應該**出現在 GitHub：
- ❌ `.env` (包含密鑰)
- ❌ `data/` (包含用戶資料)
- ❌ `*.pem` (SSH 金鑰)

### 4. 準備好 OpenAI API Key

部署時需要填入 `.env`，請先準備好您的 OpenAI API key。

---

## 🚀 開始部署

完成以上檢查後，依照 **DEPLOY_README.md** 的步驟進行部署。
